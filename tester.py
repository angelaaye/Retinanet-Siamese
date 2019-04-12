import os
import sys
import cv2
import time
import json
import copy
import math
import random
import argparse
import numpy as np
from PIL import Image
from random import Random
from numpy import genfromtxt
import matplotlib.pyplot as plt

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import datasets, models, transforms

import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

import eval_new
import model
from utils import AverageMeter
from dataloader import GetDataset, collater, Resizer, Augmenter

arg_lists = []
parser = argparse.ArgumentParser(description='RetinaNet-Siamese Network')

def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--test_trials', type=int, default=10000,
                      help='# of test 1-shot trials')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--plot', type=str2bool, default=True,
                      help="Whether to visualize data")

valid_arg = add_argument_group('Validate')
valid_arg.add_argument('--csv_classes', type=str, default='testmap.csv', help='Path to file containing class list (see readme)')
valid_arg.add_argument('--csv_val', type=str, default='testlabel.csv', help='Path to file containing validation annotations (optional, see readme)')
valid_arg.add_argument('--model_date', help='Path to model (.pt) file.')
valid_arg.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
valid_arg.add_argument('--thres', help='IoU threshold, between 0.5 and 0.95 inclusive', type=float, default=0.5)
valid_arg.add_argument('--eval_map', help='Whether to evaluate mAP', type=str2bool, default=False)
valid_arg.add_argument('--sim_thres', help='Similarity threshold', type=float, default=0.5)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def main(config):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    model_path = './models/'+ config.model_date + '/best_retinanet.pt'
    
    val_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'test'))
    dataset_val = GetDataset(train_file=config.csv_val, class_list=config.csv_classes, transform=transforms.Compose([Resizer()]), dataset=val_dataset, seed=0)

    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=1, collate_fn=collater)

    # Create the model
    if config.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
    elif config.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
    elif config.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    elif config.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    elif config.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		

    use_gpu = True

    if use_gpu:
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    retinanet.load_state_dict(torch.load(model_path))

    retinanet.eval()
    
    correct = 0
    if config.eval_map:
        mAP = eval_new.evaluate(dataset_val, retinanet, iou_threshold=config.thres)

    avg_time = AverageMeter()
    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            
            st = time.time()

            scores, classification, transformed_anchors, similarity = retinanet([data['img'].cuda().float(), data['pair'].float()])
            
            avg_time.update(time.time()-st)

            scores = scores.cpu().numpy()
            idxs = np.where(scores>0.5)

            img = np.array(255 * data['img'][0, :, :, :]).copy()
            img[img<0] = 0
            img[img>255] = 255
            img = np.transpose(img, (1, 2, 0))

            pair = np.array(255 * data['pair'][0, :, :, :]).copy()
            pair[pair<0] = 0
            pair[pair>255] = 255
            pair = np.transpose(pair, (1, 2, 0))

            if config.plot:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
                pair = cv2.cvtColor(pair.astype(np.uint8), cv2.COLOR_BGR2RGB)

            max_sim = 0.0
            annot = dataset_val.get_annot(idx)
            annot = annot['annot']
            bbox_true = annot[annot[:, 5] == 1, :4]
            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                sim = similarity[j, 0].item()
                if sim > max_sim:
                    max_sim = sim
                    bbox_est = np.asarray([[x1, y1, x2, y2]])
                if config.plot:
                    draw_caption(img, (x1, y1, x2, y2), str(round(sim, 2)))
                    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)

            iou = compute_overlap(bbox_true, bbox_est/data['scale'][0])
            if iou.any() > config.thres and max_sim > config.sim_thres:
                correct += 1
            max_sim = 0

            # print('Iter = {}, number correct = {}'.format(idx+1, correct)) 
            if config.plot:   
                cv2.imshow('pair', pair)
                cv2.imshow('img', img)
                cv2.waitKey(0)  # press q to quit, any other to view next image
    print('Final Accuracy is {}'.format(correct/config.test_trials))
    print('Average time: {}'.format(avg_time.val))

def compute_overlap(a, b):
    ''' Compute intersection between boxes in a and the current box b'''
    iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    ua = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def draw_caption(image, box, caption):

    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
    

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)