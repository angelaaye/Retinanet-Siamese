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

import model
import csv_eval
from siamese_network import SiameseNetwork
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter

arg_lists = []
parser = argparse.ArgumentParser(description='Retinanet Network')

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
data_arg.add_argument('--way', type=int, default=1,
                      help='Ways in the 1-shot trials')
data_arg.add_argument('--num_workers', type=int, default=1,
                      help='# of subprocesses to use for data loading. If using CUDA, num_workers should be set to `1`')


# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=False,
                       help='Whether to train or test the model')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--model_date', type=str, default=time.strftime('%d-%m-%Y-%H-%M-%S'),
                      help='Model date used for unique checkpointing')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--data_dir', type=str, default='./data/changed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--plot_dir', type=str, default='./plots/',
                      help='Directory in which plots are stored')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/',
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./logs/',
                      help='Directory in which logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')

valid_arg = add_argument_group('Validate')
valid_arg.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
valid_arg.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
valid_arg.add_argument('--model', help='Path to model (.pt) file.')
valid_arg.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def main(config):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    test_file = config.csv_val
    my_data = genfromtxt(test_file, delimiter=',', dtype=None, encoding="utf8")

    data_loader = get_test_loader(
        config.data_dir, config.way,
        10000, 0, my_data
    )
    img_labels = [i[-1] for i in my_data]
    try:
        layer_hyperparams = load_config(config)
    except FileNotFoundError:
        print("[!] No previously saved config. Set resume to False.")
        return

    trainer = Trainer(config, layer_hyperparams)

    dataset_val = CSVDataset(train_file=config.csv_val, class_list=config.csv_classes, transform=transforms.Compose([Resizer()]))

    #sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=3, collate_fn=collater)#, batch_sampler=sampler_val)

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
    retinanet.load_state_dict(torch.load(config.model))

    retinanet.eval()
    
    transform = transforms.ToTensor()

    correct = 0

    mAP = csv_eval.evaluate(dataset_val, retinanet)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            # st = time.time()
            scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            # print('Elapsed time: {}'.format(time.time()-st))
            scores = scores.cpu().numpy()
            idxs = np.where(scores>0.5)
            img = np.array(255 * data['img'][0, :, :, :]).copy()

            img[img<0] = 0
            img[img>255] = 255

            img = np.transpose(img, (1, 2, 0))
            PIL_img = Image.fromarray(np.uint8(img))

            # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            max_sim = 0
            if idx % config.way == 0:
                index = img_labels[idx*5:idx*5+5*config.way].index(data_loader[idx//config.way][1])
                bbox_true = np.asarray([list(my_data[idx*5+index])[1:5]])

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                cropped_img = PIL_img.crop((x1, y1, x2, y2))
                cropped_img = cropped_img.resize((105, 105))
                cropped_img = cropped_img.convert('L')
                cropped_img = transform(cropped_img)
                cropped_img = cropped_img.unsqueeze(0)
                similarity = trainer.test(cropped_img, data_loader[idx//config.way][0])
                if similarity.item() > max_sim:
                    max_sim = similarity.item()
                    bbox_est = np.asarray([[x1, y1, x2, y2]])
                # label_name = dataset_val.labels[int(classification[idxs[0][j]])]

                # draw_caption(img, (x1, y1, x2, y2), str(round(similarity.item(),2)))
                # cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            # input_img = data_loader[idx//config.way][0].squeeze()
            # input_img = np.array(input_img.unsqueeze(2))
            # cv2.imshow('input', input_img)
            if (idx+1) % config.way == 0:
                iou = compute_overlap(bbox_true, bbox_est/data['scale'][0])
                if iou > 0.5:
                    correct += 1
                max_sim = 0
            print('Iter = {}, number correct = {}'.format(idx+1, correct))    
            # cv2.imshow('img', img)
            # cv2.waitKey(0)  # press q to quit, any other to view next image
    print('Final Accuracy is {}'.format(correct/config.test_trials))

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
    
def get_model_date(config):
    model_date = config.model_date
    # error_msg = "[!] model number must be >= 1."
    # assert model_date > 0, error_msg
    return 'exp_' + str(model_date)

def load_config(config):
    model_date = get_model_date(config)
    model_dir = os.path.join(config.ckpt_dir, model_date)
    filename = 'params.json'
    param_path = os.path.join(model_dir, filename)
    params = json.load(open(param_path))
    print("[*] Loaded layer hyperparameters.")
    wanted_keys = [
        'layer_end_momentums', 'layer_init_lrs', 'layer_l2_regs'
    ]
    hyperparams = dict((k, params[k]) for k in wanted_keys if k in params)
    return hyperparams

def get_test_loader(data_dir, way, trials, seed, data_file):
    """
    Utility function for loading and returning a multi-process iterator
    over the Omniglot test dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    test_dir = os.path.join(data_dir, 'test')
    test_dataset = datasets.ImageFolder(test_dir)
    test_dataset = OmniglotTest(test_dataset, data_file, trials, way, seed=0)
    return test_dataset

class OmniglotTest(Dataset):
    def __init__(self, dataset, data_file, trials, way, seed):
        super(OmniglotTest, self).__init__()
        self.dataset = dataset
        self.data_file = data_file
        self.trials = trials
        self.way = way
        self.transform = transforms.ToTensor()
        self.seed = seed
        self.img1_labels = [i[-1] for i in self.data_file]

    def __len__(self):
        return self.trials//self.way

    def __getitem__(self, index):
        self.rng = Random(self.seed + index)

        # every 'way' number of images, have one pair from same class
        # e.g. way = 20, then pairs number 0, 20, 40... will be from same class
        # idx = index % self.way   
        # generate image pair from same class

        while True:
            img2 = self.rng.choice(self.dataset.imgs)
            if img2[1] in self.img1_labels[index*5*self.way:index*5*self.way+5*self.way]:
                break
        # if idx == 0:
        #     while True:
        #         img2 = self.rng.choice(self.dataset.imgs)
        #         if img2[1] in self.img1_labels[index*5:index*5+5]:
        #             break
        # # generate image pair from different class
        # else:
        #     while True:
        #         img2 = self.rng.choice(self.dataset.imgs)
        #         if img2[1] not in self.img1_labels[index*5:index*5+5]:
        #             break
        label = img2[1]
        img2 = Image.open(img2[0])
        img2 = img2.convert('L')
        img2 = self.transform(img2)
        img2 = img2.unsqueeze(0)
        return img2, label

class Trainer(object):
    def __init__(self, config, layer_hyperparams):
        self.config = config
        self.layer_hyperparams = layer_hyperparams
        self.model = SiameseNetwork()
        if config.use_gpu:
            self.model.cuda()
        # model params
        self.num_params = sum(
            [p.data.nelement() for p in self.model.parameters()]
        )
        self.model_date = get_model_date(config)
        self.num_layers = len(list(self.model.children()))

        print('[*] Number of model parameters: {:,}'.format(self.num_params))

        # path params
        self.ckpt_dir = os.path.join(config.ckpt_dir, self.model_date)
        self.logs_dir = os.path.join(config.logs_dir, self.model_date)

        # misc params
        self.resume = config.resume
        self.use_gpu = config.use_gpu
        self.dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        # optimization params
        self.best = config.best

        # grab layer-wise hyperparams
        self.init_lrs = self.layer_hyperparams['layer_init_lrs']
        self.end_momentums = self.layer_hyperparams['layer_end_momentums']
        self.l2_regs = self.layer_hyperparams['layer_l2_regs']

        # set global learning rates and momentums
        self.lrs = self.init_lrs

        self.optimizer = optim.Adam(
            self.model.parameters(), lr=3e-4, weight_decay=6e-5,
        )
        # load best model
        self.load_checkpoint(best=self.best)

        # switch to evaluate mode
        self.model.eval()

    def test(self, x1, x2):
        if self.use_gpu:
            x1, x2 = x1.cuda(), x2.cuda()

        # compute log probabilities
        log_probas = self.model(x1, x2)

        # get similarity value
        sim = log_probas.data.max(0)[0][0]

        return sim

    def load_checkpoint(self, best=False):
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = 'model_ckpt.tar'
        if best:
            filename = 'best_model_ckpt.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)