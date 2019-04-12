import os
import pdb
import sys
import copy
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms


import losses
import eval_new
import retinanet_roi
from anchors import Anchors
from config import get_config
from utils import AverageMeter
from dataloader import GetDataset, collater, Resizer, Augmenter


print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(config):
    # set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    # create folder for model
    newpath = './models/' + config.model_date
    if config.save_model:
        os.makedirs(newpath)

    # Create the data loaders
    if config.csv_train is None:
        raise ValueError(
            'Must provide --csv_train when training on csv,')

    if config.csv_classes is None:
        raise ValueError(
            'Must provide --csv_classes when training on csv,')

    train_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'train'))
    dataset_train = GetDataset(train_file=config.csv_train, class_list=config.csv_classes,
                               transform=transforms.Compose([Augmenter(), Resizer()]), dataset=train_dataset, seed=0)
    dataloader_train = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True,
                                  num_workers=1, collate_fn=collater)

    if config.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        valid_dataset = datasets.ImageFolder(os.path.join(config.data_dir, 'valid'))
        dataset_val = GetDataset(
            train_file=config.csv_val, class_list=config.csv_classes, transform=transforms.Compose([Resizer()]), dataset=valid_dataset, seed=0)

    # Create the model
    if config.depth == 18:
        retinanet = retinanet_roi.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif config.depth == 34:
        retinanet = retinanet_roi.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif config.depth == 50:
        retinanet = retinanet_roi.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif config.depth == 101:
        retinanet = retinanet_roi.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif config.depth == 152:
        retinanet = retinanet_roi.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if config.use_gpu:
        retinanet = retinanet.cuda()

    retinanet = torch.nn.DataParallel(retinanet).cuda()

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, verbose=True)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    best_valid_map = 0
    counter = 0
    batch_size = config.batch_size

    for epoch_num in range(config.epochs):
        print('\nEpoch: {}/{}'.format(epoch_num+1, config.epochs))
        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        train_batch_time = AverageMeter()
        train_losses = AverageMeter()
        tic = time.time()
        with tqdm(total=len(dataset_train)) as pbar:
            for iter_num, data in enumerate(dataloader_train):
                # try:
                optimizer.zero_grad()
                siamese_loss, classification_loss, regression_loss = retinanet(
                    [data['img'].cuda().float(), data['annot'], data['pair'].cuda().float()])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss + siamese_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    retinanet.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss))

                toc = time.time()
                train_losses.update(
                    float(loss), batch_size)
                train_batch_time.update(toc-tic)
                tic = time.time()

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f}".format(
                            train_batch_time.val,
                            train_losses.val,
                        )
                    )
                )
                pbar.update(batch_size)

                del classification_loss
                del regression_loss
                del siamese_loss

                # except Exception as e:
                #     print('Training error: ', e)
                #     continue

        if config.csv_val is not None:
            print('Evaluating dataset')
            mAP, correct = eval_new.evaluate(dataset_val, retinanet)
            
            # is_best = mAP[0][0] > best_valid_map
            # best_valid_map = max(mAP[0][0], best_valid_map)
            is_best = correct > best_valid_map
            best_valid_map = max(correct, best_valid_map)
            if is_best:
                counter = 0
            else:
                counter += 1
                if counter > 3:
                    print("[!] No improvement in a while, stopping training.")
                    break

        scheduler.step(np.mean(epoch_loss))
        if is_best and config.save_model:
            torch.save(retinanet.state_dict(
            ), './models/{}/best_retinanet.pt'.format(config.model_date))
        if config.save_model:
            torch.save(retinanet.state_dict(
            ), './models/{}/{}_retinanet_{}.pt'.format(config.model_date, config.depth, epoch_num))
        
        msg = "train loss: {:.3f} - val map: {:.3f} - val acc: {:.3f}%"
        print(msg.format(train_losses.avg, mAP[0][0], (100. * correct)/len(dataset_val)))

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)