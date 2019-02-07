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
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import model
import losses
import csv_eval
from anchors import Anchors
from utils import AverageMeter
from dataloader import CSVDataset, collater, AspectRatioBasedSampler, Resizer, Augmenter


print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

    parser = argparse.ArgumentParser(
        description='Simple training script for training a RetinaNet network.')

    parser.add_argument(
        '--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument(
        '--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument(
        '--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument(
        '--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument(
        '--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--model_date', type=str, default=time.strftime(
        '%d-%m-%Y-%H-%M-%S'), help='Model date used for unique checkpointing')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='Model date used for unique checkpointing')

    parser = parser.parse_args(args)

    # set seed for reproducibility
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # create folder for model
    newpath = './models/' + parser.model_date
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    # Create the data loaders
    if parser.csv_train is None:
        raise ValueError(
            'Must provide --csv_train when training on csv,')

    if parser.csv_classes is None:
        raise ValueError(
            'Must provide --csv_classes when training on csv,')

    dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                               transform=transforms.Compose([Augmenter(), Resizer()]))

    if parser.csv_val is None:
        dataset_val = None
        print('No validation annotations provided.')
    else:
        dataset_val = CSVDataset(
            train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Resizer()]))

    #sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True,
                                  num_workers=3, collate_fn=collater)  # , batch_sampler=sampler)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(
            num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(
            num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError(
            'Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.use_gpu:
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
    batch_size = 2

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []

        train_batch_time = AverageMeter()
        train_losses = AverageMeter()
        tic = time.time()
        with tqdm(total=len(dataset_train)) as pbar:
            for iter_num, data in enumerate(dataloader_train):
                try:
                    optimizer.zero_grad()

                    classification_loss, regression_loss = retinanet(
                        [data['img'].cuda().float(), data['annot']])

                    classification_loss = classification_loss.mean()
                    regression_loss = regression_loss.mean()
                    loss = classification_loss + regression_loss

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

                except Exception as e:
                    print(e)
                    continue

        if parser.csv_val is not None:
            print('Evaluating dataset')
            mAP = csv_eval.evaluate(dataset_val, retinanet)
            
            is_best = mAP[0][0] > best_valid_map
            best_valid_map = max(mAP[0][0], best_valid_map)
            if is_best:
                counter = 0
            else:
                counter += 1
                if counter > 3:
                    print("[!] No improvement in a while, stopping training.")
                    break

        scheduler.step(np.mean(epoch_loss))
        if is_best:
            torch.save(retinanet.state_dict(
            ), './models/{}/best_retinanet.pt'.format(parser.model_date))
            torch.save(retinanet.state_dict(
            ), './models/{}/retinanet_{}.pt'.format(parser.model_date, epoch_num))

    # retinanet.eval()
    # torch.save(retinanet.state_dict(
    # ), './models/{}/model_final_{}.pt'.format(parser.model_date, epoch_num))


if __name__ == '__main__':
    main()
