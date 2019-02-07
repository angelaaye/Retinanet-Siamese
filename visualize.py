import os
import pdb
import sys
import cv2
import copy
import time
import argparse
import numpy as np

import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

import model
from dataloader import CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, UnNormalizer, Normalizer


print('CUDA available: {}'.format(torch.cuda.is_available()))

def main(args=None):
	parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
	parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
	parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
	parser.add_argument('--model', help='Path to model (.pt) file.')
	parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=18)
	parser.add_argument('--use_gpu', help='Whether to run on GPU', type=bool, default=True)

	parser = parser.parse_args(args)

	dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes, transform=transforms.Compose([Resizer()]))

	#sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
	dataloader_val = DataLoader(dataset_val, batch_size=1, num_workers=3, collate_fn=collater)#, batch_sampler=sampler_val)

	# Create the model
	if parser.depth == 18:
		retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 34:
		retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 50:
		retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 101:
		retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
	elif parser.depth == 152:
		retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
	else:
		raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')		


	if parser.use_gpu:
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	retinanet.load_state_dict(torch.load(parser.model))

	retinanet.eval()

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			print('Elapsed time: {}'.format(time.time()-st))
			scores = scores.cpu().numpy()
			idxs = np.where(scores>0.5)
			img = np.array(255 * data['img'][0, :, :, :]).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			cv2.imshow('img', img)
			cv2.waitKey(0)  # press q to quit, any other to view next image

def draw_caption(image, box, caption):

	b = np.array(box).astype(int)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
	cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


if __name__ == '__main__':
	main()