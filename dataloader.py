import os
import sys
import csv
import torch
import random
import numpy as np
from PIL import Image
import skimage.transform
from random import Random
from numpy import genfromtxt
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import Sampler
from torchvision import transforms, utils, datasets

class GetDataset(Dataset):

    def __init__(self, train_file, class_list, transform, dataset, seed):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform
        self.pair_transform = transforms.ToTensor()
        self.dataset = dataset
        self.seed = seed
        self.data_file = genfromtxt(self.train_file, delimiter=',', dtype=None, encoding="utf8")
        self.img1_labels = [i[-1] for i in self.data_file]

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=',')) # result[class_name] = class_id
        except ValueError as e:
            raise ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e))

        self.labels = {}
        for key, value in self.classes.items():  # key = class_name, value = class_id
            # labels[class_id] = class_name, classes[class_name] = class_id
            self.labels[value] = key   

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                # image_data[img_file] = x1, x2, y1, y2, class_name
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e))
        self.image_names = list(self.image_data.keys())  # filenames

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise ValueError(fmt.format(e))

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')


    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise ValueError('line {}: format should be \'class_name,class_id\''.format(line))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result


    def __len__(self):
        # return len(self.image_names)
        return 10

    def __getitem__(self, idx):
        img = self.load_image(idx)  # img = np.array/255
        annot = self.load_annotations(idx)  # annot = [x1, y1, x2, y2, class_id, class_label]
        pair, label = self.get_pair(idx, annot)
        annot[:, 5] = annot[:, 5]==label
        sample = {'img': img, 'annot': annot, 'pair': pair}  
        if self.transform:
            sample = self.transform(sample)

        return sample
    
    def get_annot(self, idx):
        annot = self.load_annotations(idx)
        pair, label = self.get_pair(idx, annot)
        annot[:, 5] = annot[:, 5]==label
        sample = {'annot': annot}
        return sample 

    def get_pair(self, index, annot):
        self.rng = Random(self.seed + index)
        while True:
            img2 = self.rng.choice(self.dataset.imgs)
            if img2[1] in annot[:, 5]:
            # if img2[1] in self.img1_labels[index*5:index*5+5]:
                break
        label = img2[1]
        img2 = Image.open(img2[0])
        # if img2.mode != 'RGB':
        #     img2 = img2.convert('RGB')
        img2 = img2.convert('L')
        # img2 = self.pair_transform(img2)
        # img2 = img2.unsqueeze(0)
        img2 = np.asarray(img2)
        return img2.astype(np.float32)/255.0, label

    def load_image(self, image_index):
        img = Image.open(self.image_names[image_index])
        # img = img.convert('L')
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = np.asarray(img)
        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]  # image_data[img_filename] = [x1, x2, y1, y2, class_name]
        annotations     = np.zeros((0, 6))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation        = np.zeros((1, 6))
            
            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotation[0, 5] = a['class']
            annotations      = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise ValueError('line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line))

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    pairs = [s['pair'] for s in data]
    # labels = [s['label'] for s in data] 

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    padded_pairs = torch.zeros(batch_size, np.array([int(s.shape[0]) for s in pairs]).max(), np.array([int(s.shape[1]) for s in pairs]).max(), 1)
   
    for i in range(batch_size):
        img = imgs[i]
        pair = pairs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
        padded_pairs[i, :int(pair.shape[0]), :int(pair.shape[1]), :] = pair

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 6)) * -1  # batch_size by max_num_annots per image (5) by 6
        # label_padded = torch.ones((len(annots), 1, 1)) * -1 
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
                    # try:
                    #     label_padded[idx, :, :] = labels[idx]
                    # except:
                    #     continue
    else:
        annot_padded = torch.ones((len(annots), 1, 6)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)   # change order so it's i, ch, w, h
    padded_pairs = padded_pairs.permute(0, 3, 1, 2)
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'pair': padded_pairs}

class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots, pair = sample['img'], sample['annot'], sample['pair']
        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        # image = np.stack((image,)*1, axis=-1)

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        pair = np.stack((pair,)*1, axis=-1)
        # rows, cols, cns = pair.shape   
        # smallest_side = min(rows, cols)
        
        # min_side = 160 - 32
        # scale_pair = min_side/smallest_side
        # pair = skimage.transform.resize(pair, (int(round(rows*scale_pair)), int(round((cols*scale_pair)))))

        # rows, cols, cns = pair.shape

        # pad_w = 32 - rows%32
        # pad_h = 32 - cols%32

        # new_pair = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        # new_pair[:rows, :cols, :] = pair.astype(np.float32)

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale, 'pair': torch.from_numpy(pair)}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots, pair = sample['img'], sample['annot'], sample['pair']
            image = image[:, ::-1, :]

            rows, cols, cns = image.shape

            x1 = annots[:, 0].copy() #x1
            x2 = annots[:, 2].copy() #x2
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'pair': pair}

        return sample
