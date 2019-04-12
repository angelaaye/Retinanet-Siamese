import os
import cv2
import csv
import torch
import random
import argparse
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage



def main(args=None):
    parser = argparse.ArgumentParser(description='Simple script for generating an Omniglot detection dataset.')
    parser.add_argument('--data_dir', help='Path to folder containing train, valid, and test folders', type=str, default='./data/changed')
    parser.add_argument('--save_dir', help='Path to folder containing train, valid, and test folders for saving', type=str, default='./obj_detection')
    parser.add_argument('--background_dir', help='Path to folder containg background images', type=str, default=None)
    parser.add_argument('--num_train', help='Number of training images to generate', type=int, default=80000)
    parser.add_argument('--num_valid', help='Number of validation images to generate', type=int, default=10000)
    parser.add_argument('--num_test', help='Number of test images to generate', type=int, default=10000)
    parser = parser.parse_args(args)

    train_dir = os.path.join(parser.data_dir, 'train')
    valid_dir = os.path.join(parser.data_dir, 'valid')
    test_dir = os.path.join(parser.data_dir, 'test')
    # background_dir = os.path.join('./background')
    # train_dir = './test'
    train_dataset = datasets.ImageFolder(train_dir)  
    valid_dataset = datasets.ImageFolder(valid_dir)
    test_dataset = datasets.ImageFolder(test_dir)
    bg_dir = False
    if parser.background_dir is not None:
        bg_dir = True
    

    # set seed for reproducibility
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    trans = transforms.Compose([
        transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), shear=2, fillcolor=255),
        # transforms.ToTensor(),
    ])

    newpath = './obj_detection/'
    if not os.path.exists(newpath):
        os.makedirs(newpath)
        os.makedirs(newpath+'train/')
        os.makedirs(newpath+'valid/')
        os.makedirs(newpath+'test/')

    # write to csv file
    train_data = image_generation(parser, parser.num_train, trans, train_dataset, os.path.join(parser.save_dir, 'train/'), bg_dir)
    train_file = './trainlabel.csv'
    write_to_csv(train_file, train_data, True)

    valid_data = image_generation(parser, parser.num_valid, trans, valid_dataset, os.path.join(parser.save_dir, 'valid/'), bg_dir)
    valid_file = './validlabel.csv'
    write_to_csv(valid_file, valid_data, True)

    test_data = image_generation(parser, parser.num_test, trans, test_dataset, os.path.join(parser.save_dir, 'test/'), bg_dir)
    test_file = './testlabel.csv'
    write_to_csv(test_file, test_data, True)

    train_map_file = './trainmap.csv'
    valid_map_file = './validmap.csv'
    test_map_file = './testmap.csv'
    train_map = class_mapping(train_dataset)
    valid_map = class_mapping(valid_dataset)
    test_map = class_mapping(test_dataset)
    write_to_csv(train_map_file, train_map, True)
    write_to_csv(valid_map_file, valid_map, True)
    write_to_csv(test_map_file, test_map, True)
    # Display bounding boxes around characters for one image
    img = 'test.png'
    show_bounding_boxes(img, train_data)

def image_generation(parser, num_pics, trans, dataset, save_dir, bg_dir=False):
    csv_data = []  # csv data for training labels
    count = 1 # initialize count for unique image filenames
    if bg_dir: 
        bg_dataset = datasets.ImageFolder(parser.background_dir)
    for i in range(num_pics):
        if bg_dir:
            bg = random.choice(bg_dataset.imgs)
            bg = Image.open(bg[0])
            background = bg.convert('L')
            background = ImageEnhance.Contrast(background).enhance(0.8)
        else:
            background = Image.new('1', (500, 500), (255))  # L represents 8 bit pixel, black and white mode
        bg_w, bg_h = background.size
        num_chars = np.random.randint(2, 6)
        num_chars = 5
        for j in range (num_chars):
            im = random.choice(dataset.imgs)
            image1 = Image.open(im[0])
            # image1 = image1.convert('L')
            image1 = trans(image1)

            image = image1.load()
            w, h = image1.size
            for y in range(w):
                for x in range(h):
                    if image[x, y] < 200:
                        if y != h-1:
                            y0 = y+1
                            break
            for x in range(w):
                for y in range(h):
                    if image[x, y] < 200:
                        if x != h-1:
                            x0 = x+1
                            break
            for y in range(h-1, -1, -1):
                for x in range(w):
                    if image[x, y] < 200:
                        if y != 0:
                            y1 = y-1
                            break
            for x in range(h-1, -1, -1):
                for y in range(h):
                    if image[x, y] < 200:
                        if x != 0:
                            x1 = x-1
                            break
            a = random.randint(-x1, bg_w-x0-1)   #  ensure corners of original image won't be cropped out
            b = random.randint(-y1, bg_h-y0-1)
            if j != 0:
                inter = compute_overlap(pts, np.asarray([[0+a, 0+b, 105+a, 105+b]]))
                while np.count_nonzero(inter) > 0:
                    a = random.randint(-x1, bg_w-x0-1) 
                    b = random.randint(-y1, bg_h-y0-1)
                    inter = compute_overlap(pts, np.asarray([[0+a, 0+b, 105+a, 105+b]]))
            background.paste(im=image1, box=(a, b))
            new_file = save_dir+str(count)+'.png'
            # csv_data.append([new_file, x1+a, y1+b, x0+a, y0+b, im[0].split('/')[-2]])
            csv_data.append([new_file, x1+a, y1+b, x0+a, y0+b, im[1]])
            if j == 0:
                pts = np.asarray([[x1+a, y1+b, x0+a, y0+b]])
            else:
                pts = np.append(pts, np.asarray([[x1+a, y1+b, x0+a, y0+b]]), axis=0)
        background.save(new_file)
        background.save('test.png')
        background.show()
        count += 1
    return csv_data

def compute_overlap(a, b):
    ''' Compute intersection between boxes in a and the current box b'''
    iw = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    ih = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    intersection = iw * ih

    return intersection 

def write_to_csv(f, data, overwrite):
    if overwrite:
        csv_file = open(f, 'w')  
        with csv_file:  
            writer = csv.writer(csv_file)
            writer.writerows(data)

def class_mapping(image_folder):
    data = []     
    count = -1
    for i in range(len(image_folder.imgs)):
        if image_folder.imgs[i][1] > count:
            data.append([image_folder.imgs[i][1], 0])
            # data.append([image_folder.imgs[i][0].split('/')[-2], image_folder.imgs[i][-1]])
            count += 1
    return data

def show_bounding_boxes(img, csv_data):
    # background.save(img)
    image = cv2.imread(img)
    clone = image.copy()    #copy the image
    box_img = "image"
    cv2.namedWindow(box_img)    #creates a window   
    
    while True:
        #display the image and wait for a keypress
        cv2.imshow(box_img, clone)
        for i in range (len(csv_data)):
            cv2.rectangle(clone, (csv_data[i][1], csv_data[i][2]), (csv_data[i][3], csv_data[i][4]), (0, 255, 0), 1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):     #hit 'r' to replot image
            clone = image.copy()
        elif key == ord("c"):   #hit 'c' to confirm selection
            break
    cv2.destroyWindow(box_img)  #close the windows


if __name__ == '__main__':
    main()


'''
# from torch.utils.data.dataset import Dataset
# from torch.utils.data import DataLoader
# from numpy import genfromtxt
# my_data = genfromtxt(train_file, delimiter=',', dtype=None, encoding="utf8")
# my_data[0][0] # obj detection filename
# my_data[0][-1] # character name
# my_data2 = genfromtxt(map_file, delimiter=',', dtype=None, encoding="utf8")
# [item for item in my_data2 if item[0] == my_data[0][-1]][0][1] # returns the corresponding class number

def get_train_valid_loader(data_dir, data_dir2, csvfile, batch_size, num_train, augment, way,
                           trials, shuffle, seed, num_workers, pin_memory):
    """
    Utility function for loading and returning train and valid multi-process
    iterators over the Omniglot dataset.

    If using CUDA, num_workers should be set to `1` and pin_memory to `True`.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to load the augmented version of the train dataset.
    - num_workers: number of subprocesses to use when loading the dataset. Set
      to `1` if using GPU.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      `True` if using GPU.
    """
    train_dir = os.path.join(data_dir, 'train')
    # valid_dir = os.path.join(data_dir, 'valid')

    # create a dataset that arranges folder names in alphabetical order
    # root is train_dir
    train_dataset = datasets.ImageFolder(train_dir)
    detection_dataset = datasets.ImageFolder(data_dir2)   
    # create training pair instances
    train_dataset = OmniglotTrain(train_dataset, detection_dataset, num_train, augment, csvfile)
    train_loader = DataLoader(
        train_dataset, batch_size, shuffle, num_workers, pin_memory)

    # valid_dataset = datasets.ImageFolder(valid_dir)
    # valid_dataset = OmniglotTest(valid_dataset, trials, way, seed=0)
    # valid_loader = DataLoader(
    #     valid_dataset, batch_size=way, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    # return (train_loader, valid_loader)
    return train_loader


class OmniglotTrain(Dataset):
    def __init__(self, imageset, dataset, num_train, augment, csvfile):
        super(OmniglotTrain, self).__init__()
        self.imageset = imageset
        self.dataset = dataset
        self.num_train = num_train
        self.augment = augment
        self.csvfile = csvfile

    def __len__(self):
        return self.num_train

    def __getitem__(self, index):
        # image1 = 
        image1 = random.choice(self.dataset.imgs)

        # get image from same class
        label = None
        if index % 2 == 1:   # odd indices
            label = 1.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                # image1[0] contains the filename dir while image1[1] is an index representing
                # the image folder directory. 
                # index is the same = image from same folder (same character)
                if image1[1] == image2[1]:  
                    break
        # get image from different class
        else:
            label = 0.0
            while True:
                image2 = random.choice(self.dataset.imgs)
                if image1[1] != image2[1]:
                    break
        image1 = Image.open(image1[0])
        image2 = Image.open(image2[0])
        image1 = image1.convert('L')  # 8 bit pixels, black and white
        image2 = image2.convert('L')

        # apply affine transformation: max 10 deg rotation, 10% translation, 0.8-1.2 scale and 2 deg shear
        if self.augment:
            trans = transforms.Compose([
                transforms.RandomAffine(degrees=10, scale=(0.8, 1.2), shear=2, fillcolor=255),
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.ToTensor()

        image1 = trans(image1)
        image2 = transforms.ToTensor()(image2)
        y = torch.from_numpy(np.array([label], dtype=np.float32))
        return (image1, image2, y)
'''