import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random

class ImageDataTrain(data.Dataset):
    def __init__(self):
        self.sal_root = './data/'
        self.sal_source = './train_datalist.lst'

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)

    def __getitem__(self, item):
        sal_image = load_image(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[0]))
        sal_image90 = load_views_90(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))
        sal_image0 = load_views_0(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))
        sal_image45 = load_views_45(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))
        sal_imageM45 = load_views_M45(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[1]))

        sal_depth = load_depth(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[2]))
        sal_label = load_sal_label(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[3]))
        sal_edge = load_edge_label(os.path.join(self.sal_root, self.sal_list[item % self.sal_num].split()[4]))

        sal_image = torch.Tensor(sal_image)
        sal_image90 = torch.Tensor(sal_image90)
        sal_image0 = torch.Tensor(sal_image0)
        sal_image45 = torch.Tensor(sal_image45)
        sal_imageM45 = torch.Tensor(sal_imageM45)
        sal_depth = torch.Tensor(sal_depth)
        sal_label = torch.Tensor(sal_label)
        sal_edge = torch.Tensor(sal_edge)

        sample = {'sal_image': sal_image, 'sal_image90': sal_image90, 'sal_image0': sal_image0,
                  'sal_image45': sal_image45, 'sal_imageM45': sal_imageM45, 'sal_depth': sal_depth,
                  'sal_label': sal_label, 'sal_edge': sal_edge}
        return sample

    def __len__(self):
        return self.sal_num

def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain()
    else:
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset


def load_image(pah):
    if not os.path.exists(pah):
        print('File Not Exists:',pah)
    img_name = pah[67:-4]
    name = img_name + '.png'

    im = cv2.imread(pah)
    im = cv2.resize(im,(540,375)) ###
    print('im:',im.shape)
    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_, name


def load_views_90(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)
    #print('pah[53:58]:',pah[53:58])
 
    img_path = pah[:59]
    img_name = pah[50:58]

    print('pah:',pah)
    print('img_path:',img_path)
    print('img_name:',img_name)
    #print('img_index:',img_index)
    #print('img_index2:', pah[80:82])
    #exit()
    view_n = 7  ### 9x9 views
    slice_for_5x5 = int(0.5 * (7 - view_n))
    
    seq90d = list(
        range(14, 77, 9)[::-1][slice_for_5x5:9 - slice_for_5x5:])  # 90degree:  [76, 67, 58, 49, 40, 31, 22, 13, 4 ]
    image_array = np.zeros((7, 375, 540, 3))
    
    for i in range(7):
        img_all_path = img_path  + img_name + '_' + str(seq90d[i]) + '.png'
        print('img_all_path:', img_all_path)
        #exit()
        im = cv2.imread(img_all_path)
        in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
        in_ = np.array(in_, dtype=np.float32)
    
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i, :, :, :] = in_
    #exit()
    image_array = image_array.transpose((3, 0, 1, 2))
    return image_array


def load_views_0(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)
    
    img_path = pah[:59]
    img_name = pah[50:58]
    
    view_n = 7  ### 9x9 views
    slice_for_5x5 = int(0.5 * (7 - view_n))
    
    seq0d = list(
        range(38, 45, 1)[slice_for_5x5:9 - slice_for_5x5:])  # 0degree:  [36, 37, 38, 39, 40, 41, 42, 43, 44]
    image_array = np.zeros((7, 375, 540, 3))
    
    for i in range(7):
        img_all_path = img_path + img_name + '_' + str(seq0d[i]) + '.png'
        #print('img_all_path:', img_all_path)
        # exit()
        im = cv2.imread(img_all_path)
        in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
        in_ = np.array(in_, dtype=np.float32)
    
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i, :, :, :] = in_
    image_array = image_array.transpose((3, 0, 1, 2))
    return image_array


def load_views_45(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)

    img_path = pah[:59]
    img_name = pah[50:58]
    
    view_n = 7  ### 9x9 views
    slice_for_5x5 = int(0.5 * (7 - view_n))
    seq45d = list(
        range(17, 73, 8)[::-1][slice_for_5x5:9 - slice_for_5x5:])  # 45degree:  [72, 64, 56, 48, 40, 32, 24, 16, 8 ]
    
    image_array = np.zeros((7, 375, 540, 3))
    
    for i in range(7):
        img_all_path = img_path + img_name + '_' + str(seq45d[i]) + '.png'
        #print('img_all_path:', img_all_path)
        #exit()
        im = cv2.imread(img_all_path)
        in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
        in_ = np.array(in_, dtype=np.float32)
    
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i, :, :, :] = in_
    image_array = image_array.transpose((3, 0, 1, 2))
    return image_array


def load_views_M45(pah):
    if not os.path.exists(pah):
        print('File Not Exists:', pah)

    img_path = pah[:59]
    img_name = pah[50:58]
    
    
    view_n = 7  ### 9x9 views
    slice_for_5x5 = int(0.5 * (7 - view_n))
    
    seqM45d = list(range(11, 81, 10)[slice_for_5x5:9 - slice_for_5x5:])
    
    image_array = np.zeros((7, 375, 540, 3))
    
    for i in range(7):
        img_all_path = img_path + img_name + '_' + str(seqM45d[i]) + '.png'
        #print('img_all_path:', img_all_path)
        #exit()
        im = cv2.imread(img_all_path)
        in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
        in_ = np.array(in_, dtype=np.float32)
    
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i, :, :, :] = in_
    image_array = image_array.transpose((3, 0, 1, 2))

    return image_array

def load_depth(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    # im = Image.open(pah)
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
    label = np.array(in_, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    # im_sz = [375,540]
    # if(label.shape[0]>label.shape[1]):
    #    label = label.reshape(im_sz)
    # print('the shape of after depth reshape  *****:',label.shape)
    label = label / 255.
    label = label[np.newaxis, ...]
    return label


def load_edge_label(pah):

    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
    label = np.array(in_, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def load_sal_label(pah):
    """
    Load label image as 1 x height x width integer array of label indices.
    The leading singleton dimension is required by the loss.
    """
    if not os.path.exists(pah):
        print('File Not Exists')
    im = cv2.imread(pah)
    in_ = cv2.resize(im, (540, 375), interpolation=cv2.INTER_LINEAR)
    label = np.array(in_, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    # label = cv2.resize(label, im_sz, interpolation=cv2.INTER_NEAREST)
    label = label / 255.
    label = label[np.newaxis, ...]
    return label
