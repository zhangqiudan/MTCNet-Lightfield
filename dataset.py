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

class ImageDataTest(data.Dataset):
    def __init__(self, test_mode=1, sal_mode='e'):

        if test_mode == 1:
            if sal_mode == 'p':
                self.image_root = '/data/qiudan/lightfield/MTCNet/dataset/'
                self.image_source = '/data/qiudan/lightfield/MTCNet/dataset/test_lytro.lst'
                self.test_fold = './results/'
            elif sal_mode == 'd':
                self.image_root = '/data/qiudan/lightfield/MTCNet/dataset/'
                self.image_source = '/data/qiudan/lightfield/MTCNet/dataset/test.lst'
                self.test_fold = './results/'
            elif sal_mode == 'h':
                self.image_root = '/data/qiudan/lightfield/dataset/DUTLF_V2/dutlf/'
                self.image_source = '/data/qiudan/lightfield/dataset/DUTLF_V2/dutlf/testdutlf_mtcnet.lst'
                self.test_fold = './results/'

        with open(self.image_source, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, name = load_image(os.path.join(self.image_root, self.image_list[item % self.image_num].split()[0]))
        MV_90 = load_views_90_test(os.path.join(self.image_root, self.image_list[item%self.image_num].split()[1]))
        MV_0 = load_views_0_test(os.path.join(self.image_root, self.image_list[item%self.image_num].split()[1]))
        MV_45 = load_views_45_test(os.path.join(self.image_root, self.image_list[item%self.image_num].split()[1]))
        MV_M45 = load_views_M45_test(os.path.join(self.image_root, self.image_list[item%self.image_num].split()[1]))

        image = torch.Tensor(image)
        MV_90 = torch.Tensor(MV_90)
        MV_0 = torch.Tensor(MV_0)
        MV_45 = torch.Tensor(MV_45)
        MV_M45 = torch.Tensor(MV_M45)


        return {'image': image,'MV_90': MV_90, 'MV_0': MV_0, 'MV_45': MV_45, 'MV_M45': MV_M45, 'name': name}

    def save_folder(self):
        return self.test_fold

    def __len__(self):
        return self.image_num



def get_loader(batch_size, mode='train', num_thread=1, test_mode=0, sal_mode='e'):
    shuffle = False
    if mode == 'test':
        dataset = ImageDataTest(test_mode=test_mode, sal_mode=sal_mode)

    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_thread)
    return data_loader, dataset


def load_image(pah):
    if not os.path.exists(pah):
        print('File Not Exists:',pah)
    #img_name = pah[52:-4]
    #img_name = pah[57:-4]
    img_name = pah[54:-4] #dutlf
    name = img_name + '.png'
    #print('pah', pah)
    #print('img_name', img_name)
    #print('name', name)
    #exit()
    im = cv2.imread(pah)
    im = cv2.resize(im,(540,375)) ###

    in_ = np.array(im, dtype=np.float32)
    in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2, 0, 1))
    return in_, name

def load_views_90_test(pah):  
    if not os.path.exists(pah):
        print('File Not Exists:',pah)
    #img_path = pah[:57]
    #img_name = pah[57:65]
    #img_path = pah[:55]
    #img_name = pah[46:54]  
    img_path = pah[:57] #dutlfv2
    img_name = pah[57:61]
    #print(pah)
    #print(img_path)
    #print(img_name)
    #exit()
    view_n = 7 
    slice_for_5x5 = int(0.5 * (7 - view_n))

    seq90d = list(range(14, 77, 9)[::-1][slice_for_5x5:9 - slice_for_5x5:])  # 90degree:  [76, 67, 58, 49, 40, 31, 22, 13, 4 ]
    #print(seq90d)
    #exit()
    im_s = cv2.imread(pah)
    im_s = cv2.resize(im_s,(540,375))

    if(im_s.shape[0]==540):
        image_array = np.zeros((7,540,375,3))
    elif(im_s.shape[0]==375):
        image_array = np.zeros((7,375,540,3))

    for i in range(7):
        #img_all_path = img_path + img_name+ '_'+str(seq90d[i]) + '.jpg'
        img_all_path = img_path  + img_name +'/'+ img_name+ '_'+str(seq90d[i]) + '.jpg'
        #print(img_all_path) 
        #exit()
        im = cv2.imread(img_all_path)
        im = cv2.resize(im,(540,375))
        in_ = np.array(im, dtype=np.float32)

        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i,:,:,:] = in_
    image_array = image_array.transpose((3,0,1,2))  

    return image_array

def load_views_0_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')

    img_path = pah[:57] #dutlfv2
    img_name = pah[57:61]

    view_n = 7 
    slice_for_5x5 = int(0.5 * (7 - view_n))

    seq0d = list(range(38, 45, 1)[slice_for_5x5:9 - slice_for_5x5:])  

    im_s = cv2.imread(pah)
    im_s = cv2.resize(im_s,(540,375))
    if(im_s.shape[0]==540):
        image_array = np.zeros((7,540,375,3))
    elif(im_s.shape[0]==375):
        image_array = np.zeros((7,375,540,3))

    for i in range(7):
        img_all_path = img_path  + img_name +'/'+ img_name+ '_'+str(seq0d[i]) + '.jpg'
        im = cv2.imread(img_all_path)
        im = cv2.resize(im,(540,375))
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i,:,:,:] = in_
    image_array = image_array.transpose((3,0,1,2))

    return image_array

def load_views_45_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    img_path = pah[:57] #dutlfv2
    img_name = pah[57:61] 
    view_n = 7 
    slice_for_5x5 = int(0.5 * (7 - view_n))
    seq45d = list(range(17, 73, 8)[::-1][slice_for_5x5:9 - slice_for_5x5:])  

    im_s = cv2.imread(pah)
    im_s = cv2.resize(im_s,(540,375))
    if(im_s.shape[0]==540):
        image_array = np.zeros((7,540,375,3))
    elif(im_s.shape[0]==375):
        image_array = np.zeros((7,375,540,3))

    for i in range(7):
        img_all_path = img_path  + img_name +'/'+ img_name+ '_'+str(seq45d[i]) + '.jpg'
        im = cv2.imread(img_all_path)
        im = cv2.resize(im,(540,375))
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i,:,:,:] = in_
    image_array = image_array.transpose((3,0,1,2))

    return image_array
	
def load_views_M45_test(pah):
    if not os.path.exists(pah):
        print('File Not Exists')
    img_path = pah[:57] #dutlfv2
    img_name = pah[57:61] 

    view_n = 7 
    slice_for_5x5 = int(0.5 * (7 - view_n))

    seqM45d = list(range(11, 81, 10)[slice_for_5x5:9 - slice_for_5x5:])

    im_s = cv2.imread(pah)
    im_s = cv2.resize(im_s,(540,375))
    if(im_s.shape[0]==540):
        image_array = np.zeros((7,540,375,3))
    elif(im_s.shape[0]==375):
        image_array = np.zeros((7,375,540,3))

    for i in range(7):
        img_all_path = img_path  + img_name +'/'+ img_name+ '_'+str(seqM45d[i]) + '.jpg'

        im = cv2.imread(img_all_path)
        im = cv2.resize(im,(540,375))
        in_ = np.array(im, dtype=np.float32)
        in_ -= np.array((104.00699, 116.66877, 122.67892))
        image_array[i,:,:,:] = in_
    image_array = image_array.transpose((3,0,1,2))

    return image_array




