import cv2
import numpy as np

from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray)
from albumentations.pytorch import ToTensorV2


def load_img(fname, color="gray"):
    img = cv2.imread(fname)
    if color == "gray" : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color == "rgb": 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2RGB)
    else :
        raise NotImplementedError(color)

    return img


def get_breast_region(image):
    orig_shape = image.shape
    # 背景が白か黒か判定
    if np.mean(image.flatten()) < 100:
        image = cv2.bitwise_not(image)

    # 2値化する
    # 精度に直結する
    ret, bin_img = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    # 輪郭を抽出する。
    contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 最も大きい領域を胸部とする
    max_size = -999
    max_contour = None
    for idx, c in enumerate(contours):
        if len(c) > max_size:
            max_size = len(c)
            max_contour = c
    
    xmin, xmax = np.min(max_contour[:,:,0]), np.max(max_contour[:,:,0])
    ymin, ymax = np.min(max_contour[:,:,1]), np.max(max_contour[:,:,1])
    if len(image.shape) == 2 : # gray image
        crop_img = image[ymin:ymax, xmin:xmax]
    elif len(image.shape) == 3 : # color image
        crop_img = image[ymin:ymax, xmin:xmax,:]
    else:
        ValueError(f"the input image shape is not one as expected {image.shape}")

    # resize to original shape
    crop_img = cv2.resize(crop_img, orig_shape)

    return crop_img, [bin_img, contours, max_contour]




class Transform():
    def __init__(self, cfg):
        # Data Augmentation (custom for each dataset type)
        if cfg.aug.version == "v0.0.0" :
            self.transform_train = Compose([
                ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]),
                HorizontalFlip(p = cfg.aug.horizontal_flip),
                VerticalFlip(p = cfg.aug.vertical_flip),
                Normalize(mean=0, std=1),
                ToTensorV2(),
            ])
        else:
            raise NotImplementedError(f"The {cfg.aug_version} is not implemented yet.")
        
        self.transform_test = Compose([
            Normalize(mean=0, std=1),
            ToTensorV2(),
        ])
    
    def get(self, is_train):
        if is_train : 
            return self.transform_train
        else : 
            return self.transform_test