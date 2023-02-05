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


# -----------------------------------------------------------------
# Breast RoI
#
def crop_coords(img):
    """
    Crop ROI from image.
    """
    # Otsu's thresholding after Gaussian filtering
    #   - 画素値分布が双峰性である場合に有効
    #   - 今回は背景が黒と白の場合があるので自動的に二値化のしきい値を決めてもらうのが良い
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, breast_mask = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # 輪郭を決める
    cnts, _ = cv2.findContours(breast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cnts) == 0:
        return (0,0,img.shape[0], img.shape[1])
    # 大きな領域であるものを胸部とする
    cnt = max(cnts, key = cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def truncation_normalization(img):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    return: numpy array of the normalized image
    """
    # img[img!=0] で0ではない画素値を取ってこれる
    # np.percentile() で 5%, 99% percentile の点を取ってくる
    pmin = np.percentile(img[img!=0], 5)
    pmax = np.percentile(img[img!=0], 99)
    if pmax-pmin < 5:
        return img
    
    # percentil の点を最小・最大とするように切り出す
    truncated = np.clip(img, pmin, pmax)  
    normalized = (truncated - pmin)/(pmax - pmin)
    normalized[img==0]=0

    return normalized


def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img*255, dtype=np.uint8))
    return cl

def get_breast_region_2(image):
    try:
        orig_shape = image.shape

        (x, y, w, h) = crop_coords(image)
        img_cropped = image[y:y+h, x:x+w]
    
        img_normalized = truncation_normalization(img_cropped)
    
        # Enhancing the contrast of the image.
        cl1 = clahe(img_normalized, 1.0)
        cl2 = clahe(img_normalized, 2.0)
        img_final = cv2.merge((np.array(img_normalized*255, dtype=np.uint8),cl1,cl2))
        img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)
    
        # Resize the image to the final shape. 
        img_final = cv2.resize(img_final, orig_shape)

        return img_final, None
    except:
        return image, None

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
    def __init__(self, cfg, only_test=False):

        if only_test : 
            self.transform_test = Compose([
                Normalize(mean=0, std=1),
                ToTensorV2(),
            ])

        else :
            # Data Augmentation (custom for each dataset type)
            if cfg.aug.version == "v0.0.0" :
                self.transform_train = Compose([
                    ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]),
                    HorizontalFlip(p = cfg.aug.horizontal_flip),
                    VerticalFlip(p = cfg.aug.vertical_flip),
                    Normalize(mean=0, std=1),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.1" :
                self.transform_train = Compose([
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