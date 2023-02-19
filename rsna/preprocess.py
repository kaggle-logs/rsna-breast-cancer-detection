import cv2
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils import resample

from albumentations import (ToFloat, Normalize, VerticalFlip, HorizontalFlip, Compose, Resize,
                            RandomBrightnessContrast, HueSaturationValue, Blur, GaussNoise, CoarseDropout,
                            Rotate, RandomResizedCrop, Cutout, ShiftScaleRotate, ToGray)
import albumentations as A
from albumentations.pytorch import ToTensorV2


def load_img(fname, color="gray"):
    img = cv2.imread(fname)
    if color == "gray" : 
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif color == "rgb": 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else :
        raise NotImplementedError(color)

    return img


def df_preprocess(data, is_train = True, sampling="up"):

    if is_train : 
        # Keep only columns in test + target variable
        data = data[["patient_id", "image_id", "laterality", "view", "age", "implant", "path", "cancer"]]
        data_cancer_0 = data[data["cancer"]==0]
        data_cancer_1 = data[data["cancer"]==1]
        # oversampling 
        #   - replace : 重複を許す (True)
        if sampling == "normal":
            # そのまま
            pass
        elif sampling == "up":
            # 正例を upsamling する
            data_cancer_1_over = resample(data_cancer_1, replace=True, n_samples=len(data_cancer_0), random_state=27)
            data = pd.concat([data_cancer_0, data_cancer_1_over])
        elif sampling == "down":
            # 負例を downsamling する
            data_cancer_0_under = resample(data_cancer_0, replace=True, n_samples=len(data_cancer_1), random_state=27)
            data = pd.concat([data_cancer_0_under, data_cancer_1])
        else :
            raise NotImplementedError
    else :
        data = data[["patient_id", "image_id", "laterality", "view", "age", "implant", "path"]]

    # Encode categorical variables
    # Avoid 'SettingWithCopyWarning'
    le_laterality = LabelEncoder()
    le_view = LabelEncoder()
    encoded_laterality = le_laterality.fit_transform(data["laterality"])
    encoded_view = le_view.fit_transform(data["view"])
    data["laterality_LE"] = encoded_laterality
    data["view_LE"] = encoded_view

    # print("Number of missing values in Age:", data["age"].isna().sum())
    data['age'] = data['age'].fillna(int(data["age"].mean()))
    
    data['patient_id'] = data['patient_id'].astype(int)

    # reset index
    data = data.reset_index(drop=True)

    return data




# -----------------------------------------------------------------
# Breast RoI
#
class BreastPreprocessor:
    def __init__(self, version):
        self.version = version

    def is_flip_side(self, img, mode="pixel"):
        """
        胸部の向きを決定する（右向きに統一する）
        """
        if mode == "pixel":
            # 画素値のsumを列方向に射影して1次元配列にする
            # 真ん中で2分割する
            col_sums_split = np.array_split(np.sum(img, axis=0), 2)
            # 左右の領域に分けたときの画素値の総和を計算する
            left_col_sum = np.sum(col_sums_split[0])
            right_col_sum = np.sum(col_sums_split[1])
            # 画素値が大きい方にオブジェクト --> 胸部がある、と判定する
            # 右側に胸部があれば flip 対象とする
            if right_col_sum > left_col_sum : 
                return True
            else:
                return False
        elif mode == "var" : 
            # 画素値のsumを列方向に射影して1次元配列にする
            # 真ん中で2分割する
            col_sums_split = np.array_split(np.sum(img, axis=0), 2)
            # 左右の領域に分けたときの画素値の総和を計算する
            left_col_sum = np.var(col_sums_split[0])
            right_col_sum = np.var(col_sums_split[1])
            # 分散が大きい = 背景と胸部が写っている、と判定する
            # 右側に胸部があれば flip 対象とする
            if right_col_sum > left_col_sum : 
                return True
            else:
                return False

    def is_background_white(self, img):
        # 列方向に画素値の総和を取る
        left, right = np.array_split(np.sum(img, axis=0), 2)
        if np.var(left) < np.var(right) : 
            low_var_region  = left
        else : 
            low_var_region = right
        
        if np.mean(low_var_region) > 100 : 
            return True
        else:
            return False

    def crop_coords(self, img):
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
            # 見つからなければそのままのサイズを返す
            return (0,0,self.img.shape[0], img.shape[1])
        # 大きな領域であるものを胸部とする
        cnt = max(cnts, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, w, h)

    def truncation_normalization(self, img):
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

    def clahe(self, clip, img):
        """
        Image enhancement.
        @img : numpy array image
        @clip : float, clip limit for CLAHE algorithm
        return: numpy array of the enhanced image
        """
        clahe = cv2.createCLAHE(clipLimit=clip)
        cl = clahe.apply(np.array(img*255, dtype=np.uint8))
        return cl

    def get_breast_region(self, img):
        if self.version == "not-used":
            return img, []
        elif self.version == "v1":
            return self._get_breast_region_1(img)
        elif self.version == "v2":
            return self._get_breast_region_2(img)
        elif self.version == "v3":
            return self._get_breast_region_3(img)
        else : 
            raise NotImplementedError()
    
    def _get_breast_region_1(self, image):
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

    def _get_breast_region_2(self, img):
        orig_shape = img.shape

        # 胸部反転処理
        if self.is_flip_side(img):
            img = np.fliplr(img)

        # クロップ
        (x, y, w, h) = self.crop_coords(img)
        img_cropped = img[y:y+h, x:x+w]

        img_normalized = self.truncation_normalization(img_cropped)

        # Enhancing the contrast of the image.
        cl1 = self.clahe(img_normalized, 1.0)
        cl2 = self.clahe(img_normalized, 2.0)
        img_final = cv2.merge((np.array(img_normalized*255, dtype=np.uint8),cl1,cl2))
        img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2GRAY)

        # Resize the image to the final shape. 
        img_final = cv2.resize(img_cropped, orig_shape)

        return img_final, [img_cropped,]


    def _get_breast_region_3(self, img):
        orig_shape = img.shape
        
        # 背景色を白色に統一 (背景が白255か黒0か判定)
        if np.mean(img.flatten()) < 100:
            img = cv2.bitwise_not(img)

        # 胸部反転処理
        if self.is_flip_side(img, mode="var"):
            img = np.fliplr(img)

        # クロップ
        (x, y, w, h) = self.crop_coords(img)
        img_cropped = img[y:y+h, x:x+w]

        # Resize the image to the final shape. 
        img_final = cv2.resize(img_cropped, orig_shape)

        return img_final, [img_cropped,]

    
    def _get_breast_region_4(self, img):
        orig_shape = img.shape

        # 胸部反転処理
        if self.is_flip_side(img, mode="var"):
            img = np.fliplr(img)

        # クロップ
        (x, y, w, h) = self.crop_coords(img)
        img_cropped = img[y:y+h, x:x+w]

        # Resize the image to the final shape. 
        img_final = cv2.resize(img_cropped, orig_shape)

        return img_final, [img_cropped,]


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
                    ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]), # ランダムにアフィン変換を適用
                    HorizontalFlip(p = cfg.aug.horizontal_flip), # 水平方向にフリップ
                    VerticalFlip(p = cfg.aug.vertical_flip), # 垂直方向にフリップ
                    Normalize(mean=0, std=1),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.1" :
                self.transform_train = Compose([
                    Normalize(mean=0, std=1),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.2" :
                self.transform_train = Compose([
                    ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]), # ランダムにアフィン変換を適用
                    HorizontalFlip(p = cfg.aug.horizontal_flip), # 水平方向にフリップ
                    VerticalFlip(p = cfg.aug.vertical_flip), # 垂直方向にフリップ
                    RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
                    CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.3" :
                self.transform_train = Compose([
                    ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]), # ランダムにアフィン変換を適用
                    HorizontalFlip(p = cfg.aug.horizontal_flip), # 水平方向にフリップ
                    VerticalFlip(p = cfg.aug.vertical_flip), # 垂直方向にフリップ
                    RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
                    CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
                    Normalize(mean=(0,), std=(1,)),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.4" :
                self.transform_train = Compose([
                    ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]), # ランダムにアフィン変換を適用
                    HorizontalFlip(p = cfg.aug.horizontal_flip), # 水平方向にフリップ
                    VerticalFlip(p = cfg.aug.vertical_flip), # 垂直方向にフリップ
                    RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=1.0),
                    CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=1.0),
                    Resize(1024,512),
                    Normalize(mean=(0.2179,), std=(0.0529,)),
                    ToTensorV2(),
                ])
            elif cfg.aug.version == "v0.0.5" :
                self.transform_train = A.Compose([
                    A.OneOf([
                        A.HorizontalFlip(p = cfg.aug.horizontal_flip), # 水平方向にフリップ
                        A.VerticalFlip(p = cfg.aug.vertical_flip), # 垂直方向にフリップ
                        A.ShiftScaleRotate(rotate_limit=90, scale_limit = [0.8, 1.2]), # ランダムにアフィン変換を適用
                    ], p=0.7),
                    A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=True, p=0.25),
                    A.CoarseDropout(max_holes=4, max_height=100, max_width=100, min_holes=1, min_height=50, min_width=50, fill_value=0, p=0.25),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
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
