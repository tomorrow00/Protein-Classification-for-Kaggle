import os
import pandas as pd
import numpy as np
from PIL import Image
from imgaug import augmenters as iaa

import torch
from torchvision import transforms
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    """Fluorescence microscopy images of protein structures training dataset"""

    def __init__(self, image_dir, label_file, mode, augument=False):
        """
        Args:
            label_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
        """

        self.mode = mode
        self.image_ids = self.to_one_hot(pd.read_csv(label_file))
        self.image_dir = image_dir
        self.trans = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
        self.augument = augument

    def to_one_hot(self, df):
        if self.mode == 'train':
            tmp = df.Target.str.get_dummies(sep=' ')
            tmp.columns = map(int, tmp.columns)
            return df.join(tmp.sort_index(axis=1))
        else:
            return df

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids.iloc[idx, 0]

        image = np.zeros(shape=(512, 512, 3))
        # image = np.zeros(shape=(512, 512, 4))
        r = np.array(Image.open(os.path.join(self.image_dir, img_name + "_red.png")))
        g = np.array(Image.open(os.path.join(self.image_dir, img_name + "_green.png")))
        b = np.array(Image.open(os.path.join(self.image_dir, img_name + "_blue.png")))
        # y = np.array(Image.open(os.path.join(self.image_dir, img_name + "_yellow.png")))
        image[:,:,0] = r.astype(np.uint8)
        image[:,:,1] = g.astype(np.uint8)
        image[:,:,2] = b.astype(np.uint8)
        # image[:,:,3] = y.astype(np.uint8)
        image = image.astype(np.uint8)

        ############################## save original image #################################
        # image_ori = Image.fromarray(image).convert('RGB')
        # image_ori.save(os.path.join(self.image_dir, "../OriginalImage/" + self.mode + "/") + img_name + ".png")
        ###################################################################################

        if self.augument:
            image = self.augumentor(image)

        if self.trans:
            image = self.trans(image)

        if self.mode == 'train':
            label = self.image_ids.iloc[idx, 2:].values
            label = label.astype(np.float16)
            label = torch.from_numpy(label).type(torch.FloatTensor)

            return img_name, image.float(), label
        else:
            return img_name, image.float()

    def get_image_ids_from_dir_contents(self, image_dir):
        all_images = [name for name in os.listdir(image_dir) \
                      if os.path.isfile(os.path.join(image_dir, name))]
        return list(set([name.split('_')[0] for name in all_images]))

    def augumentor(self, image):
        augment_img = iaa.Sequential([
            iaa.OneOf([
                iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270),
                iaa.Affine(shear=(-16, 16)),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                iaa.GaussianBlur(0.1),
            ])], random_order=True)

        image_aug = augment_img.augment_image(image)
        return image_aug