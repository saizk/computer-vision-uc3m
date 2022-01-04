import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

from transforms import CropByMask


class IsicDataset(Dataset):

    def __init__(self, csv_file, img_dir, mask_dir, transform=None, max_size=2000):

        self.dataset = pd.read_csv(csv_file, header=0, dtype={'id': str, 'label': int})

        self.img_dir = img_dir
        self.mask_dir = mask_dir

        self.img_names = self.dataset.id
        self.labels = self.dataset.label

        if max_size > 0:
            idx = np.random.RandomState(seed=42).permutation(range(len(self.dataset)))
            reduced_dataset = self.dataset.iloc[idx[0: max_size]]
            self.dataset = reduced_dataset.reset_index(drop=True)

        self.transform = transform
        self.classes = ['nevus', 'melanoma', 'keratosis']

        self.img_files = [f'{img}.jpg' for img in self.img_names]
        self.masks_files = [f'{mask}.png' for mask in self.img_names]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.masks_files[idx])

        image = Image.open(img_path)

        if self.transform:
            if isinstance(self.transform.transforms[0], CropByMask):
                tr_func = self.transform.transforms[0]
                image = tr_func(image, mask=Image.open(mask_path))
                self.transform.transforms.pop(0)

            image = self.transform(image)

        label = self.labels[idx].astype("int64")

        return image, label
