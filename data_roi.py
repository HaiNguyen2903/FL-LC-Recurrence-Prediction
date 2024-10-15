import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from IPython import embed
import pandas as pd
import torch
import json
from utils import *
from PIL import Image
import random
from collections import Counter
from sklearn.model_selection import train_test_split
import albumentations as A

class Cancer_Dataset(Dataset):
    def __init__(self, data_root, transform=None) -> None:
        '''
        params:
            data_root: 
            tumor_info_json: json file contains paths to extract tumor information
            max_slices: maximum slices to used per patient (ordering by tumor size descending)
        '''
        self.data_root = data_root
        self.roi_dir = osp.join(data_root, 'ROI')
        
        self.meta_path = osp.join(data_root, 'recurrence.csv')

        # get sop uids and relevant metadata where images are available
        self.sop_uids, self.df_meta = self._get_sop_uids_with_imgs()

        # map study ids with patient ids
        self.sopuids_map = self._sopuid_to_pid()

        self.roi_list = os.listdir(self.roi_dir)

        self.transform = transform

        label_int = {'no': 0, 'yes': 1}

        self.labels_dict = dict(zip(self.df_meta['SOPInstanceUID'], self.df_meta['Recurrence'].map(lambda label: label_int[label])))
        return

    def _extract_sopuid_from_name(self, name):
        return name.split('_')[0]
    
    def _get_sop_uids_with_imgs(self):
        # get all sop uids with image available
        file_names = os.listdir(self.roi_dir)

        # extract sop uids from file names with format {SOPInstanceUID}_{slice_idx}. Using set function to return unique values
        sop_uids = list(set([self._extract_sopuid_from_name(name) for name in file_names]))

        # extract metadata of relevant sop uids
        df_meta = pd.read_csv(self.meta_path, low_memory=False)
        df_meta = df_meta[df_meta['SOPInstanceUID'].isin(sop_uids)]
        
        return sop_uids, df_meta


    def _sopuid_to_pid(self):
        # each patient can have multiple Study UID
        st_dict = dict(zip(self.df_meta['SOPInstanceUID'], self.df_meta['Subject ID']))
        return st_dict

    def _load_tiff_img(self, path):
        pil_image = Image.open(path)
        return pil_image

    def __len__(self):
        return len(self.roi_list)
    
    def __getitem__(self, idx):
        file_name = self.roi_list[idx]
        sop_uid = self._extract_sopuid_from_name(file_name)
        pid = self.sopuids_map[sop_uid]
        img = self._load_tiff_img(osp.join(self.roi_dir, file_name))
        # get recurrence label
        recurrence = self.df_meta['Recurrence'][self.df_meta['SOPInstanceUID'] == sop_uid].item()
        # 0 if no and 1 if yes
        label = 0 if recurrence == 'no' else 1
        label_tensor = torch.tensor(label, dtype=torch.long)

        # convert img to RGB and transform if needed
        img = img.convert('RGB')
        if self.transform:
            img = self.transform(img)

        sample = {
            'pid': pid,
            'sop_uid': sop_uid,
            'tumor_img': img,
            'recurrence': label_tensor
        }

        return sample

    def print_class_ratio(self, sop_uids):
        recurrences = [self.df_meta['Recurrence'][self.df_meta['SOPInstanceUID'] == sop_uid].item()
                       for sop_uid in sop_uids]

        print(Counter(recurrences))
        return

    def _split_by_patients(self, train_ratio=0.6, test_ratio=0.5):
        # random.seed(40)

        # sop_uids_train = random.sample(self.sop_uids, int(len(self.sop_uids) * train_ratio))
        # temp_uids = [id for id in self.sop_uids if id not in sop_uids_train]
        # sop_uids_test = random.sample(temp_uids, int(len(temp_uids) * test_ratio))
        # sop_uids_val = [id for id in self.sop_uids if id not in sop_uids_train + sop_uids_test]

        # split train data
        label_list = [self.labels_dict[sop_uid] for sop_uid in self.sop_uids]
        sop_uids_train, sop_uids_temp = train_test_split(self.sop_uids, test_size=1-train_ratio, 
                                                    stratify=label_list, random_state=42)
        
        # split val and test data
        label_list = [self.labels_dict[sop_uid] for sop_uid in sop_uids_temp]
        sop_uids_val, sop_uids_test = train_test_split(sop_uids_temp, test_size=test_ratio,
                                                    stratify=label_list, random_state=42)
        
        train_indices = []
        test_indices = []
        val_indices = []

        for i, file_name in enumerate(self.roi_list):
            uid = self._extract_sopuid_from_name(file_name)
            if uid in sop_uids_train:
                train_indices.append(i)
            elif uid in sop_uids_test:
                test_indices.append(i)
            else:
                val_indices.append(i)

        train_dataset = torch.utils.data.Subset(self, train_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)

        self.print_class_ratio(sop_uids_train)
        self.print_class_ratio(sop_uids_test)
        self.print_class_ratio(sop_uids_val)

        return train_dataset, val_dataset, test_dataset

        
    def get_dataloaders(self, batch_size=32, shuffle=True, num_workers=0):
        train_dataset, val_dataset, test_dataset = self._split_by_patients()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                                   shuffle=shuffle, num_workers=num_workers)
        
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                                                num_workers=num_workers)
        
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                                shuffle=False, num_workers=num_workers)

        return train_loader, val_loader, test_loader 

if __name__ == '__main__':
    data_root = 'datasets/NSCLC'

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    augmentations = A.Compose([
        A.RandomRotate(limit=15, p=0.5),  # Random rotation up to 15 degrees
        A.RandomCrop(width=512, height=512, p=0.5),  # Random cropping to 512x512
        A.HorizontalFlip(p=0.5),  # Horizontal flipping
        A.VerticalFlip(p=0.5),  # Vertical flipping
        A.RandomBrightnessContrast(p=0.5),  # Random brightness and contrast adjustment
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalization
    ])

    data = Cancer_Dataset(data_root=data_root, 
                          transform=transform)

    
    train_loader, val_loader, test_loader = data.get_dataloaders()

    #     # Sample binary list
    # binary_list = [0, 1, 0, 1, 0, 0, 1, 1, 0, 1]
    # t = range(10)

    # # Split the list into training and testing sets, maintaining the proportion of 0s and 1s
    # train_list, test_list = train_test_split(t, test_size=0.4, stratify=binary_list)

    # print("Training set:", train_list)
    # print("Testing set:", test_list)