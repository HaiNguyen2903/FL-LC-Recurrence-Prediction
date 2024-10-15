import os
import os.path as osp
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from IPython import embed
import pandas as pd
import torch
import pydicom
import pydicom_seg
import json
from utils import *
from PIL import Image

from collections import defaultdict


class Cancer_Dataset(Dataset):
    def __init__(self, data_root, tumor_info_json, max_slices=3, transform=None) -> None:
        '''
        params:
            data_root: 
            tumor_info_json: json file contains paths to extract tumor information
            max_slices: maximum slices to used per patient (ordering by tumor size descending)
        '''
        self.data_root = data_root
        self.metadata_path = osp.join(self.data_root, 'metadata.csv')
        self.clinical_path = osp.join(self.data_root, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')
        self.img_root = osp.join(self.data_root, 'NSCLC Radiogenomics')
        self.transform = transform
        self.max_slices = max_slices
        self.df_meta = pd.read_csv(self.metadata_path)
        
        # with open(tumor_info_json, 'r') as f:
        #     tumor_info = json.load(f)

        # self.tumor_info = tumor_info
        # self.labels = self._get_labels()
        # self.pids = list(self.labels.keys())

        # self.data_list = self._gather_data_list()

    def _studyid_to_pid(self, stid):
        # each patient can have multiple Study UID
        st_dict = dict(zip(self.df_meta['Study UID'], self.df_meta['Subject ID']))
        return st_dict[stid]
             

    def _get_study_info(self):
        '''
        {
            patient_1: 
                {
                    'segment dir': relative path to segment dir from image root dir
                    'CT dir': relative path to CT dir from image root dir
                },
            ...
        }
        '''
        # get studies with segmentation
        df_seg = self.df_meta[self.df_meta['Modality'] == 'SEG']

    
    def extract_dicom_uids(self, dicom_path):
        try:
            # Read the DICOM file
            dicom_data = pydicom.dcmread(dicom_path, force=True)  # force=True ensures reading even if incomplete

            # Extract UIDs
            study_uid = dicom_data.StudyInstanceUID if hasattr(dicom_data, 'StudyInstanceUID') else None
            series_uid = dicom_data.SeriesInstanceUID if hasattr(dicom_data, 'SeriesInstanceUID') else None
            sop_uid = dicom_data.SOPInstanceUID if hasattr(dicom_data, 'SOPInstanceUID') else None

            return study_uid, series_uid, sop_uid

        except Exception as e:
            # print(f"Error reading {dicom_path}: {e}")
            return None, None, None
        

    def match_segments_with_scans(self):
        # filter rows for CT scans and Segmentation
        df_ct = self.df_meta[self.df_meta['Modality'] == 'CT']
        df_seg = self.df_meta[self.df_meta['Modality'] == 'SEG']

        # get study ids
        seg_stids = list(df_seg['Study UID'].unique())

        print(len(seg_stids))

        # filter ct scans of these studies (currently missing ct scans for 3 patients in metadata file: 018, 019 and 047)

        df_ct = df_ct[df_ct['Study UID'].isin(seg_stids)]
        print(len(df_ct))

        ct_locations = defaultdict(list)
        seg_locations = defaultdict(list)

        # get segmentation locations
        for _, row in df_seg.iterrows():
            stid = row['Study UID']
            seg_dir = row['File Location']
            seg_locations[stid].append(seg_dir)

        # get ct locations
        for _, row in df_ct.iterrows():
            stid = row['Study UID']
            ct_dir = row['File Location']

            # ignore studies that don't have segmentation
            if stid not in seg_locations:
                continue

            ct_locations[stid].append(ct_dir)


        # for stid in seg_locations:
        #     path = osp.join(seg_locations[stid][0], '1-1.dcm')
        #     st_uid, sr_uid, sop_uid = self.extract_dicom_uids(path)

        #     if st_uid is not None or sr_uid is not None or sop_uid is not None:
        #         print(stid)
        





    def _get_tumor_slice_indices(self, np_arr):
        slice_indices = []
        for i in range(np_arr.shape[0]):
            slice = np_arr[i]
            if len(np.unique(slice)) > 1:
                slice_indices.append(i)    
        return slice_indices

    def _crop_tumor_bboxes(self, tumor_slice, padding=10):
        '''
        Cropping tumor bbox for 1 particular slice
        Params:
            tumor_slice: the slice to crop tumor bbox
            padding: number of pixels to crop around the tumor segmentation
        Return:
            np.array of cropped tumor bbox
        '''
        # get location of the tumor
        indices = np.where(tumor_slice == 1)

        # Get minimum and maximum indices along each axis
        min_x, max_x = np.min(indices[0]), np.max(indices[0])
        min_y, max_y = np.min(indices[1]), np.max(indices[1])

        # define bbox area in the array with padding
        min_row, max_row = min_x - padding, max_x + padding
        min_col, max_col = min_y - padding, max_y + padding

        # return cropped bbox around tumor area
        return tumor_slice[min_row:max_row, min_col: max_col]

    def _get_largest_tumor_slices(self, np_tumor, limit):
        '''
        Calculate tumor areas for all slices and retrieve slices with largest tumor areas
        Params:
            np_tumor: multiple slices of the tumor segmentation
            limit: maximum slices to retrieve based on the tumor areas
        Return:
            np.array of filtered slices with largest tumor areas
        '''
        # calculate tumor area for all slices
        tumor_areas = [np.count_nonzero(slice) for slice in np_tumor]
        # sort tumor slices in descending order based on tumor areas
        sorted_indices = sorted(range(len(tumor_areas)), key=lambda i: -tumor_areas[i])
        
        # return the slices with largest tumor area
        return np_tumor[sorted_indices[:limit], :, :]

    def _get_tumor_bboxes(self, pid, max_slices=3):
        tumor_path = osp.join(self.img_root, pid, self.tumor_info[pid]['segment_dir'], '1-1.dcm')
        # load tumor segmentation
        np_tumor = load_dcm_segment(tumor_path)
        # flip the order of np array to match with the order of original scans
        np_tumor = np.flip(np_tumor, axis=0)

        # keep only slice with tumor segmentation
        sl_with_tumor = self._get_tumor_slice_indices(np_tumor)
        np_tumor = np_tumor[sl_with_tumor, :, :]
        
        if max_slices > 0:
            # get only slices with largest tumor area
            np_tumor = self._get_largest_tumor_slices(np_tumor, limit=max_slices)

        # tumor_bboxes[pid] = [self._crop_tumor_bboxes(slice) for slice in np_tumor]
        tumor_bboxes =  [self._crop_tumor_bboxes(slice) for slice in np_tumor]
        return tumor_bboxes

    def _get_labels(self) -> dict:
        df_label = pd.read_csv(self.clinical_path)
        
        # only patient AMC-049 has Recurrence as 'Not collected'
        df_label = df_label[['Case ID', 'Recurrence']]
        
        # get only subset patients with tumor information
        df_label = df_label[df_label['Case ID'].apply(lambda id: id in self.tumor_info)]

        # return a list of dict [{ID: Recurrence}]
        labels = df_label.to_dict('records')

        # reformat to dict 
        labels = {label['Case ID']: label['Recurrence'] for label in labels}

        return labels

    def _gather_data_list(self):
        data_list = []

        for pid in self.pids:
            tumor_bboxes = self._get_tumor_bboxes(pid, max_slices=self.max_slices)
            tumor_imgs = [Image.fromarray(bbox) for bbox in tumor_bboxes]
            # convert to RGB
            tumor_imgs = [img.convert('RGB') for img in tumor_imgs]

            # transform data if needed
            if self.transform:
                # tumor_bboxes = [self.transform(cropped_box) for cropped_box in tumor_bboxes]
                tumor_imgs = [self.transform(img) for img in tumor_imgs]

            # 0 if no and 1 if yes
            label = 0 if self.labels[pid] == 'no' else 1
            label_tensor = torch.tensor(label, dtype=torch.long)
            
            # append each slice tumor to the list separately
            data_list.extend([
                {
                    'id': pid,
                    'recurrence': label_tensor,
                    'tumor_img': img
                }
                for img in tumor_imgs
            ])

        return data_list

    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # pid = list(self.labels)[idx]

        # tumor_bboxes = self._get_tumor_bboxes(pid, max_slices=self.max_slices)

        # sample = {
        #     'id': pid,
        #     'recurrence': self.labels[pid],
        #     'tumor_bboxes': tumor_bboxes
        # }

        # return sample
        return self.data_list[idx]

    def _split_by_patients(self, train_ratio=0.6, test_ratio=0.2):        
        train_size = int(self.__len__() * train_ratio)
        test_size = int(self.__len__() * test_ratio)
        val_size = self.__len__() - train_size - test_size

        train_indices = list(range(train_size))
        val_indices = list(range(train_size, train_size + val_size))
        test_indices = list(range(train_size + val_size, self.__len__()))

        train_dataset = torch.utils.data.Subset(self, train_indices)
        val_dataset = torch.utils.data.Subset(self, val_indices)
        test_dataset = torch.utils.data.Subset(self, test_indices)

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

def get_ct_scan_sop_instance_uids(dicom_file):
    """
    Extracts the SOPInstanceUIDs of CT scans referenced in a segmentation DICOM file.

    Args:
        dicom_file (str): The path to the segmentation DICOM file.

    Returns:
        list: A list of SOPInstanceUIDs in order.
    """

    ds = pydicom.dcmread(dicom_file)

    # Extract the relevant information
    sop_instance_uids = []

    for d in ds.SegmentSequence:
        print(d.ReferencedSOPInstanceUID)

    # for segment in ds.SegmentSequence:
        # referenced_sop_instance_uid = segment.SOPInstanceUID
        # sop_instance_uids.append(referenced_sop_instance_uid)

    # Sort the SOPInstanceUIDs based on their position in the segmentation file
    # sop_instance_uids.sort(key=lambda x: int(x.split('.')[-1]))

    # return sop_instance_uids


if __name__ == '__main__':
    data_root = 'datasets/NSCLC/manifest-1622561851074'

    transform = transforms.Compose([
        # transforms.Lambda(lambda x: x.repeat(3, 1, 1) if x.size(0)==1 else x),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    data = Cancer_Dataset(data_root=data_root, 
                          tumor_info_json='data_segmentation_filtered.json',
                          transform=transform)

    data.match_segments_with_scans()
    exit()

    # train_loader, val_loader, test_loader = data.get_dataloaders()

    df = pd.read_csv('datasets/NSCLC/manifest-1622561851074/metadata.csv')
    label = pd.read_csv(osp.join(data_root, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv'))
    path = osp.join(data_root, df.loc[332, 'File Location'])


    segment_path = 'datasets/NSCLC/manifest-1622561851074/NSCLC Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/1000.000000-3D Slicer segmentation result-67652/1-1.dcm'

    # ds = pydicom.dcmread(osp.join(glob.glob(osp.join(path, '*.dcm'))[0]))
    # ds = pydicom.dcmread(segment_path)

    # print(ds.SOPInstanceUID)
    # print()

    # for d in ds.PerFrameFunctionXalGroupsSequence:
    #     print(d.SOPInstanceUID)

    # ct_ids = get_ct_scan_sop_instance_uids(segment_path)
    # print(ct_ids)