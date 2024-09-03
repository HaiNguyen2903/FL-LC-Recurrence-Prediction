import os
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
from IPython import embed
import pandas as pd
import torch
import numpy as np
import pydicom
from skimage.filters import threshold_yen
from skimage.measure import label, regionprops
from medpy.filter.smoothing import anisotropic_diffusion
from PIL import Image
from typing import List, Tuple


random.seed(42)

class DICOMImageProcessor:
    def __init__(self, filename: str):
        self.filename = filename
        self.pixel_array = self.load_dicom_file()

    def load_dicom_file(self) -> np.ndarray:
        """Load and normalize the DICOM file."""
        try:
            ds = pydicom.dcmread(self.filename)
            pixel_array = ds.pixel_array.astype(np.float32)
            if np.max(pixel_array) != 0:
                pixel_array /= np.max(pixel_array)  # Normalize to [0, 1]
            return pixel_array
        except pydicom.errors.InvalidDicomError:
            raise RuntimeError(f"Invalid DICOM file: {self.filename}")
        except Exception as e:
            raise RuntimeError(f"Failed to load DICOM file: {e}")

    def apply_anisotropic_diffusion_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply anisotropic diffusion filter to the image."""
        return anisotropic_diffusion(image, niter=5, kappa=50, gamma=0.1)

    def apply_yen_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply Yen's thresholding to binarize the image."""
        threshold_value = threshold_yen(image)
        return image > threshold_value

    def apply_labeling(self, binary_image: np.ndarray) -> np.ndarray:
        """Label the binary image to identify regions."""
        return label(binary_image)

    def get_combined_region_properties(self, labeled_image: np.ndarray) -> Tuple[int, int, int, int]:
        """Get the combined bounding box of all labeled regions with an expansion margin."""
        regions = regionprops(labeled_image)
        if regions:
            min_row = min(r.bbox[0] for r in regions)
            min_col = min(r.bbox[1] for r in regions)
            max_row = max(r.bbox[2] for r in regions)
            max_col = max(r.bbox[3] for r in regions)
            
            # Expand bounding box by a margin
            margin = 10  # Adjust this value as needed
            min_row = max(min_row - margin, 0)
            min_col = max(min_col - margin, 0)
            max_row = min(max_row + margin, labeled_image.shape[0])
            max_col = min(max_col + margin, labeled_image.shape[1])
            
            return (min_row, min_col, max_row, max_col)
        return (0, 0, 0, 0)

    def process_image(self) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Process the image and return filtered image and bounding box."""
        try:
            filtered_image = self.apply_anisotropic_diffusion_filter(self.pixel_array)
            binary_image = self.apply_yen_threshold(filtered_image)
            labeled_image = self.apply_labeling(binary_image)
            bbox = self.get_combined_region_properties(labeled_image)
            return filtered_image, bbox
        except Exception as e:
            raise RuntimeError(f"Error during image processing: {e}")

    def extract_region(self, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract the region of interest based on the bounding box."""
        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
            return self.pixel_array[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        return np.zeros((1, 1), dtype=np.float32)

    def resize_image(self, image: np.ndarray, size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Resize the image to the specified size."""
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize(size, Image.LANCZOS)
        return np.array(pil_image)

    def to_pil_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to a PIL image."""
        pil_image = Image.fromarray((image * 255).astype(np.uint8))
        pil_image = pil_image.resize((224, 224), Image.LANCZOS)
        return pil_image


class Cancer_Dataset(Dataset):
    def __init__(self, data_root, transform=None) -> None:
        self.data_root = data_root
        self.metadata_path = osp.join(self.data_root, 'metadata.csv')
        self.clinical_path = osp.join(self.data_root, 'NSCLCR01Radiogenomic_DATA_LABELS_2018-05-22_1500-shifted.csv')
        self.img_root = osp.join(self.data_root, 'NSCLC Radiogenomics')
        self.transform = transform

        # get label dict
        self.labels = self._get_labels()

        # get img paths dict 
        self.img_paths = self._get_img_paths()

        # gather label and img paths
        self.data_list = self._gather_data()

        return
    
    def _get_labels(self) -> dict:
        df_label = pd.read_csv(self.clinical_path)
        
        # only patient AMC-049 has Recurrence as 'Not collected'
        df_label = df_label[['Case ID', 'Recurrence']]
        
        # get only subset patients with prefix R01
        df_label = df_label[df_label['Case ID'].apply(lambda id: id[:3]) == 'R01']

        # return a list of dict [{ID: Recurrence}]
        labels = df_label.to_dict('records')

        # reformat to dict 
        labels = {label['Case ID']: label['Recurrence'] for label in labels}

        return labels

    def _get_img_paths(self) -> dict:
        # each dir contains images of 1 patient
        img_dirs = os.listdir(self.img_root)

        # get only images for patients with labels
        img_dirs = [id for id in img_dirs if id in self.labels]

        # save data to dict with format {'ID': [img_paths]}
        img_paths = {}
        
        for patient_id in img_dirs:
            # init dict
            img_paths[patient_id] = []

            # recursively walk through all directories
            for root, _ , files in os.walk(osp.join(self.img_root, patient_id)):
                for file in files:
                    if file.endswith(".dcm"):
                        # img_paths[patient_id].append(os.path.join(root, file))
                        img_paths[patient_id].append(osp.join(root, file))
            
        return img_paths

    def _gather_data(self) -> list:
        # gather a list of ids, labels and image paths
        data = []
        for id in self.img_paths:
            data.extend([{'id': id, 'label': self.labels[id], 'path': path} for path in self.img_paths[id]])

        return data

    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        # get sample in the data list
        sample = self.data_list[idx]
        
        # Process the DICOM image
        processor = DICOMImageProcessor(sample['path'])
        filtered_image, bbox = processor.process_image()
        region_image = processor.extract_region(bbox)
        pil_image = processor.to_pil_image(region_image)

        # transform image if needed
        if self.transform:
            pil_image = self.transform(pil_image)

        # 0 if no and 1 if yes
        label = 0 if sample['label'] == 'no' else 1
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return pil_image, label_tensor

    def split_by_patients(self, train_ratio = 0.6, test_ratio = 0.5):
        # get patient ids 
        pids = list(self.labels.keys())

        # random sample data
        ids_train = random.sample(pids, int(len(pids) * train_ratio))
        ids_left = [id for id in pids if id not in ids_train]
        ids_test = random.sample(ids_left, int(len(ids_left) * test_ratio))

        print(len(ids_train), len(ids_test))

        # split data list
        train_ls = []
        test_ls = []
        val_ls = []

        for sample in self.data_list:
            id = sample['id']
            
            if id in ids_train:
                train_ls.append(sample)
            elif id in ids_test:
                test_ls.append(sample)
            else:
                val_ls.append(sample)

        return train_ls, test_ls, val_ls
    
if __name__ == '__main__':
    data_root = 'datasets/NSCLC/manifest-1622561851074'
    dataset = Cancer_Dataset(data_root)
    # train, test, val = dataset.split_by_patients()

    # train_ls, test_ls, val_ls = dataset.split_by_patients()

    # print(len(train_ls), len(test_ls), len(val_ls))
    # print(len(dataset.data_list))

    data_len = len(dataset.data_list)

    
    