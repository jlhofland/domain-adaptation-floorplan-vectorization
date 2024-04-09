import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from datasets.cubicasa5k.loaders.house import House
from tqdm import tqdm


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/'):
        
        # Parameters
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size

        # Folder and file names
        self.data_folder = data_folder
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'
        self.folders = genfromtxt(data_folder + data_file, dtype='str')

        # Set data loader function
        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(data_folder+lmdb_folder, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False
        
        # Preload all images into memory
        self.samples = []
        for i, f in tqdm(enumerate(self.folders), total=len(self.folders)):
            self.samples.append(self.get_data(i))

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index):
        sample = self.samples[index]

        if self.augmentations is not None:
            sample = self.augmentations(sample)
            
        if self.is_transform:
            sample = self.transform(sample)

        return sample

    def get_txt(self, index):
        # Load image
        image = cv2.imread(self.data_folder + self.folders[index] + self.image_file_name)

        # Correct color channels
        fplan = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get dimensions
        height, width, nchannel = fplan.shape

        # Move color channels to first dimension
        fplan = np.moveaxis(fplan, -1, 0)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, height, width)
        
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1

        # If original size is needed, resize the labels
        if self.original_size:
            # Load original image
            fplan_org = cv2.imread(self.data_folder + self.folders[index] + self.org_image_file_name)

            # Correct color channels
            fplan_org = cv2.cvtColor(fplan_org, cv2.COLOR_BGR2RGB)

            # Get dimensions
            height_org, width_org, nchannel = fplan_org.shape

            # Move color channels to first dimension
            fplan_org = np.moveaxis(fplan_org, -1, 0)

            # Load labels
            label = label.unsqueeze(0)

            # Resize labels
            label = torch.nn.functional.interpolate(label, size=(height_org, width_org), mode='nearest')
            
            # Remove batch dimension
            label = label.squeeze(0)

            # Calculate the scaling factor
            coef_height = float(height_org) / float(height)
            coef_width = float(width_org) / float(width)

            # Resize heatmaps
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        # Convert to tensor
        img = torch.tensor(fplan.astype(np.float32))

        # Return dictionary
        return {
            'image': img,
            'label': label,
            'folder': self.folders[index],
            'heatmaps': heatmaps,
            'scale': coef_width
        }

    def get_lmdb(self, index):
        key = self.folders[index].encode()
        with self.lmdb.begin(write=False) as f:
            data = f.get(key)

        sample = pickle.loads(data)
        return sample

    def transform(self, sample):
        fplan = sample['image']
        # Normalization values to range -1 and 1
        fplan = 2 * (fplan / 255.0) - 1

        sample['image'] = fplan

        return sample
