import lmdb
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from numpy import genfromtxt
from datasets.cubicasa5k.loaders.house import House
from tqdm import tqdm
import multiprocessing
from functools import partial
import os

############################
## ADJUSTED FROM CUBICASA ##
############################

class FloorplanSVGMMD(Dataset):
    def __init__(self, source_list, target_list, cfg, is_transform=True, augmentations=None, img_norm=True, pre_load=False):
        # Parameters
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = self.get_txt
        self.cfg = cfg
        self.pre_load = pre_load

        # Folder and file names
        self.data_folder = cfg.dataset.files.root
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'

        # Load source and target lists
        self.source_folders = genfromtxt(self.data_folder + source_list, dtype='str')
        self.target_folders = genfromtxt(self.data_folder + target_list, dtype='str')

        # Set data file name (grayscale or RGB)
        if cfg.dataset.grayscale:
            self.data_file = '/data_g.pkl'
        else:
            self.data_file = '/data.pkl'

        # Save label and heatmaps in folder of the image
        if cfg.dataset.save_samples and pre_load:
            self.samples_to_pickle_dist() 

    def __len__(self):
        """__len__"""
        return len(self.source_folders)

    def __getitem__(self, index):
        # Load image with {image, label, folder, heatmaps, scale} dictionary
        if self.cfg.dataset.load_samples and self.pre_load:
            sample = self.load_pkl(self.data_folder + self.source_folders[index] + self.data_file)
        else:
            sample = self.get_data(index)

        # Apply augmentations
        if self.augmentations is not None:
            sample = self.augmentations(sample)
        
        # Apply transformations
        if self.is_transform:
            sample = self.transform(sample)

        # Return the sample
        return sample
    
    def load_image(self, index, folder, file_name):
        # Load image
        image = cv2.imread(self.data_folder + folder[index] + file_name)

        if (self.cfg.dataset.grayscale):
            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Expand dimensions to make it a single-channel tensor
            image = np.expand_dims(image, axis=-1)
        else:
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get image dimensions
        height, width, nchannel = image.shape

        # Move channels to first dimension (H, W, C) -> (C, H, W)
        fplan = np.moveaxis(image, -1, 0)

        # Return the image with (C, H, W) format
        return fplan, image, height, width, nchannel

    def get_txt(self, index):
        # Load image: (C, H, W), height, width, nchannel
        fps, ims, hs, ws, cs = self.load_image(index, self.source_folders, self.image_file_name)

        # Shuffle target list (to get random first target image)
        np.random.shuffle(self.target_folders)

        # I want to loop over the target images and check if the aspect ratio is approximately the same
        # If it is, I will use that image as the target image
        for i in range(len(self.target_folders)):
            fpt, imt, ht, wt, ct = self.load_image(i, self.target_folders, self.image_file_name)

            # Calculate aspect ratio
            aspect_ratio = float(hs) / float(ws)
            target_aspect_ratio = float(ht) / float(wt)

            # If aspect ratio is approximately the same, break the loop
            if abs(aspect_ratio - target_aspect_ratio) < 0.1:
                break

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.source_folders[index] + self.svg_file_name, hs, ws)
        
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1

        # If original size is set, resize the label to the original size
        if self.cfg.dataset.original_size:
            fps, ims_org, hs_org, ws_org, cs_org = self.load_image(index, self.source_folders, self.org_image_file_name)
            fpt, imt, ht, wt, ct = self.load_image(i, self.target_folders, self.org_image_file_name)

            # Resize the label to the original size
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,size=(hs_org, ws_org),mode='nearest')
            label = label.squeeze(0)

            # Calculate the scaling factor
            coef_height = float(hs_org) / float(hs)
            coef_width = float(ws_org) / float(ws)

            # Resize the heatmaps
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

            # I want to resize fpt to the size of fps
            fpt = cv2.resize(imt, (ws_org, hs_org), interpolation=cv2.INTER_CUBIC)
        else: 
            # I want to resize fpt to the size of fps
            fpt = cv2.resize(imt, (ws, hs), interpolation=cv2.INTER_CUBIC)

        # expand dimensions if grayscale
        if self.cfg.dataset.grayscale:
            fpt = np.expand_dims(fpt, axis=-1)

        # Move channels to first dimension (H, W, C) -> (C, H, W)
        fpt = np.moveaxis(fpt, -1, 0)

        # Convert to tensor
        source_image = torch.tensor(fps.astype(np.float32))
        target_image = torch.tensor(fpt.astype(np.float32))

        # Return dictionary
        return {
            'image': source_image,
            'label': label, 
            'folder': self.source_folders[index],
            'heatmaps': heatmaps, 
            'scale': coef_width,
            'target': target_image
        }

    def transform(self, sample):
        # Normalization values to range -1 and 1
        sample['image'] = 2 * (sample['image'] / 255.0) - 1
        sample['target'] = 2 * (sample['target'] / 255.0) - 1

        # Return the sample
        return sample
    
    def save_pkl(self, data_dict, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)

    def load_pkl(self, filename):
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def samples_to_pickle(self):
        print('Saving samples to pickle files...')
        for i, f in tqdm(enumerate(self.source_folders), total=len(self.source_folders)):
            # If exists, skip
            if os.path.exists(self.data_folder + f + self.data_file) and not self.cfg.dataset.overwrite:
                continue

            # Get sample and save to pickle file
            sample = self.get_txt(i)
            self.save_pkl(sample, self.data_folder + f + self.data_file)
        print('Samples saved to pickle files.')

    def save_sample_to_pkl(self, index, get_txt_func, save_pkl_func, data_folder, folders):
        # If exists, skip
        if os.path.exists(data_folder + folders[index] + self.data_file) and not self.cfg.dataset.overwrite:
            return
        sample = get_txt_func(index)
        save_pkl_func(sample, data_folder + folders[index] + self.data_file)

    def samples_to_pickle_dist(self):
        print('Saving samples to pickle files...')
        
        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=self.cfg.dataset.num_workers) as pool:
            # Use partial to pass additional arguments to the save_sample_to_pkl function
            save_partial = partial(self.save_sample_to_pkl, get_txt_func=self.get_txt, save_pkl_func=self.save_pkl,
                                data_folder=self.data_folder, folders=self.source_folders)
            
            # Use tqdm to track progress
            with tqdm(total=len(self.source_folders)) as pbar:
                for _ in pool.imap_unordered(save_partial, range(len(self.source_folders))):
                    pbar.update(1)
        
        print('Samples saved to pickle files.')