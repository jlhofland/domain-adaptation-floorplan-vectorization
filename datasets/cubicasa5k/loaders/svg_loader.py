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


class FloorplanSVG(Dataset):
    def __init__(self, data_folder, data_file, is_transform=True,
                 augmentations=None, img_norm=True, format='txt',
                 original_size=False, lmdb_folder='cubi_lmdb/', 
                 grayscale=False, load_ram=False, num_workers=0,
                 save_samples=False, load_samples=False):
        # Parameters
        self.img_norm = img_norm
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.get_data = None
        self.original_size = original_size
        self.grayscale = grayscale
        self.load_ram = load_ram
        self.save_samples = save_samples
        self.load_samples = load_samples
        self.num_workers = num_workers

        # Folder and file names
        self.data_folder = data_folder
        self.image_file_name = '/F1_scaled.png'
        self.org_image_file_name = '/F1_original.png'
        self.svg_file_name = '/model.svg'
        self.folders = genfromtxt(data_folder + data_file, dtype='str')

        if grayscale:
            self.data_file = '/data_g.pkl'
        else:
            self.data_file = '/data.pkl'

        # Set data loader function
        if format == 'txt':
            self.get_data = self.get_txt
        if format == 'lmdb':
            self.lmdb = lmdb.open(data_folder+lmdb_folder, readonly=True,
                                  max_readers=8, lock=False,
                                  readahead=True, meminit=False)
            self.get_data = self.get_lmdb
            self.is_transform = False
            self.load_ram = False

        # Save label and heatmaps in folder of the image
        if save_samples:
            self.samples_to_pickle_dist() 
        
        # Preload all images into memory
        if load_ram:
            self.images_to_ram()

    def __len__(self):
        """__len__"""
        return len(self.folders)

    def __getitem__(self, index):
        # Load image with {image, label, folder, heatmaps, scale} dictionary
        if self.load_samples:
            sample = self.load_pkl(self.data_folder + self.folders[index] + self.data_file)
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
    
    def load_image(self, index, file_name):
        # Load image
        image = cv2.imread(self.data_folder + self.folders[index] + file_name)

        if (self.grayscale):
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
        return fplan, height, width, nchannel

    def get_txt(self, index):
        # Load image in format (C, H, W), height, width, number of channels
        if self.load_ram:
            fp, h, w, c = self.images[index]
        else:
            fp, h, w, c = self.load_image(index, self.image_file_name)

        # Getting labels for segmentation and heatmaps
        house = House(self.data_folder + self.folders[index] + self.svg_file_name, h, w)
        
        # Combining them to one numpy tensor
        label = torch.tensor(house.get_segmentation_tensor().astype(np.float32))
        heatmaps = house.get_heatmap_dict()
        coef_width = 1

        # If original size is set, resize the label to the original size
        if self.original_size:
            fp, h_org, w_org, c_org = self.load_image(index, self.org_image_file_name)

            # Resize the label to the original size
            label = label.unsqueeze(0)
            label = torch.nn.functional.interpolate(label,size=(h_org, w_org),mode='nearest')
            label = label.squeeze(0)

            # Calculate the scaling factor
            coef_height = float(h_org) / float(h)
            coef_width = float(w_org) / float(w)

            # Resize the heatmaps
            for key, value in heatmaps.items():
                heatmaps[key] = [(int(round(x*coef_width)), int(round(y*coef_height))) for x, y in value]

        # Convert to tensor
        img = torch.tensor(fp.astype(np.float32))

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
    
    def save_pkl(self, data_dict, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data_dict, f)

    def load_pkl(self, filename):
        with open(filename, 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict

    def samples_to_pickle(self):
        print('Saving samples to pickle files...')
        for i, f in tqdm(enumerate(self.folders), total=len(self.folders)):
            # If exists, skip
            if os.path.exists(self.data_folder + f + self.data_file):
                continue

            # Get sample and save to pickle file
            sample = self.get_txt(i)
            self.save_pkl(sample, self.data_folder + f + self.data_file)
        print('Samples saved to pickle files.')

    def save_sample_to_pkl(self, index, get_txt_func, save_pkl_func, data_folder, folders):
        sample = get_txt_func(index)
        save_pkl_func(sample, data_folder + folders[index] + self.data_file)

    def samples_to_pickle_dist(self):
        print('Saving samples to pickle files...')
        
        # Create a multiprocessing pool
        with multiprocessing.Pool(processes=self.num_workers) as pool:
            # Use partial to pass additional arguments to the save_sample_to_pkl function
            save_partial = partial(self.save_sample_to_pkl, get_txt_func=self.get_txt, save_pkl_func=self.save_pkl,
                                data_folder=self.data_folder, folders=self.folders)
            
            # Use tqdm to track progress
            with tqdm(total=len(self.folders)) as pbar:
                for _ in pool.imap_unordered(save_partial, range(len(self.folders))):
                    pbar.update(1)
        
        print('Samples saved to pickle files.')
    
    def images_to_ram(self):
        print('Loading images to RAM...')
        self.images = []
        for i, f in tqdm(enumerate(self.folders), total=len(self.folders)):
            fp, h, w, c = self.load_image(i, self.image_file_name)
            self.images.append((fp, h, w, c))
        print('Images loaded to RAM.')