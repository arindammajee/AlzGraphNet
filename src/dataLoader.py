import os
import nibabel as nib
import numpy as np
import random
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from torch.nn.functional import one_hot
import torch


DATA_PATH = os.path.join('/home/arindam/Alzheimer', 'Data/MIRIAD/miriad')
config = {
    'img_size': 256,
    'depth' : 64,
    'batch_size' : 8
}
random.seed(21)

class DataPaths():
    def __init__(self):
        DATA_PATH = os.path.join('/home/arindam/Alzheimer', 'Data/MIRIAD/miriad')

    def image_paths_loading(self):
        AD_path, HC_path = [], []
        for scan_dir in os.listdir(DATA_PATH):
            scan_dir_path = os.path.join(DATA_PATH, scan_dir)
            if os.path.isdir(scan_dir_path):
                for visit in os.listdir(scan_dir_path):
                    scan_visit = os.path.join(scan_dir_path, visit)
                    for f in os.listdir(scan_visit):
                        f_path = os.path.join(scan_visit, f)
                        if f.endswith(".nii"):
                            if 'AD' in f:
                                AD_path.append(f_path)
                            else:
                                HC_path.append(f_path)
                                
                
        print("Total number of images are: {}\n".format(len(AD_path)+len(HC_path)))     
        incons = []
        for path in AD_path+HC_path:
            scan = nib.load(path)
            data = scan.get_fdata()
            if data.shape != (256, 256, 124):
                print('Shape inconsistancy found! {} for {}'.format(data.shape, path))
                incons.append(path)
                
        print("\n\nRemove shape insconsitent images.")
        for path in incons:
            if 'AD' in path:
                AD_path.remove(path)
            else:
                HC_path.remove(path)
                
        print("After Removing shape inconsistent images total number of images {}.".format(len(AD_path)+len(HC_path)))
        print("Number of Alzheimer infected MRI scans: {}".format(len(AD_path)))
        print("Number of Healthy MRI scans: {}".format(len(HC_path)))

        return AD_path, HC_path


    def train_test_split(self, path_list, train_split_ratio=0.8, val_split_ratio=0.1):
        test_split_ratio = 1 - (train_split_ratio + val_split_ratio)
        random.shuffle(path_list)
        
        train_image_paths = path_list[:int(train_split_ratio*len(path_list))]
        valid_image_paths = path_list[int(train_split_ratio*len(path_list)):int((train_split_ratio+val_split_ratio)*len(path_list))]
        test_image_paths = path_list[int((train_split_ratio+val_split_ratio)*len(path_list)):]
        

        if len(train_image_paths)+len(valid_image_paths)+len(test_image_paths)==len(path_list):
            print("Everything is fine. No of images in train, val and test set is {}, {} and {} respectively.".format(len(train_image_paths), len(valid_image_paths), len(test_image_paths)))
        else:
            print("Something wrong. Go and start debug!")
        
        return train_image_paths, valid_image_paths, test_image_paths

    def data_path_loading(self):
        AD_path, HC_path = self.image_paths_loading()
        train_ad_image_paths, val_ad_image_paths, test_ad_image_paths = self.train_test_split(AD_path)
        train_hc_image_paths, val_hc_image_paths, test_hc_image_paths = self.train_test_split(HC_path)

        train_img_paths = train_ad_image_paths + train_hc_image_paths
        val_img_paths = val_ad_image_paths + val_hc_image_paths
        test_img_paths = test_ad_image_paths + test_hc_image_paths
        
        random.shuffle(train_img_paths)
        random.shuffle(val_img_paths)
        random.shuffle(test_img_paths)

        return train_img_paths, val_img_paths, test_img_paths


class ProcessScan:
    def __init__(self):
        pass
    
    def read_nifti_file(self, filepath):
        """Read and load volume"""
        scan = nib.load(filepath)
        scan = scan.get_fdata()
        return scan


    def normalize(self, volume):
        """Normalize the volume"""
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
        return volume.astype('float32')


    def resize_volume(self, img, desired_width=256, desired_height=256, desired_depth=64):
        """Resize the volume"""
        width_factor = desired_width / img.shape[0]
        height_factor = desired_height / img.shape[1]
        depth_factor = desired_depth / img.shape[-1]

        #img = ndimage.rotate(img, 90, reshape=False)
        img = zoom(img, (width_factor, height_factor, depth_factor), order=1)
        return img


    def process_scan(self, path):
        """Read and resize volume"""
        volume = self.read_nifti_file(path)
        volume = self.normalize(volume)
        volume = self.resize_volume(volume, config['img_size'], config['img_size'], config['depth'])

        return volume
    
    def label_extract(self, path):
        """Label Extraction"""
        path = path.split('/')[-1]
        if 'AD' in path:
            return 1
        elif 'HC' in path:
            return 0
    


class MIRIADAlzheimerDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
        self.scan_process = ProcessScan()
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = self.scan_process.process_scan(image_filepath)
        image = image.reshape(1, image.shape[0], image.shape[1], image.shape[-1])
        
        label = self.scan_process.label_extract(image_filepath)
        label = one_hot(torch.tensor([label]), num_classes=2)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        
        return image, label
    


def LoadDatasets(return_type='loader'):
    data_path = DataPaths()
    train_img_paths, val_img_paths, test_img_paths = data_path.data_path_loading()
    print("Total no of train, validation and test images are {}, {} and {} respectively.".format(len(train_img_paths), len(val_img_paths), len(test_img_paths)))

    train_dataset = MIRIADAlzheimerDataset(train_img_paths)
    valid_dataset = MIRIADAlzheimerDataset(val_img_paths)
    test_dataset = MIRIADAlzheimerDataset(test_img_paths)

    if return_type=='loader':
        train_loader = DataLoader(
            train_dataset, batch_size=config['batch_size'], shuffle=True
        )

        valid_loader = DataLoader(
            valid_dataset, batch_size=config['batch_size'], shuffle=True
        )


        test_loader = DataLoader(
            test_dataset, batch_size=config['batch_size'], shuffle=False
        )

        return train_loader, valid_loader, test_loader
    
    else:
        return train_dataset, valid_dataset, test_dataset
