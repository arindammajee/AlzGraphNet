import os
import pandas as pd
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


DATA_PATH = os.path.join('/home/arindam/Alzheimer/Data/ADNI 300 Collection', 'ADNI')
config = {
    'img_size': 256,
    'depth' : 128,
    'batch_size' : 4
}
labels_path = '/home/arindam/Alzheimer/Data/ADNI_300_Collection_9_27_2023.csv'
random.seed(21)

class DataPaths():
    def __init__(self, data_path=None, csv_path=None):
        if data_path==None:
            self.data_path = DATA_PATH
        else:
            self.data_path = data_path

        if csv_path==None:
            self.csv_path = labels_path
        else:
            self.csv_path = csv_path

    def patient_id_loading(self):
        df = pd.read_csv(self.csv_path)
        mri_scan_list = []
        id = 0

        for patient_dir in os.listdir(self.data_path):
            patient_dir_path = os.path.join(self.data_path, patient_dir)
            if os.path.isdir(patient_dir_path):
                for des_dir in os.listdir(patient_dir_path):
                    des_dir_path = os.path.join(patient_dir_path, des_dir)
                    if os.path.isdir(des_dir_path):
                        for visit in os.listdir(des_dir_path):
                            visit_path = os.path.join(des_dir_path, visit)
                            if os.path.isdir(visit_path):
                                for image_data_dir in os.listdir(visit_path):
                                    image_data_dir_path = os.path.join(visit_path, image_data_dir)
                                    if os.path.isdir(image_data_dir_path):
                                        for image in os.listdir(image_data_dir_path):
                                            image_dict = {}
                                            image_path = os.path.join(image_data_dir_path, image)
                                            if image.endswith('.nii'):
                                                image_dict['image_path'] = image_path
                                                image_dict['patient_id'] = patient_dir
                                                image_dict['image_id'] = image_data_dir
                                                image_dict['label'] = df[df['Image Data ID']==image_data_dir]['Group'].values[0]

                                                mri_scan_list.append(image_dict)
                                                if id > 0:
                                                    id -= 1
                                                    print(image_dict)
            
        
        random.shuffle(mri_scan_list)
        len_train = int(len(mri_scan_list)*0.7)
        len_val = int(len(mri_scan_list)*0.15)
        len_test = len(mri_scan_list) - (len_train + len_val)
        print(len_train, len_val, len_test)
        
        trin_img_df = pd.DataFrame(mri_scan_list[:len_train])
        trin_img_df_path = os.path.join(os.getcwd(), 'train_mri_scan_list.csv')
        trin_img_df.to_csv(trin_img_df_path, index=False)

        val_img_df = pd.DataFrame(mri_scan_list[len_train:len_train+len_val])
        val_img_df_path = os.path.join(os.getcwd(), 'val_mri_scan_list.csv')
        val_img_df.to_csv(val_img_df_path, index=False)

        test_img_df = pd.DataFrame(mri_scan_list[len_train+len_val:])
        test_img_df_path = os.path.join(os.getcwd(), 'test_mri_scan_list.csv')
        test_img_df.to_csv(test_img_df_path, index=False)

        return trin_img_df_path, val_img_df_path, test_img_df_path
    
    


class ADNIAlzheimerDataset(Dataset):
    def __init__(self, image_df_paths, transform=None):
        self.image_df_paths = image_df_paths
        self.transform = transform
        self.df = pd.read_csv(self.image_df_paths)
        self.desired_width = config['img_size']
        self.desired_height = config['img_size']
        self.desired_depth = config['depth']
        labels = self.df['label'].values

    def __label_extract(self, group):
        if group=='CN':
            return 0
        elif group=='MCI':
            return 1
        elif group=='AD':
            return 2
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        image_filepath = self.df['image_path'][idx]
        image = nib.as_closest_canonical(nib.load(image_filepath))
        image = image.get_fdata()
        image = image.reshape(image.shape[2], image.shape[1], image.shape[0])

        width_factor = self.desired_width / image.shape[0]
        height_factor = self.desired_height / image.shape[1]
        depth_factor = self.desired_depth / image.shape[-1]

        image = zoom(image, (width_factor, height_factor, depth_factor), order=1)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = image.astype('float32')
        #image = torch.from_numpy(image)

        label = self.df['label'][idx]
        label = self.__label_extract(label)
        #data['mri'] = image

        return image.reshape(1, image.shape[0], image.shape[1], image.shape[2]), label
    


def LoadDatasets(return_type='loader'):
    data_path = DataPaths()
    train_img_paths, val_img_paths, test_img_paths = data_path.patient_id_loading()
    #print("Total no of train, validation and test images are {}, {} and {} respectively.".format(len(train_img_paths), len(val_img_paths), len(test_img_paths)))

    train_dataset = ADNIAlzheimerDataset(train_img_paths)
    valid_dataset = ADNIAlzheimerDataset(val_img_paths)
    test_dataset = ADNIAlzheimerDataset(test_img_paths)

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



if __name__ == "__main__":
    dataPath = DataPaths()