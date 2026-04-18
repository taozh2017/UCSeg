import os
import torch
import cv2
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

class build_Dataset(Dataset):
    def __init__(self, data_dir, split, transform=None, model="None"):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.sample_list = []
        self.model = model

        if self.split == "train":
            labeled_path = os.path.join(self.data_dir + "/labeled/image")
            sample_list_labeled = os.listdir(labeled_path)
            sample_list_labeled = [os.path.join(labeled_path, item) for item in sample_list_labeled]
            self.sample_list = sample_list_labeled
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "train_semi":
            labeled_path = os.path.join(self.data_dir + "/labeled/image")
            unlabeled_path = os.path.join(self.data_dir + "/unlabeled/image")
            sample_list_labeled = os.listdir(labeled_path)
            sample_list_unlabeled = os.listdir(unlabeled_path)
            sample_list_labeled = [os.path.join(labeled_path, item) for item in sample_list_labeled]
            sample_list_unlabeled = [os.path.join(unlabeled_path, item) for item in sample_list_unlabeled]
            self.sample_list = sample_list_labeled + sample_list_unlabeled
            print("train total {} labeled samples, {} unlabeled samples".
                  format(len(sample_list_labeled), len(sample_list_unlabeled)))
        elif self.split == "val":
            val_path = os.path.join(self.data_dir + "/val/image")
            sample_list_val = os.listdir(val_path)
            self.sample_list = [os.path.join(val_path, item) for item in sample_list_val]
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "train_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/train_slices.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/slices/" + image_name + ".h5" for image_name in self.image_list]
            print("train total {} samples".format(len(self.sample_list)))
        elif self.split == "val_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/val.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            print("val total {} samples".format(len(self.sample_list)))
        elif self.split == "test_acdc_list":
            labeled_path = os.path.join(self.data_dir + "/test.list")
            with open(labeled_path, 'r') as f:
                self.image_list = f.readlines()
            self.image_list = [item.replace('\n','') for item in self.image_list]
            self.sample_list = [self.data_dir + "/data/" + image_name + ".h5" for image_name in self.image_list]
            print("test total {} samples".format(len(self.sample_list)))
        elif "test" in self.split:
            if "CVC-300" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-300/image")
            elif "thyroid" in self.split:
                test_path = os.path.join(self.data_dir + "/test_all/image")
            elif "CVC-ClinicDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-ClinicDB/image")
            elif "CVC-ColonDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/CVC-ColonDB/image")
            elif "ETIS-LaribPolypDB" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/ETIS-LaribPolypDB/image")
            elif "Kvasir" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/Kvasir/image")
            elif "ISIC2018" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            elif "DDTI" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/DDTI/image")
            elif "tn3k" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/tn3k/image")
            elif "BrainMRI" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            elif "MRI_Hippocampus" in self.split:
                test_path = os.path.join(self.data_dir + "/TestDataset/image")
            print('test_path: ', test_path)
            sample_list_val = os.listdir(test_path)
            self.sample_list = [os.path.join(test_path, item) for item in sample_list_val]
            print("test total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if "acdc" not in self.split:
            case = self.sample_list[idx]
            image = cv2.imread(case) / 255.0
            label_path = case.replace("image", "mask")
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE) / 255

            if self.transform:
                image = image.astype(np.float32)
                data = self.transform(image=image, mask=label)
                image = data['image']
                label = data['mask']
            channel1 = np.zeros_like(label)
            channel2 = np.zeros_like(label)
            channel1[label < 0.5] = 1
            channel2[label > 0.5] = 1
            label = np.stack((channel1, channel2), axis=-1)
            image = image.transpose(2, 0, 1).astype('float32')
            label = label.transpose(2, 0, 1).astype('float32')
            image, label = torch.tensor(image), torch.tensor(label)
            sample = {"image": image, "label": label}
            return sample
        else:
            if "val" not in self.split and "test" not in self.split:
                case = self.sample_list[idx]
                h5f = h5py.File(case)
                image = h5f['image'][:]
                label = h5f['label'][:]
                if self.transform:
                    data = self.transform(image=image, mask=label)
                    image = data['image']
                    label = data['mask']
                background = label == 0
                class_1 = label == 1
                class_2 = label == 2
                class_3 = label == 3
                background = np.expand_dims(background, axis=0)
                class_1 = np.expand_dims(class_1, axis=0)
                class_2 = np.expand_dims(class_2, axis=0)
                class_3 = np.expand_dims(class_3, axis=0)
                image = np.expand_dims(image, axis=0).astype('float32')
                label = np.concatenate((background, class_1, class_2, class_3), axis=0).astype('float32')

                image, label = torch.tensor(image), torch.tensor(label)
                image = image.repeat(3, 1, 1)

                sample = {"image": image, "label": label}

                return sample
            else:
                case = self.sample_list[idx]
                h5f = h5py.File(case)
                image = h5f['image'][:].astype('float32')
                label = h5f['label'][:].astype('float32')

                image, label = torch.tensor(image), torch.tensor(label)

                sample = {"image": image, "label": label}

                return sample

