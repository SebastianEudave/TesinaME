import glob
import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class MEDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None, csv_file=""):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.microExpresions = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.microExpresions)

    def __getitem__(self, idx):
        #if torch.is_tensor(idx):
            #idx = idx.tolist()

        images_path = os.path.join(self.root_dir, 'train', str(self.microExpresions.iloc[idx, 0]),
                                   str(self.microExpresions.iloc[idx, 1]), str(self.microExpresions.iloc[idx, 2]))
        print(images_path)
        image_list = []
        for filename in glob.glob(images_path+'/*.jpg'):
            im = Image.open(filename)
            image_list.append(im)
        emotion = self.microExpresions.iloc[idx, 0]
        sample = {'images': image_list, 'emotion': emotion}

        if self.transform:
            sample = self.transform(sample)

        return sample


dataset = MEDataset(root_dir='G:\\tesina\\Licencias\\MicroExpressions_Data2', csv_file='train_data.csv')
print(len(dataset[0]['images']))


for im in dataset[0]['images']:
    im.show()
    break


