import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import numpy as np
# from convert_to_num2 import ConvertToNum
import config
import pandas as pd



class NumDataset(Dataset):
    def __init__(self, train=True):
        all_inputs = config.records[['bd_level', 'star_level']]
        all_classes = config.records[['target']]
        # print(all_inputs)
        (training_inputs,
         testing_inputs,
         training_classes,
         testing_classes) = train_test_split(all_inputs, all_classes, train_size=0.8, random_state=1)
        if train:
            self.data = training_inputs
            self.label = training_classes
        else:
            self.data = testing_inputs
            self.label = testing_classes
        self.size = len(self.data)
    def __getitem__(self, index):
        feature = self.data.iloc[index]
        label = self.label.iloc[index]
        feature = torch.FloatTensor([int(i) for i in list(feature)])
        label = torch.FloatTensor([int(i) for i in label])
        # print(feature, label)
        return feature, label

    def __len__(self):
        # print(self.size)
        return self.size

def get_data_loader(train=True):
    data_loader = DataLoader(NumDataset(train), batch_size=config.batch_size, shuffle=True)
    return data_loader

if __name__ == '__main__':
    loader = get_data_loader()
    for idx,(input,target) in enumerate(loader):
        print(idx)
        print(input)
        print(input.size())
        print(target)
        print(target.size())
        print(len(loader.dataset))
        break



