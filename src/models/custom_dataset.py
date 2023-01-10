import torch
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

from torchvision import datasets, transforms


datafolder='../../data/processed/'

class MyAwesomeDataset(Dataset):
    def __init__(self, test=False):
        # load data from processed
        if test:
            data = torch.load(os.path.join(datafolder, 'test.pt'))
        else:
            data = torch.load(os.path.join(datafolder, 'train.pt'))
        self.images, self.labels = data

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = np.expand_dims(self.images[idx], axis=0), self.labels[idx]
        return sample



# We plot the first 10 images in the dataset
# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(2, 5, figsize=(10, 4))
# for i in range(10):
#    ax[i//5, i%5].imshow(train[i][0].numpy().squeeze(), cmap='gray')
#    ax[i//5, i%5].set_title(train[i][1].item())
#    ax[i//5, i%5].axis('off')
#    #set the figure title to train
#    fig.suptitle('Train')
# plt.show()

# plot the next 10 images
# fig, ax = plt.subplots(2, 5, figsize=(10, 4))
# for i in range(10):
#    ax[i//5, i%5].imshow(test[i+10][0].numpy().squeeze(), cmap='gray')
#    ax[i//5, i%5].set_title(test[i+10][1].item())
#    ax[i//5, i%5].axis('off')
#    #set the figure title to test
#    fig.suptitle('Test')
# plt.show()

# dataset = torch.load('../../../data/corruptmnist/train_0.npz')
# 'C:/Users/rune7/Documents/GitHub/dtu_mlops/data/corruptmnist/train_0.npz'
