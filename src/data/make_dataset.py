# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import numpy as np
import os
from torchvision import datasets, transforms
import torch


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    # transform that makes mean 0 and std 1
    data_transform = transforms.Compose([
        transforms.Normalize((0.5,), (0.5,))])

    test = np.load(os.path.join(input_filepath, 'corruptmnist/test.npz'))
    images = test['images']
    labels = test['labels']
    test_images = torch.from_numpy(images).float()
    test_labels = torch.from_numpy(labels).long()
    test_images = data_transform(test_images)

    #save to processed
    torch.save((test_images, test_labels), os.path.join(output_filepath, 'test.pt'))

    #do the same for train


    # load train_0.npz to train_4.npz
    train_0 = np.load(os.path.join(input_filepath, 'corruptmnist/train_0.npz'))
    train_1 = np.load(os.path.join(input_filepath, 'corruptmnist/train_1.npz'))
    train_2 = np.load(os.path.join(input_filepath, 'corruptmnist/train_2.npz'))
    train_3 = np.load(os.path.join(input_filepath, 'corruptmnist/train_3.npz'))
    train_4 = np.load(os.path.join(input_filepath, 'corruptmnist/train_4.npz'))
    # make a listt
    train = [train_0, train_1, train_2, train_3, train_4]

    images = [dat['images'] for dat in train]
    labels = [dat['labels'] for dat in train]
    # concatenate the images and labels
    images = np.concatenate(images)
    labels = np.concatenate(labels).reshape(-1, 1)
    images = torch.from_numpy(images).float()
    train_labels = torch.from_numpy(labels).long()
    train_images = data_transform(images)

    #save to processed
    torch.save((train_images, train_labels), os.path.join(output_filepath, 'train.pt'))




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
