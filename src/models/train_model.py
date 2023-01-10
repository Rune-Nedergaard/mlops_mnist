import argparse
import sys
import torch
import click

from custom_dataset import MyAwesomeDataset
from model import MyAwesomeModel


datafolder='../../data/processed/'



@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help="learning rate to use for training")
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    train_set = MyAwesomeDataset(datafolder)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
    # set the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    # set the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # set the number of epochs
    epochs = 30
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    model.to(device)
    # set the model to training mode
    model.train()

    # loop through the epochs
    for epoch in range(epochs):
        # loop through the batches
        for data, labels in train_loader:
            # get the data and labels

            # move the data and labels to the device
            data = data.to(device)
            labels = labels.to(device)
            # zero the gradients
            optimizer.zero_grad()
            # forward pass
            output = model(data)
            # compute the loss
            loss = loss_fn(output, labels.squeeze())
            # backward pass
            loss.backward()
            # update the weights
            optimizer.step()
        # print the loss
        print(f"Epoch: {epoch+1}, Loss: {loss.item():.4f}")
    # save the model
    torch.save(model.state_dict(), "../../models/model.pth")

train()