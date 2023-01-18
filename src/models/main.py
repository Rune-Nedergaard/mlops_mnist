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
    epochs = 10
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    model.to(device)
    # set the model to training mode
    model.train()
    # loop through the epochs
    for epoch in range(epochs):
        # loop through the batches
        for batch in train_loader:
            # get the data and labels
            data, labels = batch
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
    torch.save(model.state_dict(), "model.pth")


# loading model checkpoint from saved model
model_checkpoint = torch.load("models/model.pth")


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    # load the model from last saved checkpoint
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint))
    # model = torch.load(model_checkpoint)
    test_set = MyAwesomeDataset(datafolder, test=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    # do inference on the test set
    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # move the model to the device
    model.to(device)
    # set the model to evaluation mode
    model.eval()
    # set the loss function
    loss_fn = torch.nn.CrossEntropyLoss()
    # set the number of correct predictions
    correct = 0
    # set the total loss
    total_loss = 0
    # loop through the batches
    for batch in test_loader:
        # get the data and labels
        data, labels = batch
        # move the data and labels to the device
        data = data.to(device)
        labels = labels.to(device)
        # forward pass
        output = model(data)
        # compute the loss
        loss = loss_fn(output, labels.squeeze())
        # update the total loss
        total_loss += loss.item()
        # get the predictions
        _, preds = torch.max(output, 1)
        # update the number of correct predictions
        correct += torch.sum(preds == labels.squeeze()).item()
    # compute the accuracy
    accuracy = correct / len(test_set)
    # compute the average loss
    avg_loss = total_loss / len(test_loader)
    # print the accuracy and loss
    print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")


cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()
