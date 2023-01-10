import torch
import click

from custom_dataset import MyAwesomeDataset
from model import MyAwesomeModel

datafolder='../../data/processed/'

model_checkpoint = "../../models/model.pth"


def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here

     # set the device
    device = torch.device("cpu")

    # load the model from last saved checkpoint
    model = MyAwesomeModel()
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    # model = torch.load(model_checkpoint)
    test_set = MyAwesomeDataset(test=True)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
    # do inference on the test set
   
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
        correct += torch.sum(preds == labels)
    # compute the accuracy
    accuracy = correct / len(test_set)
    # compute the average loss
    avg_loss = total_loss / len(test_loader)
    # print the accuracy and loss
    print(f"Accuracy: {accuracy:.4f}, Average Loss: {avg_loss:.4f}")

evaluate(model_checkpoint)