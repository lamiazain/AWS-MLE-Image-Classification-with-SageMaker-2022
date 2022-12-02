#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
from torchvision import datasets, transforms
import argparse
import logging


import argparse
import json
import logging
import os
import sys


import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))
#TODO: Import dependencies for Debugging andd Profiling



def test(model, test_loader,criterion):
    
    #TODO: Complete this function that can take a model and a 
         # testing data loader and will get the test accuray/loss of the model
          #Remember to include any debugging/profiling hooks that you might need
    
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    logger.info(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )

def train(args,model, train_loader,test_loader, criterion, optimizer):
    
    #TODO: Complete this function that can take a model and
          #data loaders for training and will get train the model
          #Remember to include any debugging/profiling hooks that you might need
    
    for epoch in range(1, (args.epochs) + 1):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )
        test(model, test_loader,criterion)
    save_model(model, args.model_dir)

    
def net():
    
    #TODO: Complete this function that initializes your model
          #Remember to use a pretrained model
    
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features  #2048
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1000),nn.Linear(1000, 100),nn.Linear(100, 10))
    
    return model
    
transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    )

transform_valid = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ]
    ) 
def create_train_data_loaders(batch_size, training_dir):
    
    #This is an optional function that you may or may not need to implement
    #depending on whether you need to use data loaders or not
    
    
    logger.info("Get train data loader")
    dataset = datasets.CIFAR10(training_dir,train=True,transform = transform_train)

    return torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=True)


def create_test_data_loaders(test_batch_size, training_dir):
    
    #This is an optional function that you may or may not need to implement
    #depending on whether you need to use data loaders or not
    
    dataset=datasets.CIFAR10(training_dir, train=False, transform=transform_valid)
    return torch.utils.data.DataLoader(dataset, batch_size=test_batch_size,shuffle=True)


def save_model(model, model_dir):
    #This Function saves the model as model.pth in the Model directory of the Operating system
    #
    logger.info("Saving the model.")
    path = os.path.join(model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)
      

def main(args):
    #
    #TODO: Initialize a model by calling the net function
    #
    model=net()
    
    #
    #TODO: Create your loss and optimizer
    #
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #
    #calling train and test loaders
    #
    train_loader = create_train_data_loaders(args.BatchSize, args.data_dir)
    test_loader = create_test_data_loaders(args.test_batch_size, args.data_dir)
    #
    #TODO: Call the train function to start training your model
    #Remember that you will need to set up a way to get training data from S3
    #
    model = train(args, model, train_loader, test_loader, loss_criterion, optimizer)
    
    #
    #TODO: Test the model to see its accuracy
    #
    #test(model, test_loader, loss_criterion)
    
    #
    #TODO: Save the trained model
    #
    #save_model(model, args.model_dir)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    #
    #TODO: Specify any training args that you might need
    #
      # Data and model checkpoints directories
    parser.add_argument(
        "--BatchSize",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)"
    )
    parser.add_argument(
        "--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)"
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num- ", type=int, default=os.environ["SM_NUM_GPUS"])

    
    args=parser.parse_args()
    
    main(args)
