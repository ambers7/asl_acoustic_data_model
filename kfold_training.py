import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import glob
import os
import torchvision.models as models
from math import ceil
from add_padding import collate_various_size
import random
import pickle
from acoustic_model_resnet import save_cm_figure
from acoustic_model_resnet import AcousticDataset
from sklearn.model_selection import KFold

all_data_dir = '../combined_data/'
all_files = glob.glob(os.path.join(all_data_dir, '*.npy'))

classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z')
class_to_idx = {letter: idx for idx, letter in enumerate(classes)}

train_data = []
for file_path in all_files:
    fname = os.path.basename(file_path)
    label = fname.split('_')[-1][0]
    label_idx = class_to_idx[label]
    train_data.append((file_path, label_idx))


kf = KFold(n_splits=5, shuffle=True, random_state=42)

train_data_folds = []
test_data_folds = []

for train_index, test_index in kf.split(train_data):
    train_data_folds.append([train_data[i] for i in train_index])
    test_data_folds.append([train_data[i] for i in test_index])

#train ith model: train_data_folds[i] and test_data_folds[i]
fold = 0

if __name__ == "__main__":
    trainset = AcousticDataset(train_data_folds[i], is_train=True) #create an instance of the AcousticDataset class, passing in the data directory and label dictionary
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_various_size) 
    
    testset = AcousticDataset(test_data_folds[i]) 
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_various_size)

    print("Train set size:", len(trainset))
    print("Batch size:", trainloader.batch_size)
    print("Batches per epoch:", len(trainloader))

    net = models.resnet18(num_classes=26)
    net.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False) 
    net.maxpool = nn.Identity()  # Remove maxpool for small images

   
    criterion = nn.CrossEntropyLoss() #defines the loss function- want to minimize this loss function
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) #optimizer: SGD (Stochastic gradient descent); lr: learning rate; momentum: accelerates SGD in relevant direction

    for epoch in range(2):  # loop over the dataset multiple times: 2 epochs (epoch: a pass over the entire dataset)

        running_loss = 0.0 #keeps track of loss during each mini-batch
        for i, data in enumerate(trainloader, 0): #enumeration starts from 0
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients: by default gradients accumulate so if dont zero out then gradients from current mini-batches get added to gradients from prev mini-batches -> incorrect updates
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs) #forward pass: inputs (mini-batch of images) is fed into net (the network) -> outputs (logits for each image in the patch)
            loss = criterion(outputs, labels) #calculates loss; criterion (CrossEntropyLoss) compares the networks outputs w/ the true labels to quantify how well the network is performing
            loss.backward() #backward pass: pytorch computes the gradients of loss w/ respect to all networks parameters that require gradients
            
            optimizer.step() #updates the networks parameters: uses gradients computed during the backward pass to adjust the weights and biases of the network in a direction that minimizes the loss

            # print statistics
            running_loss += loss.item() #accumulates the loss for the current mini-batch
            if i % 10 == 9:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}') #print current epoch number and current mini-batch, and average loss over last 2000 batches rounded to 3 decimal places
                running_loss = 0.0 #resets running_loss after printing statistics for last 2000 lines

    print('Finished Training')

    #save trained model
    PATH = f'./resnet_kfold{fold+1}.pth'
    torch.save(net.state_dict(), PATH)

    correct = 0
    total = 0
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    ground_truth = []
    predictions = []

    with torch.no_grad():  # since we're not training, we don't need to calculate the gradients for our outputs
        for data in testloader:
            images, labels = data
            outputs = net(images) # calculate outputs by running images through the network
            _, predicted = torch.max(outputs, 1) # the class with the highest energy is what we choose as prediction

            #for confusion matrix
            ground_truth.extend(labels.cpu().numpy())
            predictions.extend(predicted.cpu().numpy())

            #for total correct predictions
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #calculates # of correct predictions in the current batch and adds it to the correct count
            
            #for letter accuracy
            for label, prediction in zip(labels, predicted): #loop thru true labels and predicted labels for each image w/in current batch
                print(f"True: {classes[label]}, Predicted: {classes[prediction]}")
                if label == prediction:
                    correct_pred[classes[label]] += 1 #if prediction correct increment count in correct_pred
                total_pred[classes[label]] += 1 #increments total count, regardless of whether prediction was correct or not2qq

    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')

    run = fold+1  #current run of training the model, +1 since non kfold model is 0
    
    #save confusion matrix
    save_cm_figure(ground_truth, predictions, classes, f'cms/acoustic_cnn_cm_{run}.png')

    # save ground_truth and predictions so can get aggregate confusion matrix later
    with open(f'ground_truth_run{run}.pkl', 'wb') as f:
        pickle.dump(ground_truth, f)
    with open(f'predictions_run{run}.pkl', 'wb') as f:
        pickle.dump(predictions, f)