#https://docs.pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
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


'''1. load npy files from a directory
2. create a torch.utils.data.Dataset to load my npy files
3. create a DataLoader to load the dataset in batches
4. define a simple 1D CNN model
5. train the model on the dataset
6. save the trained model
'''
class AcousticDataset(Dataset):
        def __init__(self, data_dir, label_dict, is_train=False):
            self.files = glob.glob(os.path.join(data_dir, '*.npy')) #glob: finds all files in data_dif directory that match the pattern *.npy
            self.label_dict = label_dict #label_dict: dictionary mapping file names to labels
            self.is_train = is_train

        def __len__(self):
            return len(self.files)
        
        def __getitem__(self, idx):
            arr = np.load(self.files[idx])  # shape: (4, 600, 472)

            arr = arr.astype(np.float32)
            
            if self.is_train:
                #apply masking (set random values to zero)
                if (random.random() > 0.2):
                    mask_width = random.randint(10, 20)
                    rand_start = random.randint(0, arr.shape[1] - mask_width)
                    arr[:, rand_start: rand_start + mask_width, :] = 0.0
                #print('mask')

                #add noise 
                if random.random() > 0.2:
                    noise_arr = np.random.random(arr.shape).astype(np.float32) * 0.1 + 0.95
                    arr *= noise_arr
                    #print('noise: ', noise_arr.shape, noise_imu.shape)
                
            #normalize data
            for c in range(arr.shape[0]):
                # instance-level norm
                mu, sigma = np.mean(arr[c]), np.std(arr[c])
                #print( mu, sigma)
                arr[c] = (arr[c] - mu) / sigma
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            # prepares data for 1D cnn, each row is a feature vector of length 4, and there are 600*472 such vectors
            arr = torch.tensor(arr, dtype=torch.float32)
            #convert reshaped numpy array into a pytorch tensor

            fname = os.path.basename(self.files[idx])
            label = self.label_dict[fname] #get corresponding asl letter (label) from the filename
            return arr, label    

def save_cm_figure(true_label,predict_label, best_save_path, acc, lst): 
    true_labels= [label_dic_reverse[i] for i in  true_label]
    #predicted_labels = df["Predicted Label"].tolist()
    predicted_labels= [label_dic_reverse[i] for i in predict_label]
    # Get unique class names and sort them (ensures correct label order)
    unique_classes = sorted(set(true_labels) | set(predicted_labels))
    # Compute confusion matrix with string labels
    cm = confusion_matrix(true_labels, predicted_labels, labels=unique_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    # Plot confusion matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap="Blues", linewidths=0.5)
    # Keep the label order in figure
    plt.xticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=90)
    plt.yticks(ticks=np.arange(len(lst)) + 0.5, labels=lst, rotation=0)

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - Best Accuracy : %.3f"%acc + " %")
    plt.xticks(rotation=45)  # Rotate class labels for better visibility
    plt.yticks(rotation=0)
    plt.savefig(best_save_path+"confusion_matrix.png", dpi=300, bbox_inches="tight")  # Saves as a high-quality PNG

if __name__ == "__main__":
       
    '''
    create a torch.utils.data.Dataset to load npy files
    '''
    
    # create label dictionary that maps the letter to the file name 
    train_data_dir = '../train/'
    test_data_dir = '../test/'

    classes = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z')
    class_to_idx = {letter: idx for idx, letter in enumerate(classes)}
   
    def create_label_dict(data_dir):
        npy_files = glob.glob(os.path.join(data_dir, '*.npy'))
        label_dict = {}
        for file_path in npy_files:
            fname = os.path.basename(file_path)
            label = fname.split('_')[-1][0]  # Gets the first character after the last underscore
            label_idx = class_to_idx[label]  # Convert letter to integer index
            label_dict[fname] = label_idx
        return label_dict
        
    train_label_dict = create_label_dict(train_data_dir) #create a dictionary mapping file names to labels (asl letters)
    test_label_dict = create_label_dict(test_data_dir) #create a dictionary mapping file names to labels for the test set

    trainset = AcousticDataset(train_data_dir, train_label_dict, is_train=True) #create an instance of the AcousticDataset class, passing in the data directory and label dictionary
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2, collate_fn=collate_various_size) 
    
    testset = AcousticDataset(test_data_dir, test_label_dict) 
    testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_various_size)

    print("Train set size:", len(trainset))
    print("Batch size:", trainloader.batch_size)
    print("Batches per epoch:", len(trainloader))

    net = models.resnet18(num_classes=26)
    net.conv1 = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False) 
    net.maxpool = nn.Identity()  # Remove maxpool for small images

    # net = models.vgg16(num_classes=26) 
    # net.features[0] = nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1, bias=False) #change first convolutional layer to accept 4 channels
    # net.classifier[6] = nn.Linear(4096, 26) #change the last fully connected layer to output 26 classes 

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
            #the gradients indicate hwo much each parameter needs to reduce the loss
            '''gradient (calclus): direction and rate of sttepest increase of loss function w/ respect to inputs (the model's parameters- weights and biases)
            '''
            optimizer.step() #updates the networks parameters: uses gradients computed during the backward pass to adjust the weights and biases of the network in a direction that minimizes the loss

            # print statistics
            running_loss += loss.item() #accumulates the loss for the current mini-batch
            if i % 10 == 9:    
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}') #print current epoch number and current mini-batch, and average loss over last 2000 batches rounded to 3 decimal places
                running_loss = 0.0 #resets running_loss after printing statistics for last 2000 lines

    print('Finished Training')

    #save trained model
    PATH = './acoustic_vgg16.pth'
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

    #save confusion matrix
    save_cm_figure(ground_truth, predictions, classes, 'cms/acoustic_cnn_cm.png')

    # save ground_truth and predictions so can get aggregate confusion matrix later
    run = 1  #current run of training the model
    with open(f'ground_truth_run{run}.pkl', 'wb') as f:
        pickle.dump(ground_truth, f)
    with open(f'predictions_run{run}.pkl', 'wb') as f:
        pickle.dump(predictions, f)