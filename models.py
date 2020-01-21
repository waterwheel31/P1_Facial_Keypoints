## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        # input channel 1, output channle 32, conv kernel = 5 
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv_bn = nn.BatchNorm2d(32)
        self.conv_drop = nn.Dropout2d(p=0.2)
        
        self.conv2 = nn.Conv2d(32, 64, 3) 
        
        self.pool = nn.MaxPool2d(2, 2) 
        
        self.linear1 = nn.Linear(64*40*40, 272) 
        self.linear2 = nn.Linear(272, 68 * 2)
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # size changes from (1, 168, 168) to  (32, 164, 164)  
        #print("x.size():", x.size())
        x = self.conv1(x)
        x = self.conv_bn(x)
        x = F.relu(x)
        
        # size changes from ( 32, 164, 164 )  to (32, 82, 82) 
        # print("x.size():", x.size())
        x = self.conv_drop(x)
        x = self.pool(x) 
        
        # size changes from ( 32, 82, 82 )  to (64, 40, 40) 
        #print("x.size():", x.size())
        x = self.conv2(x) 
        x = F.relu(x)
        x = self.pool(x) 
        
        
        # size changes from (32, 40, 40) to (1, 32*40*40) 
        #print("x.size():", x.size())
        x = x.view(x.size()[0],-1) 
        
        #print("x.size():", x.size())
        x = self.linear1(x)
        x = F.relu(x)
        
        #print("x.size():", x.size())
        x = self.linear2(x)
        
        #print("x.size():", x.size())
        x = x.view(x.size()[0],-1) 
        #print("x.size():", x.size())
        
        return x
