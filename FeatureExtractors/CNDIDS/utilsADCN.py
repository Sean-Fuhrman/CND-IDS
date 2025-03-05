import numpy as np
import pandas as pd
import time 
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import linalg as LA
import scipy
from scipy import io
import sklearn
from sklearn import preprocessing
import pdb
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class meanStdCalculator(object):
	# developed and modified from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    # license BSD 3-Clause "New" or "Revised" License
    def __init__(self):
        self.mean     = 0.0
        self.mean_old = 0.0
        self.std      = 0.001
        self.count    = 0.0
        self.minMean  = 100.0
        self.minStd   = 100.0
        self.M_old    = 0.0
        self.M        = 0.0
        self.S        = 0.0
        self.S_old    = 0.0
        
    def updateMeanStd(self, data, cnt = 1):
        self.data     = data
        self.mean_old = self.mean  # copy.deepcopy(self.mean)
        self.M_old    = self.count*self.mean_old
        self.M        = self.M_old + data
        self.S_old    = self.S     # copy.deepcopy(self.S)
        if self.count > 0:
            self.S    = self.S_old + ((self.count*data - self.M_old)**2)/(self.count*(self.count + cnt) + 0.0001)
        
        self.count   += cnt
        self.mean     = self.mean_old + (data-self.mean_old)/((self.count + 0.0001))  # np.divide((data-self.mean_old),self.count + 0.0001)
        self.std      = np.sqrt(self.S/(self.count + 0.0001))
        
        # if (self.std != self.std).any():
        #     print('There is NaN in meanStd')
        #     pdb.set_trace()
    
    def resetMinMeanStd(self):
        self.minMean = self.mean  # copy.deepcopy(self.mean)
        self.minStd  = self.std   # copy.deepcopy(self.std)
        
    def updateMeanStdMin(self):
        if self.mean < self.minMean:
            self.minMean = self.mean  # copy.deepcopy(self.mean)
        if self.std < self.minStd:
            self.minStd  = self.std   # copy.deepcopy(self.std)

    def reset(self):
        self.__init__()

def probitFunc(meanIn,stdIn):
    stdIn += 0.0001  # for safety
    out    = meanIn/(torch.ones(1) + (np.pi/8)*stdIn**2)**0.5
    
    return out

def reduceLabeledData(dataTrain, labelTrain, nLabeled):
    labeledData   = torch.Tensor().float()
    labeledLabel  = torch.Tensor().long()

    nData    = dataTrain [labelTrain==torch.unique(labelTrain)[0].item()].shape[0]
    nLabeled = int(nLabeled*nData)

    min_i = torch.unique(labelTrain)[0].item()
    max_i = torch.unique(labelTrain)[-1].item()

    for i in range(min_i, max_i + 1):
        dataClass  = dataTrain [labelTrain==i]
        labelClass = labelTrain[labelTrain==i]

        labeledData  = torch.cat((labeledData,dataClass[0:nLabeled]),0)
        labeledLabel = torch.cat((labeledLabel,labelClass[0:nLabeled]),0)

    # shuffle
    try:
        row_idxs = list(range(labeledData.shape[0]))
        random.shuffle(row_idxs)
        labeledData  = labeledData[torch.tensor(row_idxs), :]
        labeledLabel = labeledLabel[torch.tensor(row_idxs)]
    except:
        pdb.set_trace()

    return labeledData, labeledLabel

def stableSoftmax(data):
    # data is in the form of numpy array n x m, where n is the number of data point and m is the number of classes
    # output = exp(output - max(output,[],2));
    # output = output./sum(output, 2);
    data = data/np.max(data,1)[:,None]
    data = np.exp(data)
    data = data/np.sum(data,1)[:,None]

    return data

def deleteRowTensor(x, index, mode = 1):
    if mode == 1:
        x = x[torch.arange(x.size(0))!=index] 
    elif mode == 2:
        # delete more than 1 row
        # index is a list of deleted row
        allRow = torch.arange(x.size(0)).tolist()
        
        for ele in sorted(index, reverse = True):  
            del allRow[ele]

        remainderRow = torch.tensor(allRow).long()

        x = x[remainderRow]
    
    return x

def deleteColTensor(x,index):
    x = x.transpose(1,0)
    x = x[torch.arange(x.size(0))!=index]
    x = x.transpose(1,0)
    
    return x

def clusteringLoss(latentFeatures, oneHotClust, centroids):
    # criterion = nn.MSELoss()
    # lossClust = criterion(latentFeatures, torch.matmul(oneHotClust,centroids))      # ((latentFeatures-torch.matmul(oneHotClust,centroids))**2).mean()
    # pdb.set_trace()
    # torch.dist(y,x,2)
    lossClust = torch.mean(torch.norm(latentFeatures - torch.matmul(oneHotClust,centroids),dim=1))
    
    return lossClust

def maskingNoise(x, noiseIntensity = 0.1):
    # noiseStr: the ammount of masking noise 0~1*100%
    
    nData, nInput = x.shape
    nMask         = np.max([int(noiseIntensity*nInput),1])
    for i,_ in enumerate(x):
        maskIdx = np.random.randint(nInput,size = nMask)
        x[i][maskIdx] = 0
    
    return x

def show_image(x):
    plt.imshow(x.numpy())

def imageNoise(x, noiseIntensity = 0.3, device = torch.device('cpu')):
    noiseIntensity = 0.3

    noise         = torch.from_numpy(noiseIntensity * np.random.normal(loc= 0.5, scale= 0.5, size= x.shape)).float().to(device)
    X_train_noisy = x + noise
    X_train_noisy = torch.clamp(X_train_noisy, 0., 1.)

    return x

def reinitNet(cnn, netlist):
    for netLen in range(len(netlist)):
        netlist[netLen].network.apply(weight_reset)

    cnn.apply(weight_reset)
        
    return netlist, cnn

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

def showImage(img):
    plt.imshow(img[0].numpy())
    plt.show()

def plotPerformance(Iter,accuracy,hiddenNode,hiddenLayer,nCluster):
    SMALL_SIZE = 10
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=8)                   # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig, axes = plt.subplots(4,1,figsize=(8, 12))
#     fig.tight_layout()

    ax1 = axes[0]
    ax2 = axes[1]
    ax3 = axes[2]
    ax4 = axes[3]
    
    ax1.plot(Iter,accuracy,'k-')
#     ax1.set_title('Testing accuracy')
    ax1.set_ylabel('Áccuracy (%)')
#     ax1.set_xlabel('Number of bathces')
    ax1.yaxis.tick_right()
    ax1.autoscale_view('tight')
    ax1.set_ylim(ymin=0,ymax=100)
    ax1.set_xlim(xmin=0,xmax=len(Iter))

    ax2.plot(Iter,hiddenNode,'k-')
#     ax2.set_title('Testing loss')
    ax2.set_ylabel('Hidden node')
#     ax2.set_xlabel('Number of bathces')
    ax2.yaxis.tick_right()
    ax2.autoscale_view('tight')
    ax2.set_ylim(ymin=0)
    ax2.set_xlim(xmin=0,xmax=len(Iter))

    ax3.plot(Iter,hiddenLayer,'k-')
#     ax3.set_title('Hidden node evolution')
    ax3.set_ylabel('Hidden layer')
#     ax3.set_xlabel('Number of bathces')
    ax3.yaxis.tick_right()
    ax3.autoscale_view('tight')
    ax3.set_ylim(ymin=0)
    ax3.set_xlim(xmin=0,xmax=len(Iter))

    ax4.plot(Iter,nCluster,'k-')
#     ax4.set_title('Hidden layer evolution')
    ax4.set_ylabel('Cluster')
    ax4.set_xlabel('Number of bathces')
    ax4.yaxis.tick_right()
    ax4.autoscale_view('tight')
    ax4.set_ylim(ymin=0)
    ax4.set_xlim(xmin=0,xmax=len(Iter))

class CustomDatasetFromCSV():
    def __init__(self, csv_path, height, width, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            height (int): image height
            width (int): image width
            transform: pytorch transforms for transforms and tensor conversion
        """
        train = pd.read_csv(csv_path)
        labels = train['label'].values
        labels = torch.from_numpy(labels)
        labels = labels.int()
        
        train.drop('label', axis = 1, inplace = True)
        images = train.values
        images = torch.from_numpy(images)
        
        images = images.double()
        nData = images.shape[0]
        images = images.view(nData,height,width)
        
        images = images.unsqueeze(1)
        
        self.height = height
        self.width = width
        self.transforms = transforms
        
        self.data    = images    #torch.from_numpy(images)
        self.labels  = labels  #torch.from_numpy(labels)
        self.classes = ('0','1','2','3','4','5','6','7','8','9')

    def __getitem__(self, index):
        data  = self.data[index]
        label = self.labels[index]
            
        # Return image and the label
        return (data, label)

    def __len__(self):
        return len(self.data)