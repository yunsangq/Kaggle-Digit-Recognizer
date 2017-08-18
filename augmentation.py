import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import copy

data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])


train = pd.read_csv('train.csv')
Y_train = train.ix[:, 0].values.astype('int32')
X_train = (train.ix[:, 1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

X_train = X_train.reshape(42000, 28, 28).astype(np.int32)
Y_train = np.array(Y_train, dtype=np.int64)



for i in range(len(X_train)):
    img = Image.fromarray(X_train[i])
    s = data_transforms(img)
    # image = Image.fromarray(s[0])
    # plt.imshow(image)
    # plt.show()

# X_train = torch.FloatTensor(X_train)
# Y_train = torch.LongTensor(Y_train)

'''
class trainset(object):
    x_train = None
    y_train = None


train = trainset()
train.x_train = X_train
train.y_train = Y_train

trainloader = torch.utils.data.DataLoader(train, )
'''