import pandas as pd
from torch import np # Torch wrapper for Numpy

import os
from PIL import Image

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

TRAIN_DATA = 'train.csv'

'''
train = pd.read_csv('train.csv')
Y_train = train.ix[:, 0].values.astype('int32')
X_train = (train.ix[:, 1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')
'''

class KaggleDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, transform=None):
        tmp_df = pd.read_csv(csv_path)

        self.transform = transform

        self.X_train = (tmp_df.ix[:, 1:].values).astype('float32')
        scale = np.max(self.X_train)
        self.X_train /= scale
        self.X_train = self.X_train.reshape(42000, 28, 28).astype('uint8')
        self.y_train = tmp_df.ix[:, 0].values.astype('int64').reshape(42000, 1)
        # self.lb = LabelBinarizer()
        # self.tmp = self.lb.fit(range(max(self.y_train)+1))
        # self.y_train = self.lb.transform(self.y_train)

    def __getitem__(self, index):
        img = Image.fromarray(self.X_train[index])
        if self.transform is not None:
            img = self.transform(img)

        label = torch.from_numpy(self.y_train[index])
        return img, label

    def __len__(self):
        return len(self.X_train)

transformations = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

dset_train = KaggleDataset(TRAIN_DATA, transformations)

train_loader = DataLoader(dset_train,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1600, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1) # Flatten layer
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.softmax(x)

model = Net().cuda()

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data).cuda(), Variable(target).cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target.squeeze())
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))


for epoch in range(100):
    train(epoch)
