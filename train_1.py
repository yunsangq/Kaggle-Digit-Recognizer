import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms, models
import matplotlib.pyplot as plt
from PIL import Image
import copy

train = pd.read_csv('train.csv')
Y_train = train.ix[:, 0].values.astype('int32')
X_train = (train.ix[:, 1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

X_train = X_train.reshape(42000, 28, 28)
Y_train = np.array(Y_train, dtype=np.int64)

X_train = torch.FloatTensor(X_train)
Y_train = torch.LongTensor(Y_train)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.001, lr_decay_epoch=30):
    lr = init_lr * (0.1**(epoch//lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def save_checkpoint(state, filename='layer.pth.tar'):
    torch.save(state, filename)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.flatten = nn.Linear(64*7*7, 256)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.dropout(x, p=0.25, training=self.training)

        x = F.relu(self.conv3(x))
        x = F.max_pool2d(F.relu(self.conv4(x)), 2, stride=2)
        x = F.dropout(x, p=0.25, training=self.training)

        x = x.view(-1, 64*7*7)
        x = F.relu(self.flatten(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.softmax(self.fc(x))
        return x


def train_model(model, criterion, optimizer, lr_scheduler, num_epochs):
    best_model = model
    best_acc = 0.0

    for epoch in range(1000):
        optimizer = lr_scheduler(optimizer, epoch)
        running_loss = 0.0
        running_corrects = 0.0
        val_corrects = 0.0
        val_loss = 0.0
        train_len = 0
        val_len = 0
        model.training = True


        model.training = False

        print('[%d] loss: %f train_acc: %.4f val_loss: %f val_acc: %.4f' %
              (epoch + 1, running_loss / 41800, running_corrects / 41800, val_loss, val_acc))
        '''
        if val_acc >= best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
            save_checkpoint(best_model, filename=str(epoch) + '.pth.tar')
        '''
    print('Finished Training')
    return best_model

model = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

best = train_model(model, criterion, optimizer_conv, exp_lr_scheduler, num_epochs=500)

save_checkpoint(best)
