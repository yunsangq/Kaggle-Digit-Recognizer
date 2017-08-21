import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import copy

TRAIN_DATA = 'train.csv'


class KaggleDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, csv_path, transform=None, mode=0, valid_size=1000):
        self.mode = mode
        tmp_df = pd.read_csv(csv_path)

        self.transform = transform
        if self.mode == 0: # train
            self.X_train = (tmp_df.ix[:, 1:].values).astype('float32')[valid_size:]
            self.X_train = self.X_train.reshape(len(self.X_train), 28, 28).astype('uint8')
            self.y_train = tmp_df.ix[:, 0].values.astype('int64')[valid_size:].reshape(len(self.X_train), 1)
        elif self.mode == 1: # val
            self.X_train = (tmp_df.ix[:, 1:].values).astype('float32')[:valid_size]
            self.X_train = self.X_train.reshape(len(self.X_train), 28, 28).astype('uint8')
            self.y_train = tmp_df.ix[:, 0].values.astype('int64')[:valid_size].reshape(len(self.X_train), 1)
        else: # test
            self.X_train = (tmp_df.values).astype('float32')
            self.X_train = self.X_train.reshape(len(self.X_train), 28, 28).astype('uint8')

    def __getitem__(self, index):
        img = Image.fromarray(self.X_train[index])
        if self.transform is not None:
            img = self.transform(img)

        if self.mode == 2:
            return img
        else:
            label = torch.from_numpy(self.y_train[index])
            return img, label

    def __len__(self):
        return len(self.X_train)


train_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1309], std=[0.3084])
])

valid_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.1309], std=[0.3084])
])

dset_train = KaggleDataset(TRAIN_DATA, train_transformations)
dset_valid = KaggleDataset(TRAIN_DATA, valid_transformations, mode=1, valid_size=1000)
dset_test = KaggleDataset('test.csv', valid_transformations, mode=2)

train_loader = DataLoader(dset_train,
                          batch_size=128,
                          shuffle=True,
                          num_workers=4)

valid_loader = DataLoader(dset_valid,
                          batch_size=128,
                          shuffle=False,
                          num_workers=1)

test_loader = DataLoader(dset_test,
                         batch_size=100,
                         shuffle=False,
                         num_workers=4)


def save_checkpoint(state, filename='layer.pth.tar'):
    torch.save(state, filename)

'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(256)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # 28*28
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        # 14*14
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.batch_norm4(self.conv4(x))), 2)
        # 7*7
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = F.relu(self.batch_norm7(self.conv7(x)))

        # 1*1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
'''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(128)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm6 = nn.BatchNorm2d(256)
        self.conv7 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.batch_norm7 = nn.BatchNorm2d(256)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        # 28*28
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        # 14*14
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.max_pool2d(F.relu(self.batch_norm4(self.conv4(x))), 2)
        # 7*7
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = F.relu(self.batch_norm7(self.conv7(x)))

        # 1*1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x)

model = Net().cuda()

criterion = nn.CrossEntropyLoss()
optimizer_conv = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)


def exp_lr_scheduler(optimizer, epoch, init_lr=0.01, lr_decay_epoch=20):
    lr = init_lr * (0.1**(epoch//lr_decay_epoch))
    if lr >= 0.00001:
        if epoch % lr_decay_epoch == 0:
            print('LR is set to {}'.format(lr))

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer


def train(_model, _epoch, optimizer, lr_scheduler):
    best_model = _model
    best_acc = 0.0
    for epoch in range(_epoch):
        running_loss = 0.0
        running_corrects = 0.0
        val_corrects = 0.0
        val_loss = 0.0
        train_len = 0
        val_len = 0

        model.train(True)
        optimizer = lr_scheduler(optimizer, epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = Variable(data).cuda(), Variable(target).squeeze().cuda()
            # plt.imshow(data.data.cpu().numpy()[0][0])
            # plt.show()
            optimizer.zero_grad()
            output = _model(data)
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]
            running_corrects += torch.sum(preds == target.data)
            train_len += data.size()[0]

        model.train(False)
        for batch_idx, (data, target) in enumerate(valid_loader):
            data, target = Variable(data).cuda(), Variable(target).squeeze().cuda()
            output = _model(data)
            _, preds = torch.max(output.data, 1)
            loss = criterion(output, target)

            val_loss += loss.data[0]
            val_corrects += torch.sum(preds == target.data)
            val_len += data.size()[0]

        val_loss = val_loss/val_len
        val_acc = val_corrects/val_len

        print('[%d] train_loss: %f train_acc: %.4f, val_loss: %f val_acc: %.4f' %
              (epoch + 1, running_loss/train_len, running_corrects/train_len,
               val_loss, val_acc))

        if val_acc >= best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(_model)
            save_checkpoint(best_model, filename=str(epoch) + '.pth.tar')

    return best_model


# b_model = train(model, 100, optimizer_conv, exp_lr_scheduler)
# save_checkpoint(b_model)


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

test_model = torch.load("10.pth.tar").cuda()
test_model.train(False)
test_len = 0
predictions = np.zeros(28000, dtype=np.int64)

for batch_idx, data in enumerate(test_loader):
    data = Variable(data).cuda()
    output = test_model(data)
    _, preds = torch.max(output.data, 1)
    preds = preds.cpu().numpy()
    preds = preds.squeeze()
    predictions[test_len:test_len+data.size()[0]] = preds
    test_len += data.size()[0]

write_preds(predictions, "result.csv")

