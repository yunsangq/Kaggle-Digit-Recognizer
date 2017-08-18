import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)


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

        self.conv8 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_norm8 = nn.BatchNorm2d(512)
        self.conv9 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm9 = nn.BatchNorm2d(512)
        self.conv10 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm10 = nn.BatchNorm2d(512)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        # 28*28
        x = F.relu(self.batch_norm1(self.conv1(x)))
        x = F.max_pool2d(F.relu(self.batch_norm2(self.conv2(x))), 2)
        # 14*14
        x = F.relu(self.batch_norm3(self.conv3(x)))
        x = F.avg_pool2d(F.relu(self.batch_norm4(self.conv4(x))), 2)
        # 7*7
        x = F.relu(self.batch_norm5(self.conv5(x)))
        x = F.relu(self.batch_norm6(self.conv6(x)))
        x = F.relu(self.batch_norm7(self.conv7(x)))

        x = F.relu(self.batch_norm8(self.conv8(x)))
        x = F.relu(self.batch_norm9(self.conv9(x)))
        x = F.relu(self.batch_norm10(self.conv10(x)))

        # 1*1
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


net = torch.load("8.pth.tar").cuda()

train = pd.read_csv('train.csv')
Y_train = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

'''
for i in range(10):
    image = Image.fromarray(X_train[i])
    plt.imshow(image)
    plt.show()
'''
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

X_train = X_train.reshape(42000, 1, 28, 28)
X_test = X_test.reshape(28000, 1, 28, 28)

print('Validation...')

inputs = X_train[40000:42000]
labels = np.array(Y_train[40000:42000], dtype=np.int64)
inputs = torch.FloatTensor(inputs)
labels = torch.LongTensor(labels)
inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()
outputs = net(inputs)
_, preds = torch.max(outputs.data, 1)
val_corrects = torch.sum(preds == labels.data)
val_acc = val_corrects/2000
print(val_acc)
'''
print('Testing...')

predictions = np.zeros(28000, dtype=np.int64)

for i in range(40, 28001, 40):
    inputs = X_test[i-40:i]
    inputs = torch.FloatTensor(inputs)
    inputs = Variable(inputs).cuda()
    outputs = net(inputs)
    _, preds = torch.max(outputs.data, 1)
    preds = preds.cpu().numpy()
    preds = preds.squeeze()
    predictions[i-40:i] = preds

write_preds(predictions, "result.csv")
'''
# 0.99300
