import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=True)
        x = self.fc2(x)
        return F.log_softmax(x)


net = torch.load("first.pth.tar").cuda()

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

print('Testing...')

inputs = X_test
inputs = torch.FloatTensor(inputs)
inputs = Variable(inputs).cuda()
outputs = net(inputs)
_, preds = torch.max(outputs.data, 1)
preds = preds.cpu().numpy()
preds = preds.squeeze()


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

write_preds(preds, "result.csv")
