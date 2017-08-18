import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import network


def write_preds(preds, fname):
    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

net = torch.load("8.pth.tar").cuda()

train = pd.read_csv('train.csv')
Y_train = train.ix[:,0].values.astype('int32')
X_train = (train.ix[:,1:].values).astype('float32')
X_test = (pd.read_csv('test.csv').values).astype('float32')

scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

X_train = X_train.reshape(42000, 1, 28, 28)
X_test = X_test.reshape(28000, 1, 28, 28)
'''
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

# 0.
