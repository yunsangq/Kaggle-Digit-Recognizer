import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import copy
import network

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


def save_checkpoint(state, filename='layer.pth.tar'):
    torch.save(state, filename)

model = network.DenseNet(growthRate=12, depth=100, reduction=0.5, bottleneck=True, nClasses=10).cuda()

optimizer = optim.SGD(model.parameters(), lr=1e-1, momentum=0.9, weight_decay=1e-4)

best_model = model
best_acc = 0.0
for epoch in range(500):
    running_loss = 0.0
    running_corrects = 0
    val_corrects = 0

    for i in range(40, 40001, 40):
        inputs = X_train[i-40:i]
        labels = np.array(Y_train[i-40:i], dtype=np.int64)

        inputs = torch.FloatTensor(inputs)
        labels = torch.LongTensor(labels)

        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        optimizer.zero_grad()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        running_corrects += torch.sum(preds == labels.data)

    for j in range(40040, 42001, 40):
        inputs = X_train[j-40:j]
        labels = np.array(Y_train[j-40:j], dtype=np.int64)
        inputs = torch.FloatTensor(inputs)
        labels = torch.LongTensor(labels)
        inputs, labels = Variable(inputs).cuda(), Variable(labels).cuda()

        outputs = model(inputs)
        _, preds = torch.max(outputs.data, 1)
        val_corrects += torch.sum(preds == labels.data)

    val_acc = val_corrects/2000

    print('[%d] loss: %.3f train_acc: %.4f val_acc: %.4f' %
          (epoch + 1, running_loss/42000, running_corrects/42000, val_acc))

    if val_acc >= best_acc:
        best_acc = val_acc
        best_model = copy.deepcopy(model)
        save_checkpoint(best_model, filename=str(epoch) + '.pth.tar')

print('Finished Training')
