from __future__ import print_function

import sys
from os.path import join as oj

import torch
import torch.nn as nn
import torch.optim as optim
# pytorch stuff
import torch.utils.data as data
import torchvision.models as tv_models
from torch.autograd import Variable

# set up data
sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
import data

# transformations = transforms.Compose([transforms.Scale(32), transforms.Normalize(0, 1)])

dset = data.GlaucomaDataset('data')
print('data len', len(dset), 'im shape', dset[0][0].shape)
net = tv_models.alexnet()  # alexnet.alexnet()

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=12,
                                           shuffle=True,
                                           num_workers=4
                                           # pin_memory=True # CUDA only
                                           )

# set up training params
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# train
for epoch in range(2):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # print(inputs.shape)
        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)
        inputs = inputs.type(torch.ByteTensor)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 1999:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')
