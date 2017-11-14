import numpy as np
from torchvision import transforms
from os import listdir
from PIL import Image
from os.path import join as oj
import sys
import matplotlib.image as mpimg

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path

# pytorch stuff
import torch.utils.data as data
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from models import alexnet
import torchvision.models as tv_models


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = mpimg.imread(filepath).astype(np.float32)
    return np.transpose(img, [2, 0, 1]) # return CHW


class Glaucoma_dset(data.Dataset):
    def __init__(self, data_dir, input_transform=None):
        super(Glaucoma_dset, self).__init__()
        fnames_control = [oj(data_dir, 'control', x)
                          for x in listdir(oj(data_dir, 'control'))
                          if is_image_file(x)]
        fnames_glaucoma = [oj(data_dir, 'glaucoma', x)
                           for x in listdir(oj(data_dir, 'glaucoma'))
                           if is_image_file(x)]
        self.fnames = np.array(fnames_control + fnames_glaucoma)
        self.labels = np.hstack((np.ones(len(fnames_control)) * -1,
                                 np.ones(len(fnames_glaucoma)) * 1))

        self.input_transform = input_transform

    def __getitem__(self, index):
        input = load_img(self.fnames[index])
        label = self.labels[index]
        if self.input_transform:
            input = self.input_transform(input)
        return input, label

    def __len__(self):
        return self.fnames.size


# transformations = transforms.Compose([transforms.Scale(32), transforms.ToTensor()])

dset = Glaucoma_dset("data")
print("data len", len(dset), "im shape", dset[0][0].shape)
net = tv_models.alexnet() # alexnet.alexnet()

train_loader = torch.utils.data.DataLoader(dset,
                                           batch_size=10,
                                           shuffle=True,
                                           num_workers=4
                                           # pin_memory=True # CUDA only
                                           )

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data

        # wrap them in Variable
        inputs, labels = Variable(inputs), Variable(labels)

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
