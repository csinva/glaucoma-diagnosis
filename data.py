import numpy as np
from os import listdir
from os.path import join as oj
import sys
import matplotlib.image as mpimg

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path
from scipy.misc import imresize
# pytorch stuff
import torch.utils.data as data
from torchvision import transforms


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = mpimg.imread(filepath).astype(np.float32)
    img = imresize(img, size=(32, 32))
    return np.transpose(img, [2, 0, 1]).tobytes()  # return CHW


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
