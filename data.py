import sys
from os import listdir
from os.path import join as oj

import matplotlib.image as mpimg
import numpy as np
# pytorch stuff
import torch.utils.data as data
from PIL import Image

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg'])


def load_img(filepath):
    img = mpimg.imread(filepath).astype(np.float32)
    img = np.array(Image.fromarray(img).resize((32, 32)))
    return np.transpose(img, [2, 0, 1]).tobytes()  # return CHW


class GlaucomaDataset(data.Dataset):
    def __init__(self, data_dir, input_transform=None):
        super(GlaucomaDataset, self).__init__()
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
        img = load_img(self.fnames[index])
        label = self.labels[index]
        if self.input_transform:
            img = self.input_transform(img)
        return img, label

    def __len__(self):
        return self.fnames.size
