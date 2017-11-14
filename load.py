import numpy as np

import os
import cv2
import scipy
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import torch.utils.data as data
from torchvision import transforms
from os import listdir
from PIL import Image
from os.path import join as oj
import sys

sys.path.insert(1, oj(sys.path[0], '..'))  # insert parent path


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('YCbCr')
    y, _, _ = img.split()
    return y


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
        input = load_img(self.image_filenames[index])
        label = self.labels[index]
        if self.input_transform:
            input = self.input_transform(input)
        return input, label

    def __len__(self):
        return self.fnames.size


# transformations = transforms.Compose([transforms.Scale(32), transforms.ToTensor()])

dset = Glaucoma_dset("data")
print(len(dset))
exit(0)

train_loader = DataLoader(dset,
                          batch_size=256,
                          shuffle=True,
                          num_workers=4
                          # pin_memory=True # CUDA only
                          )


def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []

    for img_type in os.listdir(folder):
        if not img_type.startswith('.'):
            if img_type in ['GLAUCOMA']:
                label = 'GLAUCOMA'
            else:
                label = 'NOT_GLAUCOMA'
            for image_filename in os.listdir(folder + img_type):
                img_file = cv2.imread(folder + img_type + '/' + image_filename)
                if img_file is not None:
                    # Downsample the image to 120, 160, 3
                    img_file = scipy.misc.imresize(arr=img_file, size=(175, 175, 3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


    # X_train, y_train = get_data('images/TRAIN/')
    # X_test, y_test = get_data('images/TEST/')
    #
    #
    # encoder = LabelEncoder()
    # encoder.fit(y_train)
    # y_test = encoder.transform(y_test)




    # # Accuracy

    # In[13]:

    # from sklearn.metrics import accuracy_score
    #
    # print('Predicting on test data')
    # y_pred = np.rint(model.predict(X_test))
    #
    # print(accuracy_score(y_test, y_pred))



    # # Confusion Matrix

    # In[14]:

    # from sklearn.metrics import confusion_matrix
    #
    # print(confusion_matrix(y_test, y_pred))

    # # Confusion Matrix

    # In[14]:






    # # dimensions of our images
    # img_width, img_height = 175, 175

    # # load the model we saved
    # model = load_model('saved_model.h5')
    # model.compile(loss='binary_crossentropy',
    #               optimizer='rmsprop',
    #               metrics=['accuracy'])

    # # predicting images
    # img = image.load_img('73.jpg', target_size=(img_width, img_height))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)

    # images = np.vstack([x])
    # classes = model.predict_classes(images, batch_size=10)
    # print classes

    # # predicting multiple images at once
    # img = image.load_img('74.jpg', target_size=(img_width, img_height))
    # y = image.img_to_array(img)
    # y = np.expand_dims(y, axis=0)

    # # pass the list of multiple images np.vstack()
    # images = np.vstack([x, y])
    # classes = model.predict_classes(images, batch_size=10)

    # # print the classes, the images belong to
    # print classes
    # print classes[0]
    # print classes[0][0]
