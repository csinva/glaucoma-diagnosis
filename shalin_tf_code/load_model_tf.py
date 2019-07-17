from __future__ import print_function

import os

import cv2
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder


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
                    img_file = np.array(Image.fromarray(img_file).resize((175, 175)).convert(3))
                    img_arr = np.asarray(img_file)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


X_train, y_train = get_data(os.path.join('images', 'TRAIN'))
X_test, y_test = get_data(os.path.join('images', 'TEST'))

encoder = LabelEncoder()
encoder.fit(y_train)
y_test = encoder.transform(y_test)

# # Accuracy

# In[13]:

from sklearn.metrics import accuracy_score

print('Predicting on test data')
y_pred = np.rint(model.predict(X_test))

print(accuracy_score(y_test, y_pred))

# # Confusion Matrix

# In[14]:

from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

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
