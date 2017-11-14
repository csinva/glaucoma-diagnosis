import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras.layers import Dense
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
import cv2
import scipy
import os
# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt



epochs = 20
# BASE_DIR = '../'
batch_size = 32


def get_model():
    model = Sequential()
    model.add(Lambda(lambda x: x * 1./255., input_shape=(175, 175, 3), output_shape=(175, 175, 3)))
    model.add(Conv2D(32, (3, 3), input_shape=(175, 175, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.7))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])

    return model






model = get_model()
print(model.summary())







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
    return X,y



X_train, y_train = get_data('images/TRAIN/')
X_test, y_test = get_data('images/TEST/')

encoder = LabelEncoder()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)



model = get_model()

# fits the model on batches
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=epochs,
    shuffle=True,
    batch_size=batch_size)


model.save_weights('binary_model.h5')
model.save('saved_model.h5')


# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights('binary_model.h5')
# print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")









def plot_learning_curve(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    plt.clf()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')

plot_learning_curve(history)


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








# # Images Misclassified

# In[14]:

false_positive = np.intersect1d(np.where(y_pred == 1), np.where(y_test == 0))


# In[31]:


for i in false_positive:
	img = X_test[false_positive[i]]
	plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# # Mononuclear Cells Classified Correctly

# In[16]:

true_positive_glaucoma = np.intersect1d(np.where(y_pred == 1), np.where(y_test == 1))


# In[32]:

img = X_test[true_positive_glaucoma[0]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[33]:

img = X_test[true_positive_glaucoma[5]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[34]:

img = X_test[true_positive_glaucoma[8]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# # Polynuclear Cells Classified Correctly

# In[22]:

true_positive_not_glaucoma = np.intersect1d(np.where(y_pred == 0), np.where(y_test == 0))


# In[27]:

img = X_test[true_positive_not_glaucoma[21]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[29]:

img = X_test[true_positive_not_glaucoma[53]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[30]:

img = X_test[true_positive_not_glaucoma[16]]
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

