# mounting dataset from drive
from google.colab import drive
drive.mount('/content/drive')

# importing necessary libraries
import random
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from timeit import default_timer as timer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from keras.utils.np_utils import to_categorical
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dense, Activation, Flatten, Conv2D, MaxPool2D, Dropout
import os
from keras.models import load_model
from keras.utils import load_img
from keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16


# assigning variables to data
all_0 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_0/all"
all_1 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_1/all"
all_2 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_2/all"

hem_0 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_0/hem"
hem_1 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_1/hem"
hem_2 = "/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_2/hem"
path_val ='/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data'
val_labels = pd.read_csv('/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/validation_data/C-NMC_test_prelim_phase_data_labels.csv')


# showing sample data from cancer cell and normal cell
cancer_img = cv2.imread('/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_0/all/UID_11_10_1_all.bmp')
plt.imshow(cancer_img)
plt.title('Cancer')
plt.show()

h_img = cv2.imread('/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_0/hem/UID_H11_10_1_hem.bmp')
plt.imshow(h_img)
plt.title('Normal')
plt.show()


# assigning number of training, validation, and testing samples
train_samples = 800
validation_samples = 200
test_samples = 100


# function to get the list of all the files (images) from a specified folder
def get_path_image(folder):
    image_paths = []
    image_fnames = os.listdir(folder) 
    for img_id in range(len(image_fnames)):
        img = os.path.join(folder,image_fnames[img_id])
        image_paths.append(img)
    
    return image_paths
  
  
# getting the number of cancer images and normal images
# to get a relatively balanced amount of data for each class, only 1 folder from cancerous cells and 2 folders from normal cells are used
cancer_lst = []

for i in [all_0]:
    paths = get_path_image(i)
    cancer_lst.extend(paths)
print('No. of cancer images:', len(cancer_lst))

normal_lst = []
for i in [hem_0,hem_1]:
    paths = get_path_image(i)
    normal_lst.extend(paths)
print('No. of normal images:', len(normal_lst))


# showing the pie chart for data imbalance
cancer_dict = {"x_col":cancer_lst, "y_col":[np.nan for x in range(len(cancer_lst))]}
cancer_dict["y_col"] = "ALL"

normal_dict = {"x_col":normal_lst, "y_col":[np.nan for x in range(len(normal_lst))]}
normal_dict["y_col"] = "HEM"

cancer_df = pd.DataFrame(cancer_dict)
normal_df = pd.DataFrame(normal_dict)

plt.pie([len(cancer_lst),len(normal_lst)],labels=["ALL","Normal"],autopct='%.f')
plt.title('Pie Chart for percentage of each cell type')
plt.show()


# compiling all data that will be used for training (cancer + normal)
train_df = cancer_df.append(normal_df, ignore_index=True)


# compiling validation data
validation_list = get_path_image(path_val)
validation_dict = {"x_col":validation_list ,"y_col":val_labels["labels"]}
validation_df = pd.DataFrame(validation_dict)
validation_df["y_col"].replace(to_replace = [1,0], value = ["ALL","HEM"], inplace = True)


#compiling test data
test_data = '/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/testing_data/C-NMC_test_final_phase_data'
test_list = get_path_image(test_data)
test_dict = {"x_col":test_list}
test_df = pd.DataFrame(test_dict)


# scaling to 256x256
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_dataframe(
                  train_df,
                  x_col = "x_col",
                  y_col = "y_col",
                  target_size = (256, 256),
                 
                  batch_size = 32,
                  color_mode = "rgb",
                  shuffle = True,
                  class_mode = "binary"
)

val_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = val_datagen.flow_from_dataframe(
                  validation_df,
                  x_col = "x_col",
                  y_col = "y_col",
                  target_size = (256, 256),                  
                  batch_size = 32,
                  color_mode = "rgb",
                  shuffle = True,
                  class_mode = "binary")

test_datagen = ImageDataGenerator(rescale=1./255 )
test_generator = test_datagen.flow_from_dataframe(
                  test_df,
                  x_col = "x_col",
                  target_size = (256, 256),
                  color_mode = "rgb",
                  class_mode = None,
                  shuffle = False
)


# creating the cnn structure
model = Sequential()
input_shape = (256, 256, 3)
input_img = Input(shape= input_shape, name = 'img_input')

# every convolutional layer has activation function relu and no padding
model = Conv2D(8,(3,3), padding= 'valid', activation='relu', name = 'layer_1') (input_img) # first layer has 8 filters
model = MaxPool2D((2,2), strides= (2, 2), name = 'layer_2') (model) # using max pooling with stride length 2
model = Dropout(0.25) (model) # dropout function used to reduce chances of overfitting

model = Conv2D(16,(3,3), padding= 'valid', activation='relu', name = 'layer_3') (model)
model = MaxPool2D((2,2), strides= (2, 2), name = 'layer_4') (model)
model = Dropout(0.25) (model)

model = Conv2D(64,(3,3), padding= 'valid', activation='relu', name = 'layer_5') (model)

model = Flatten(name = 'fc_1')(model) # applying fully connected layers
model = Dense(64, name ='layer_6')(model)
model = Dropout(0.5) (model)
model = Dense(1, activation='sigmoid', name='prediction') (model)

model = tf.keras.Model(inputs=input_img, outputs= model) # model takes input image and outputs result from model

model.summary()

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', 'Recall'])


# training the model
start = timer()
batch_size = 32

history = model.fit(train_generator, steps_per_epoch=train_samples // batch_size, epochs=10, validation_data=validation_generator, validation_steps=validation_samples // batch_size)

end = timer()
elapsed = end - start
print('Total Time Elapsed: ', int(elapsed//60), ' minutes ', (round(elapsed%60)), ' seconds')


# plotting training accuracy vs validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')


#save the model
model.save('model.h5')


# testing the model
model = load_model('/content/model.h5')

image = load_img('/content/drive/Shareddrives/Deep Learning/C-NMC_Leukemia/training_data/fold_0/all/UID_11_10_1_all.bmp', 
                 target_size=(256, 256))

img = np.array(image)
img = img/255.0
img = img.reshape(1, 256, 256, 3)
label = model.predict(img)
plt.imshow(image)

print("0 - Normal, 1 - Leukemia") # this means that the closer it is to 1, the more likely it is to be a Leukemia cell, while the closer it is to 0, the more likely it is to be a normal cell
print("Result: ", label[0][0])


