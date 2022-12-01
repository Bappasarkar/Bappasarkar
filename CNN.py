import tensorflow as tf
from keras.preprocessing import image
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Conv2D

#       **********Train data generator [Generates train sample from images]********** 

train_datagen = image.ImageDataGenerator(
    rescale=1.0/255.0,
    shear_range = 0.05,
    zoom_range = 0.1,
    horizontal_flip = True,
    fill_mode='constant',
    cval=0
)

#       ***********Validation data generator [Generates validation sample]**********

validation_datagen = image.ImageDataGenerator(
    rescale = 1./255
)

#       ***********Get images from drive and generate***********

train = train_datagen.flow_from_directory(
    '/home/kawsar/aA-WORKING/archive/TrainDataFolder/Brinjal/train',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

validation = validation_datagen.flow_from_directory(
    '/home/kawsar/aA-WORKING/archive/TrainDataFolder/Brinjal/val',
    target_size = (224, 224),
    batch_size = 32,
    class_mode = 'categorical'
)

test = validation_datagen.flow_from_directory(
    '/home/kawsar/aA-WORKING/archive/TrainDataFolder/Brinjal/test',
    target_size = (224, 224),
    batch_size = 32,
    shuffle = False,
    class_mode = 'categorical'
)

print('Classes: \n',test.class_indices)


model = tf.keras.models.Sequential()
model.add(Conv2D(16, (3, 3),input_shape=(224,224,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(32, (5, 5)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(64, (7, 7)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))
model.add(Conv2D(128, (7, 7)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048))
model.add(Activation("relu"))
model.add(Dropout(0.7))
model.add(Dense(512))
model.add(Activation("relu"))
model.add(Dropout(0.8))
model.add(Dense(2))
model.add(Activation('sigmoid'))


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer = opt,
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)
model.summary()
history = model.fit(
    x = train, 
    validation_data = validation,
    #initial_epoch=6,
    epochs = 20
)

# Evaluate the loss and accuracy
loss, accuracy = model.evaluate(test)

# Print the accuracy
print("Accuracy: " + str(accuracy))
# Print the loss
print("Loss: " + str(loss))

#Save the model if it pass the requirements only!!
model.save('/home/kawsar/aA-WORKING/archive/TrainDataFolder/Pepper/LeafRecog.h')

import numpy as np
from sklearn import metrics

predictions = model.predict_generator(test)
predicted_classes = np.argmax(predictions, axis = 1)
true_classes = test.classes
class_labels = ['Brinjal_Healthy', 'Pepper__bell___healthy','Potato___healthy','Tomato_healthy']
report = metrics.classification_report(true_classes, predicted_classes, target_names = class_labels)
print(report)

import matplotlib.pyplot as plt

acc=history.history['accuracy']
loss=history.history['loss']
plt.plot(acc)
plt.plot(loss)
plt.show()

val_acc=history.history['val_accuracy']
val_loss=history.history['val_loss']
plt.plot(val_acc)
plt.plot(val_loss)
plt.show()

plt.plot(loss)
plt.plot(val_loss)
plt.show()

plt.plot(acc)
plt.plot(val_acc)
plt.show()

import seaborn as sn
import pandas as pd

cm = metrics.confusion_matrix(true_classes, predicted_classes)
df_cm = pd.DataFrame(cm, index = [i for i in ['Brinjal_Healthy', 'Pepper__bell___healthy','Potato___healthy','Tomato_healthy']], columns = [i for i in ['Brinjal_Healthy', 'Pepper__bell___healthy','Potato___healthy','Tomato_healthy']])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, fmt='d')
plt.show()

# from os import listdir
# root='/home/kawsar/aA-WORKING/archive/TrainDataFolder/Potato/test/Potato___healthy/'
# for i in listdir('/home/kawsar/aA-WORKING/archive/TrainDataFolder/Potato/test/Potato___healthy/'):
#     img= image.load_img(root+i,target_size=(224,224))
#     plt.imshow(img)
#     plt.show()
#     X=image.img_to_array(img)/255.0
#     X=np.expand_dims(X,axis=0)
#     val=model.predict([X])
#     val=np.argmax(val)
#     if val==0:
#       print('Diseased')
#     if val==1:
#       print('Diseased2')
#     else:
#       print('Healthy')