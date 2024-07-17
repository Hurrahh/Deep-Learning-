import os
import tensorflow as tf
import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten
from keras import Sequential


train_labels = test_labels = val_labels = os.listdir('./weather dataset/train')

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, verbose=1, restore_best_weights=True)

train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

train_dir = "./weather dataset/train/"
val_dir = "./weather dataset/validation/"
test_dir = "./weather dataset/test/"

train_data = train_datagen.flow_from_directory(train_dir,
                                               batch_size=64,
                                               target_size=(224,224),
                                               class_mode='categorical')

valid_data = valid_datagen.flow_from_directory(val_dir,
                                               batch_size=64,
                                               target_size=(224,224),
                                               class_mode="categorical")
test_data = valid_datagen.flow_from_directory(test_dir,
                                               batch_size=64,
                                               target_size=(224,224),
                                               class_mode='categorical')

data_augmentation = keras.Sequential(
  [
    tf.keras.layers.RandomFlip("horizontal",input_shape=(224,224,3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
  ])

model = Sequential()
model.add(data_augmentation)
model.add(Conv2D(filters=16, kernel_size=3, activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(32, 3, activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Conv2D(64, 3, activation="relu"))
model.add(MaxPooling2D(2,2))
model.add(Flatten())
model.add(Dense(1050,activation='relu'))
model.add(Dense(4,activation='softmax'))

model.compile(loss="CategoricalCrossentropy",optimizer=tf.keras.optimizers.Adam(name='adam',learning_rate=0.001),metrics=["accuracy"])

model.fit(train_data,epochs=20,validation_data=valid_data,callbacks=early_stopping)

test_loss,test_accu = model.evaluate(test_data)
print(test_accu)