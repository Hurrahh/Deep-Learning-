import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,BatchNormalization,Dropout
from tensorflow import keras
import matplotlib.pyplot as plt



early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, verbose=1, restore_best_weights=True)


data_dir = "../PetImages"
cat_dir = os.path.join(data_dir, "Cat")
dog_dir = os.path.join(data_dir, "Dog")

cat_images = [os.path.join(cat_dir, fname) for fname in os.listdir(cat_dir) if fname.endswith('.jpg')]
dog_images = [os.path.join(dog_dir, fname) for fname in os.listdir(dog_dir) if fname.endswith('.jpg')]

all_images = cat_images + dog_images
labels = ['0'] * len(cat_images) + ['1'] * len(dog_images)

train_images, test_images, train_labels, test_labels = train_test_split(all_images, labels, test_size=0.2, random_state=40, stratify=labels)

def create_dataframe(filepaths, labels):
    return pd.DataFrame({
        'filename': filepaths,
        'class': labels
    })

train_df = create_dataframe(train_images, train_labels)
test_df = create_dataframe(test_images, test_labels)

train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    target_size=(256,256),
    batch_size=100,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='filename',
    y_col='class',
    target_size=(256,256),
    batch_size=100,
    class_mode='binary'
)

model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(64,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Conv2D(128,kernel_size=(3,3),padding='valid',activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(train_generator, epochs=10, validation_data=test_generator, callbacks = early_stopping,verbose=1)

model.save('cat_dog_classifier.h5')

scores = model.evaluate(test_generator)
print(f'Test accuracy: {scores[1]*100:.2f}%')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))
plt.plot(epochs, acc, 'r', "Training Accuracy")
plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
plt.title('Training and validation accuracy')
plt.show()
print("")

plt.plot(epochs, loss, 'r', "Training Loss")
plt.plot(epochs, val_loss, 'b', "Validation Loss")
plt.show()