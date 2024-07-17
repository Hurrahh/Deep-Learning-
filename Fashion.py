import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten


early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',patience=5, verbose=1, restore_best_weights=True)

fashion_data = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_data.load_data()

train_images = train_images.reshape(train_images.shape[0],28,28,1).astype('float32')
test_images = test_images.reshape(test_images.shape[0],28,28,1).astype('float32')

train_images = train_images/255.0
test_images = test_images/255.0

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

model = Sequential()

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=keras.optimizers.Adam(name="adam",learning_rate=0.001), metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=50, validation_split=0.2,callbacks=early_stopping)

test_loss,test_accu = model.evaluate(test_images,test_labels)
print("Accuracy is: ", test_accu)


















# class_names=['Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankleboot']
# def model(learning_rate):
#     model=keras.Sequential([
#                             keras.layers.Flatten(input_shape=(28,28)),
#                             keras.layers.Dense(100, activation="relu"),
#                             # keras.layers.Dense(64, activation="relu"),
#                             keras.layers.Dense(10,activation="softmax")]
#     )
#
#     model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
#                   loss='sparse_categorical_crossentropy',
#                   metrics=['accuracy'])
#
#     model.fit(train_images, train_labels, epochs=100,callbacks=early_stopping,validation_split=0.2)
#     test_loss, test_acc = model.evaluate(test_images, test_labels)
#     return test_acc
#
#
# learning_rates = [1.0, 0.1, 0.01, 0.001, 0.0001, 0.00001, 0.000001]
# accuracies= []
#
# for i in learning_rates:
#     accu = model(i)
#     accuracies.append(accu)
#
#
# result = pd.DataFrame(list(zip(learning_rates,accuracies)),columns=["Learning_rates", "Test_Accuracy"])
# print(result)