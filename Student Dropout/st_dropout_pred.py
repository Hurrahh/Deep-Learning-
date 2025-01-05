import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import keras

early_stopping = keras.callbacks.EarlyStopping(
    monitor = 'accuracy',
    patience = 15
)
df = pd.read_csv('https://raw.githubusercontent.com/oluwole-packt/datasets/main/Students-Dropout-Prediction.csv', index_col=0)
df.drop(columns=['acad_year'], inplace=True)
df = pd.get_dummies(df,drop_first=True)


X = df.drop("graduate",axis=1)
y = df["graduate"]

scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

X_train,X_test,y_train,y_test = train_test_split(X_norm,y,test_size=0.2,random_state=15)


model = Sequential()
model.add(Dense(64,activation='relu',input_dim = len(X_train.columns)))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics = ["accuracy"])
history = model.fit(X_train,y_train,epochs = 500,validation_split=0.2,callbacks = early_stopping)


# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Number of epochs')
# plt.legend(['loss plot'], loc='upper right')
# plt.show()


y_pred=model.predict(X_test).flatten()
y_pred = np.round(y_pred).astype('int')
df_predictions = pd.DataFrame(
    {'Ground_Truth': y_test, 'Model_prediction': y_pred},
    columns=[ 'Ground_Truth', 'Model_prediction'])
len(df_predictions[(df_predictions[
    'Ground_Truth']!=df_predictions['Model_prediction'])])

print(df_predictions.sample(20))
