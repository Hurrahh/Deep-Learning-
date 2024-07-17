import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import metrics

#--------------------------------------------- TRAINING --------------------------------------------
early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=10)

df = pd.read_csv('https://raw.githubusercontent.com/oluwole-packt/datasets/main/salary_dataset.csv')
df = df.dropna()
df = df.drop(columns=['Name', 'Phone_Number','Date_Of_Birth'])
df = pd.get_dummies(df, drop_first=True)
X = df.drop("Salary",axis=1)
y = df['Salary']

scaler = MinMaxScaler()
X_norm = pd.DataFrame(scaler.fit_transform(X), columns = X.columns)

X_train, X_test, y_train, y_test = train_test_split(X_norm,y,test_size=0.2,random_state=10)


model = Sequential()

model.add(Dense(64, activation='relu', input_shape=[len(X_train.columns)]))
model.add(Dense(64,activation='relu'))
model.add(Dense(1))

model.compile(loss="mae", optimizer="adam", metrics =["mae"])
history = model.fit(X_train,y_train, epochs =1000,callbacks = [early_stopping])

# Prediction
y_preds = model.predict(X_test).flatten()

df_predictions = pd.DataFrame({'Ground_Truth': y_test, 'Model_prediction': y_preds}, columns=[ 'Ground_Truth', 'Model_prediction'])
df_predictions['Model_prediction']= df_predictions['Model_prediction'].astype(int)
df_predictions['diff']=df_predictions['Ground_Truth']-df_predictions['Model_prediction']

model.save('salarypredictor.h5')


# ymin, ymax = None,None
# plt.plot(history.history['loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Number of epochs')
# plt.ylim([ymin, ymax])
# plt.legend(['loss plot'], loc='upper right')
# plt.show()


#------------------------------- PREDICTION ---------------------------------------
@tf.keras.utils.register_keras_serializable()
def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

class CustomMAE(metrics.Metric):
    def __init__(self, name="mae", **kwargs):
        super(CustomMAE, self).__init__(name=name, **kwargs)
        self.mae = tf.keras.metrics.MeanAbsoluteError()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.mae.update_state(y_true, y_pred, sample_weight)

    def result(self):
        return self.mae.result()

    def reset_states(self):
        self.mae.reset_states()

tf.keras.utils.get_custom_objects().update({"mae": CustomMAE})
def salary_predictor(df):
    
    #Use the model salarypredictor.h5
    load_model = tf.keras.models.load_model("../salarypredictor.h5",custom_objects={"mae": mae})

    df_hires= df.drop(columns=['Name', 'Phone_Number','Date_Of_Birth' ])
    df_hires = pd.get_dummies(df_hires, drop_first=True)
    scaler = MinMaxScaler()
    X_norm = pd.DataFrame(scaler.fit_transform(df_hires), columns=df.columns)
    y_preds=load_model.predict(X_norm).flatten()
    df_predictions = pd.DataFrame({ 'Model_prediction': y_preds}, columns=[ 'Model_prediction'])
    df_predictions['Model_prediction']= df_predictions['Model_prediction'].astype(int)
    df['Salary']=df_predictions['Model_prediction']
    return df

test_df = pd.read_csv('new_hires.csv')
test_df = salary_predictor(test_df)



