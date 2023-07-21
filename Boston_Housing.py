import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.boston_housing.load_data(test_split=0.2, seed=85)

mean = X_train.mean(axis=0)
X_train -= mean
std = X_train.std(axis=0)
X_train /= std
X_test -= mean
X_test /= std

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(13,)))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='linear'))

model.compile(loss='mse', optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), metrics=[tf.keras.metrics.MeanAbsoluteError()])
model.summary()

history = model.fit(X_train, y_train, epochs=250, batch_size=64)

test_loss, _ = model.evaluate(X_test, y_test)

print(f"Test Loss {test_loss}")

predict = model.predict(X_test)[0][0]
predict

y_real = y_test[0]
y_real

print(f"Predict value: {predict}, real_value: {y_real}")

plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

test_predictions = model.predict(X_test).flatten()
train_predictions = model.predict(X_train).flatten()

plt.scatter(y_test, test_predictions)
plt.xlabel('True Values [MEDV]')
plt.ylabel('Predictions [MEDV]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])