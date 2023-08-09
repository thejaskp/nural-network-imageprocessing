import tensorflow as tf
from tensorflow import keras

import pandas as pd
import numpy as np

train_df = pd.read_csv('./data/train.csv')
one_hot_color = pd.get_dummies(train_df.color).values
one_hot_marker = pd.get_dummies(train_df.marker).values

labels = np.concatenate((one_hot_color, one_hot_marker), axis=1)

print(labels[0])

print(train_df.head())

model = keras.Sequential([
    keras.layers.Dense(128, input_shape=(2,), activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(9, activation='sigmoid')])

model.compile(optimizer='adam',
              loss=keras.losses.CategoricalFocalCrossentropy(from_logits=True),
              metrics=['accuracy'])

x = np.column_stack((train_df.x.values, train_df.y.values))

np.random.RandomState(seed=10).shuffle(x)
np.random.RandomState(seed=10).shuffle(labels)

model.fit(x, labels, batch_size=1, epochs=5)

test_df = pd.read_csv('./data/test.csv')
test_x = np.column_stack((test_df.x.values, test_df.y.values))

test_one_hot_color = pd.get_dummies(test_df.color).values
test_one_hot_marker = pd.get_dummies(test_df.marker).values

labels = np.concatenate((test_one_hot_color, test_one_hot_marker), axis=1)

print('EVALUATE')
model.evaluate(test_x, test_df.color.values)

print("Prediction: ", np.round(model.predict(np.array([[0, 3]]))))
