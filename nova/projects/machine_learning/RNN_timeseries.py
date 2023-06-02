import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

mpl.rcParams["figure.figsize"] = (8, 6)
mpl.rcParams["axes.grid"] = False

read_file = False
zip_path = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip",
    fname="jena_climate_2009_2016.csv.zip",
    extract=False,
)
csv_path = os.path.splitext(zip_path)[0]
hfd_path = os.path.splitext(csv_path)[0] + ".hdf"
if read_file or not os.path.isfile(hfd_path):
    df = pd.read_csv(csv_path)
    df.to_hdf(hfd_path, "jena")
else:
    df = pd.read_hdf(hfd_path, "jena")


def sample(dataset, start_index, end_index, history_length, target_length):
    start_index += history_length
    if end_index is None:
        end_index = len(dataset) - target_length
    n_index = end_index - start_index
    data = np.zeros((n_index, history_length, 1))
    target = np.zeros((n_index, 1, 1))
    for i, j in enumerate(range(start_index, end_index)):
        index = range(j - history_length, j)
        data[i, :, 0] = dataset[index]
        target[i, 0, 0] = dataset[j + target_length]
    return data, target


N = 10000
window = 3

uni_data = df["T (degC)"].values[:10000]
train_split = int(0.6 * N)

mean = uni_data[:train_split].mean()
std = uni_data[:train_split].std()

uni_data = (uni_data - mean) / std

x_train, y_train = sample(uni_data, 0, train_split, window, 0)
x_val, y_val = sample(uni_data, train_split, None, window, 0)

"""
t = range(-300, 0)
plt.plot(t, x_train[0])
plt.plot(0, y_train[0], 'rX')
"""

batch = 10
# buffer = 10000

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.cache().batch(batch, drop_remainder=True).repeat()  # .shuffle(buffer)

val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val = val.batch(batch, drop_remainder=True).repeat()


model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(
            8,
            input_shape=x_train.shape[-2:],
            stateful=True,
            batch_input_shape=(batch, window, 1),
        ),
        tf.keras.layers.Dense(1),
    ]
)

model.compile(optimizer="adam", loss="mae")

history = model.fit(train, epochs=10, steps_per_epoch=200)
# , validation_data=val, validation_steps=1

# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])


for __ in range(2):
    plt.plot(model.predict(val.take(1)))

    plt.plot(model.predict(train.take(1)))
