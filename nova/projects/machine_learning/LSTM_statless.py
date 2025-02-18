# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 11:35:18 2019

@author: mcintos
"""

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# ----------------------------------------------------------
# EDITABLE PARAMETERS
# Read the documentation in the script head for more details
# ----------------------------------------------------------

# length of input
input_len = 1000

# The window length of the moving average used to generate
# the output from the input in the input/output pair used
# to train the LSTM
# e.g. if tsteps=2 and input=[1, 2, 3, 4, 5],
#      then output=[1.5, 2.5, 3.5, 4.5]
tsteps = 2

# The input sequence length that the LSTM is trained on for each output point
lahead = 10

# training parameters passed to "model.fit(...)"
batch_size = 1
epochs = 10

# ------------
# MAIN PROGRAM
# ------------

print("*" * 33)
if lahead >= tsteps:
    print("STATELESS LSTM WILL ALSO CONVERGE")
else:
    print("STATELESS LSTM WILL NOT CONVERGE")
print("*" * 33)

np.random.seed(1986)

print("Generating Data...")


def gen_uniform_amp(amp=1, xn=10000):
    """Generates uniform random data between
    -amp and +amp
    and of length xn

    # Arguments
        amp: maximum/minimum range of uniform data
        xn: length of series
    """
    data_input = np.random.uniform(-1 * amp, +1 * amp, xn)
    data_input = pd.DataFrame(data_input)
    return data_input


# Since the output is a moving average of the input,
# the first few points of output will be NaN
# and will be dropped from the generated data
# before training the LSTM.
# Also, when lahead > 1,
# the preprocessing step later of "rolling window view"
# will also cause some points to be lost.
# For aesthetic reasons,
# in order to maintain generated data length = input_len after pre-processing,
# add a few points to account for the values that will be lost.
to_drop = max(tsteps - 1, lahead - 1)
data_input = gen_uniform_amp(amp=0.1, xn=input_len + to_drop)

# set the target to be a N-point average of the input
expected_output = data_input.rolling(window=tsteps, center=False).mean()

# when lahead > 1, need to convert the input to "rolling window view"
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
if lahead > 1:
    data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
    data_input = pd.DataFrame(data_input)
    for i, c in enumerate(data_input.columns):
        data_input[c] = data_input[c].shift(i)

# drop the nan
expected_output = expected_output[to_drop:]
data_input = data_input[to_drop:]

print("Input shape:", data_input.shape)
print("Output shape:", expected_output.shape)
print("Input head: ")
print(data_input.head())
print("Output head: ")
print(expected_output.head())
print("Input tail: ")
print(data_input.tail())
print("Output tail: ")
print(expected_output.tail())

print("Plotting input and expected output")
plt.plot(data_input[0][:10], ".")
plt.plot(expected_output[0][:10], "-")
plt.legend(["Input", "Expected output"])
plt.title("Input")
plt.show()


def create_model(stateful):
    model = Sequential()
    model.add(
        LSTM(20, input_shape=(lahead, 1), batch_size=batch_size, stateful=stateful)
    )
    model.add(Dense(1))
    model.compile(loss="mse", optimizer="adam")
    return model


print("Creating Stateful Model...")
model_stateful = create_model(stateful=True)


# split train/test data
def split_data(x, y, ratio=0.8):
    to_train = int(input_len * ratio)
    # tweak to match with batch_size
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # tweak to match with batch_size
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[: -1 * to_drop]
        y_test = y_test[: -1 * to_drop]

    # some reshaping
    def reshape_3(x):
        return x.values.reshape((x.shape[0], x.shape[1], 1))

    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    def reshape_2(x):
        return x.values.reshape((x.shape[0], 1))

    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
print("x_train.shape: ", x_train.shape)
print("y_train.shape: ", y_train.shape)
print("x_test.shape: ", x_test.shape)
print("y_test.shape: ", y_test.shape)

print("Training")
for i in range(epochs):
    print("Epoch", i + 1, "/", epochs)
    # Note that the last state for sample i in a batch will
    # be used as initial state for sample i in the next batch.
    # Thus we are simultaneously training on batch_size series with
    # lower resolution than the original series contained in data_input.
    # Each of these series are offset by one step and can be
    # extracted with data_input[i::batch_size].
    model_stateful.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=1,
        verbose=1,
        validation_data=(x_test, y_test),
        shuffle=False,
    )
    model_stateful.reset_states()

print("Predicting")
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

print("Creating Stateless Model...")
model_stateless = create_model(stateful=False)

print("Training")
model_stateless.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test),
    shuffle=False,
)

print("Predicting")
predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)

# ----------------------------

print("Plotting Results")
plt.subplot(3, 1, 1)
plt.plot(y_test)
plt.title("Expected")
plt.subplot(3, 1, 2)
# drop the first "tsteps-1" because it is not possible to predict them
# since the "previous" timesteps to use do not exist
plt.plot((y_test - predicted_stateful).flatten()[tsteps - 1 :])
plt.title("Stateful: Expected - Predicted")
plt.subplot(3, 1, 3)
plt.plot((y_test - predicted_stateless).flatten())
plt.title("Stateless: Expected - Predicted")
plt.show()

sl = model_stateless.predict(x_test[tsteps - 1 : tsteps + 2, :])

plt.figure()
nplot = 5
t = -np.arange(nplot)
plt.plot(t, y_test.flatten()[tsteps - 1 :][:nplot], "k-", label="gt")
# plt.plot(t, predicted_stateful.flatten()[tsteps - 1:][:nplot], 'ro-', label='stateful')
plt.plot(
    t, predicted_stateless.flatten()[tsteps - 1 :][:nplot], "k--", label="stateless"
)


for i in range(nplot):
    plt.plot(np.arange(10) - 10 - i, x_test[i, :], f"C{i}")
    plt.plot(
        np.arange(2) - 1 - i,
        [x_test[i, -1], predicted_stateless.flatten()[tsteps - 1 :][i]],
        # y_test.flatten()[tsteps - 1:][i]],
        f"C{i}o-",
        label=f"n{i}",
    )
plt.legend()
