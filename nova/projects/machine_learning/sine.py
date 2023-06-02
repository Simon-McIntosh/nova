import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
from amigo.pyplot import plt

tf.keras.backend.clear_session()  # reset state

N = 100
f = 0.5
t = np.linspace(0, 4, N + 1)
y = np.sin(t * f * 2 * np.pi)
x_train = y[:-1].reshape(-1, 1, 1)
y_train = y[1:]  # .reshape(-1, 1, 1)

inputs = Input(batch_shape=(N, 1, 1), name="sin_input")
rnn = layers.LSTM(3, stateful=True, return_sequences=True)(inputs)  #
outputs = layers.TimeDistributed(layers.Dense(1))(rnn)
model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer="adam", loss="mae")

train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.cache().batch(N)
# model.fit(train)

for __ in range(1000):
    model.fit(train, epochs=1)
    model.reset_states()

y_predict = model.predict(train)
plt.plot(t, y)
plt.plot(t[1:], y_predict[:, 0, 0])


"""
x = 0.0
y_predict = np.zeros(len(y))
for i in range(len(y)):
    y_predict[i] = model.predict(np.array([x]).reshape(1, 1, 1))
    x = y_predict[i]

plt.plot(t, y)
plt.plot(t, y_predict)
"""
