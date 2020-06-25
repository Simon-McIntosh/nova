import tensorflow as tf
from tensorflow.keras import layers, Input, Model
import numpy as np
from amigo.pyplot import plt

tf.keras.backend.clear_session()  # reset state

batch_size = 10
ntime = 100
nfeature = 1

in_out_neurons = 1
hidden_neurons = 3
stateful = False

inp = layers.Input(batch_shape=(batch_size, ntime, nfeature), name='input')
rnn = layers.LSTM(hidden_neurons, return_sequences=True,
                  stateful=stateful, name='RNN', activation='tanh')(inp)
dens = layers.TimeDistributed(
        layers.Dense(in_out_neurons, name='dense', activation='linear'))(rnn)
model = Model(inputs=[inp], outputs=[dens])
model.compile(loss='mean_squared_error',
              sample_weight_mode='temporal',
              optimizer='rmsprop')

# construct dataset
x = np.linspace(0, 4*2*np.pi, ntime+1)
y = x[1:]
x = x[:-1].reshape(batch_size, -1, 1)
y = y.reshape(batch_size, -1, 1)

mean, std = np.mean(x), np.std(x)
x = (x - mean) / std
y = (y - mean) / std
train = tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size)

# train model
model.fit(train, epochs=200, shuffle=False, verbose=2)

plt.plot(x[0, :, 0], y[0, :, 0])
