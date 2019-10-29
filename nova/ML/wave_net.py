import numpy as np
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Conv1D, Dense, Dropout
from tensorflow.keras.metrics import mean_squared_error
from amigo.pyplot import plt


def generate(batch_size, ntime, nc=3):
    t, dt = np.linspace(0, 1, ntime, retstep=True)
    fn = np.pi / dt  # Nyquist
    fmax = fn / 2.5   # maximum frequency content
    #fmin = np.max([fmax/8, np.pi / 2])  # quater period
    fmin = np.pi / 2
    Tmin = 2 * np.pi / fmax
    Tmax = 2 * np.pi / fmin
    dT = Tmax - Tmin  # signal period bandwidth
    series = np.zeros((batch_size, ntime))
    weight = np.random.rand(nc)
    weight /= np.sum(weight)  # normalize amplitude weights
    #weight *= 10
    for i, w in enumerate(weight):  # component number
        # amplitude, shift, period
        ao, shift, To = np.random.rand(3, batch_size, 1)
        T = Tmin + To * dT  # baseline period
        f = 2 * np.pi / T
        T_shift = shift * T
        series += w * np.sin((t - T_shift) * f)
        #series += w * np.sin(t * f)
    return series[..., np.newaxis].astype(np.float32)


def last_time_step_mse(true, pred):
    return mean_squared_error(true[:, -1], pred[:, -1])

'''
series = generate(1, 20, nc=1)

mode = 'valid'
c1 = conv = np.convolve(series[0, :, 0], [1, 1], mode=mode)
c2 = conv = np.convolve(series[0, :, 0], [1, 0, 1], mode=mode)
c4 = conv = np.convolve(series[0, :, 0], [1, 0, 0, 0, 1], mode=mode)
c8 = conv = np.convolve(series[0, :, 0], [1, 0, 0, 0, 0, 0, 0, 0, 1], mode=mode)

plt.plot(series[0, :, 0])
plt.plot(c1)
#plt.plot(c2)
#plt.plot(c4)
plt.plot(c8)
'''


batch_size = 100
ntime = 200
nfilter = 100
split = 0.9  # train validate split
ntrain = int(split*batch_size)
series = generate(batch_size, ntime+nfilter, nc=3)
X_train = series[:ntrain, :ntime]
X_valid = series[ntrain:, :ntime]

Y = np.empty((batch_size, ntime, nfilter))
for i in range(1, nfilter + 1):
    Y[:, :, i - 1] = series[:, i:i+ntime, 0]
Y_train = Y[:ntrain]
Y_valid = Y[ntrain:]


model = Sequential()
model.add(InputLayer(input_shape=[None, 1]))
for rate in (1, 2, 4, 8, 16) * 1:
    model.add(Conv1D(filters=100, kernel_size=2, padding='causal',
                     dilation_rate=rate))
    # model.add(Dropout(0.05))
model.add(Conv1D(filters=nfilter, kernel_size=1))
# model.add(Dense(nfilter))
model.compile(loss='mse', optimizer='adam', metrics=[last_time_step_mse])  # , metrics=[last_time_step_mse]

history = model.fit(X_train, Y_train, epochs=50,
                    validation_split=0.1, verbose=2)

pred = model.predict(X_valid)

nfuture = nfilter
for i in range(5):
    plt.figure()
    plt.plot(np.arange(ntime), X_valid[i, :, 0], '-o', ms=12)
    plt.plot(np.arange(ntime)+1, pred[i, :, 0], 'C3--', ms=12)
    plt.plot(np.arange(nfuture) + ntime, Y_valid[i, -nfuture:, nfuture-1], '-s')
    plt.plot(np.arange(nfuture) + ntime, pred[i, -nfuture:, nfuture-1], 'C2--')
