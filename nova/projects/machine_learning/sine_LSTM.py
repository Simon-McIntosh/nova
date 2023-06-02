from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(10, activation="tanh", input_shape=(), batch_side=(), statefull=False))
model.add(Dense(1, activation="tanh"))
model.compile(optimizer="adam", loss="mse")
