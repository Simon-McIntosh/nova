import tensorflow as tf
from sklearn import preprocessing
import numpy as np
from scipy.optimize import minimize_scalar
from amigo.pyplot import plt
from os.path import join, isfile
import pandas as pd


# generate data-set
class Data:

    # model constants
    m = 1  # mass kg
    cp = 500  # J/kg K
    To = 283  # enviroment temperature
    rho = 8000  # kg/m3
    V = m / rho
    r = (3/4 * V / np.pi)**(1/3)  # sphere radius
    A = 4 * np.pi * r**2  # sphere surface area
    sb = 5.67e-8  # w/m2 K4
    h = 300

    def __init__(self, solve=False, **kwargs):
        self.solve = solve
        self.set_filepath()
        if 'model' in kwargs:
            self.load_model(**kwargs)

    def set_filepath(self):
        self.filepath = './Data/lumped_parameter.hdf5'

    def load_model(self, model, **kwargs):
        solve = kwargs.get('solve', self.solve)
        N = kwargs.get('N', 100)
        if solve or not isfile(self.filepath):
            self.solve_model(model, N)
        else:
            try:
                self.read_data(model)
                if self.frame.shape[0] != N:
                    self.solve_model(model, N)  # re-solve
            except KeyError:  # model key not found - re-solve
                self.solve_model(model, N)
        self.preprocess()

    def read_data(self, model):
        self.frame = pd.read_hdf(self.filepath, model)

    def solve_model(self, model, N):
        if hasattr(self, model):
            self.frame = getattr(self, model)(N)
            self.frame.to_hdf(join(self.filepath), model, mode='a')
        else:
            raise AttributeError(f'model: {model} not defined in class')

    # models
    def ss_conv_rad(self, N, noise=0.05):
        '''
        generate steady state data S=-qA-R=A(h(T-To)+sb(T**4-To**4))

        Attributes:
            N (int): sample size
        '''
        S = np.linspace(10, 10e3, N)  # W
        T = np.zeros(len(S))
        for i, s in enumerate(S):
            T[i] = minimize_scalar(self.T_conv_rad, args=(s)).x
        T_noise = T * np.random.normal(1, noise, N)  # gaussian signal-noise
        dataframe = pd.DataFrame({'S': S, 'T': T, 'T_noise': T_noise})
        return dataframe.astype('float32')

    # functions
    def T_conv_rad(self, T, *args):
        S = args[0]
        _T = (S/self.A - self.sb*(T**4 - self.To**4))/self.h + self.To
        return abs(T-_T)

    def preprocess(self, frac=0.8):
        self.stats = self.frame.describe().T
        self.train = self.norm(self.frame.sample(frac=0.8, random_state=0))
        self.test = self.norm(self.frame.drop(self.train.index))
        self.dataset = {}
        self.dataset['train'] = tf.data.Dataset.from_tensor_slices(
                (self.train['S'].values, self.train['T'].values))
        self.dataset['test'] = tf.data.Dataset.from_tensor_slices(
                (self.test['S'].values, self.test['T'].values))

    def norm(self, data):
        return (data - self.stats['mean']) / self.stats['std']

    def denorm(self, data, label):
        return data * self.stats.at[label, 'std'] + self.stats.at[label, 'mean']

# load data
data = Data()
data.load_model(model='ss_conv_rad', N=500)


# construct neural-net
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(3, activation='relu'))
# model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error',
              metrics=['mean_squared_error'])
history = model.fit(data.train['S'].values,
                    data.train['T_noise'].values, epochs=200, verbose=1)

#model.evaluate(test.values, test_target.values, verbose=0)
S_norm = data.norm(data.frame)['S'].values
plt.plot(data.frame['S'], data.frame['T'])
plt.plot(data.frame['S'],
         data.denorm(model.predict(S_norm), 'T'))



'''
if __name__ is '__main__':

    d = Data()
    d.load_model('ss_conv_rad', N=200, solve=False)

    plt.plot(d.data['S'], d.data['T_noise'])
    plt.plot(d.data['S'], d.data['T'])
'''

# pre-process data


# construct neural-net


