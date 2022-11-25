
import numpy as np
import scipy

from nova.plot import plt

coef = np.array([ 0.        +0.00000000e+00j,  0.25730663+7.12056556e-01j,
       -0.46931815-5.57185997e-01j, -0.37108191-2.14703542e-01j,
       -0.28057591-4.99450146e-02j, -0.20239453+3.52731887e-02j,
       -0.12738651+7.14616327e-02j, -0.05977957+7.17774492e-02j,
       -0.01637061+4.11230576e-02j, -0.00137264-1.68100403e-19j])

magnitude, phase = abs(coef), np.angle(coef)
for k in np.arange(1, 10):
    magnitude[k] /= k * np.pi/9
    phase[k] += k*np.pi/36
coef = magnitude * np.exp(1j * phase)

def transform(gain, poles, k):
    return gain / (k*(k + poles[0]))

def error(x):
    gain = x[0]
    poles = x[1:]
    k = np.arange(2, 10)*1j
    return np.sum(abs(transform(gain, poles, k)-coef[2:])**2)


sol = scipy.optimize.minimize(error, [1e2, 5, 5], method='Powell')
print(sol)

gain, pole = sol.x[:2]

plt.plot(coef.real, coef.imag, 'o', label='ANSYS')
for i in range(10):
    plt.text(coef.real[i], coef.imag[i], i)

k = np.linspace(1, 10)*1j
f = transform(sol.x[0], sol.x[1:], k)
plt.plot(f.real, f.imag, '-', color='gray',
         label=rf'$TF=\frac{{{gain:1.0f}}}{{s(s+{pole:1.1f})^4}}$')

k = np.arange(1, 10)*1j
f = transform(sol.x[0], sol.x[1:], k)
plt.plot(f.real, f.imag, 'o', ms=5, color='C3')

phi = np.linspace(0, 2*np.pi)
plt.plot(np.cos(phi), np.sin(phi), '--', color='gray')
plt.plot([-1, 1], [0, 0], '-.', color='gray')
plt.plot([0, 0], [-1, 1], '-.', color='gray')
plt.text(1.05, 0, 'Re', va='center', color='gray')
plt.text(0, 1.05, 'Im', ha='center', color='gray')

plt.axis('equal')
plt.axis('off')
plt.legend()
