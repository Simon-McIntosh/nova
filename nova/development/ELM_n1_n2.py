import numpy as np

from amigo.pyplot import plt

N = 200
diff = np.zeros(N)

for i, s in enumerate(np.linspace(0, 2 * np.pi, 1)):
    x = np.linspace(0, 2 * np.pi, N)
    diff[i] = np.max(abs(np.sin(x) - np.sin(s + x / 2)))

print(np.min(diff))

plt.plot(x, np.sin(x))
plt.plot(x, np.sin(np.pi + x / 2))
plt.plot(x, diff)


# 1.8 times amplitude
