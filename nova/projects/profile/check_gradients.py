import pygmo as pg
import numpy as np
from amigo.pyplot import plt


def fitness(x):
    rss = np.sum((np.dot(inv.wG, x.reshape(-1, 1)) - inv.wT) ** 2)
    # rss = np.sum(np.dot(inv.wG, x.reshape(-1, 1))**2)
    # rss = -2 * np.sum(np.dot(inv.wG, x.reshape(-1, 1)) * inv.wT)
    return [rss]


x = np.linalg.lstsq(inv.wG, inv.wT, rcond=None)[0][:, 0]
x *= 1.1
dx = pg.estimate_gradient_h(callable=fitness, x=x, dx=1e-8)

dx_analytic = 2 * inv.wG.T @ inv.wG @ x - 2 * inv.wG.T @ inv.wT[:, 0]

X = range(inv.nC)
ax = plt.subplots(1, 1)[1]
ax.bar(X, dx, width=0.95)
ax.bar(X, dx_analytic, width=0.7, color="C3")
ax.set_xticks(X)
ax.set_xticklabels(inv.adjust_coils, rotation=90)

wGI = np.dot(inv.wG, x.reshape(-1, 1)) ** 2
