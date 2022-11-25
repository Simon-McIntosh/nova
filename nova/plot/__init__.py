
__all__ = [
    "mpl",
    "plt",
    "sns"
]

import lazy_loader as lazy

mpl = lazy.load('matplotlib')
plt = lazy.load('matplotlib.pyplot')
sns = lazy.load('seaborn')
