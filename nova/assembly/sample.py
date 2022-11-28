
import numpy as np
import pymc3 as pm

import matplotlib.pyplot as plt

with pm.Model() as model:

    dr_coil = pm.Uniform('dr', -5, 5, shape=5)
    noise = pm.Gamma('noise', alpha=2, beta=1)

    np.fft.rfft(dr_coil, axis=-1)

    data = pm.Deterministic('data', dr_coil)
    '''
    y_observed = pm.Normal(
        "y_observed",
        mu=X @ weights,
        sigma=noise,
        observed=y,
    )
    '''

    #prior = pm.sample_prior_predictive()
    #posterior = pm.sample()
    #posterior_pred = pm.sample_posterior_predictive(posterior)

#pm.model_to_graphviz(linear_model)

with model:
    trace = pm.sample()

plt.hist(trace['data'], bins=31, rwidth=0.85)
