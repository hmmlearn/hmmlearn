"""
Sampling from and decoding a Poisson HMM
----------------------------------------

This script shows how to sample points from a Poisson Hidden Markov Model
(HMM) with three states:

The plots show the sequence of observations generated with the transitions
between them. We can see that, as specified by our transition matrix,
there are no transition between component 1 and 3.

Then, we decode our model to recover the input parameters.
"""

import numpy as np
import matplotlib.pyplot as plt

from hmmlearn import hmm

# Prepare parameters for a 3-components HMM
# Initial population probability
startprob = np.array([0.6, 0.3, 0.1])
# The transition matrix
transmat = np.array([[0.1, 0.2, 0.7],
                     [0.3, 0.5, 0.2],
                     [0.5, 0.5, 0.0]])
# The means of each component
lambdas = np.array([[17.4, 22.1],
                    [35.3, 60.8],
                    [50.1, 12.9]])

# Build an HMM instance and set parameters
gen_model = hmm.PoissonHMM(n_components=3, random_state=99)

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
gen_model.startprob_ = startprob
gen_model.transmat_ = transmat
gen_model.lambdas_ = lambdas

# Generate samples
X, Z = gen_model.sample(500)

# Plot the sampled data
fig, ax = plt.subplots()
ax.plot(X[:, 0], X[:, 1], ".-", label="observations", ms=6,
        mfc="orange", alpha=0.7, linewidth=0.1)

# Indicate the component numbers
for i, m in enumerate(lambdas):
    ax.text(m[0], m[1], 'Component %i' % (i + 1),
            size=17, horizontalalignment='center',
            bbox=dict(alpha=.7, facecolor='w'))
ax.legend(loc='best')
fig.show()

# %%
# Now, let's ensure we can recover our parameters.

scores = list()
models = list()
for idx in range(50):
    # define our hidden Markov model
    model = hmm.PoissonHMM(n_components=3, random_state=idx)
    model.fit(X[:X.shape[0] // 2])  # 50/50 train/validate
    models.append(model)
    scores.append(model.score(X[X.shape[0] // 2:]))
    print(f'Converged: {model.monitor_.converged}'
          f'\tScore: {scores[-1]}')

# get the best model
model = models[np.argmax(scores)]
print(f'The best model had a score of {max(scores)}')

# use the Viterbi algorithm to predict the most likely sequence of states
# given the model
states = model.predict(X)

# %%
# Let's plot our states compared to those generated and our transition matrix
# to get a sense of our model. We can see that the recovered states follow
# the same path as the generated states, just with the identities of the
# states transposed (i.e. instead of following a square as in the first
# figure, the nodes are switch around but this does not change the basic
# pattern). The same is true for the transition matrix.

# plot model states over time
fig, ax = plt.subplots()
ax.plot(Z, states)
ax.set_title('States compared to generated')
ax.set_xlabel('Generated State')
ax.set_ylabel('Recovered State')
fig.show()

# plot the transition matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 5))
ax1.imshow(gen_model.transmat_, aspect='auto', cmap='spring')
ax1.set_title('Generated Transition Matrix')
ax2.imshow(model.transmat_, aspect='auto', cmap='spring')
ax2.set_title('Recovered Transition Matrix')
for ax in (ax1, ax2):
    ax.set_xlabel('State To')
    ax.set_ylabel('State From')

fig.tight_layout()
fig.show()
