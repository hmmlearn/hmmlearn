from __future__ import division
import numpy as np
from hmmlearn import hmm

states = ["Rainy", "Sunny"]
n_states = len(states)

observations = ["walk", "shop", "clean"]
n_observations = len(observations)

start_probability = np.array([0.6, 0.4])

transition_probability = np.array([
    [0.7, 0.3],
    [0.4, 0.6]
])

emission_probability = np.array([
    [0.1, 0.4, 0.5],
    [0.6, 0.3, 0.1]
])

model = hmm.MultinomialHMM(n_components=n_states)
model.startprob = start_probability
model.transmat = transition_probability
model.emissionprob = emission_probability

# predict a sequence of hidden states based on visible states
bob_says = np.array([[0, 2, 1, 1, 2, 0]]).T
model = model.fit(bob_says)
logprob, alice_hears = model.decode(bob_says, algorithm="viterbi")
print ("Bob says:", ", ".join(map(lambda x: observations[x[0]], bob_says)))
print ("Alice hears:", ", ".join(map(lambda x: states[x], alice_hears)))
