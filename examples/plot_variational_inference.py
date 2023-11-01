"""
Learning an HMM using VI and EM over a set of Gaussian sequences
----------------------------------------------------------------

We train models with a variety of number of states
(N) for each algorithm, and then examine which model is "best", by printing
the log-likelihood or variational lower bound for each N.  We will see that
an HMM trained using VI will prefer the correct number of states, while an
HMM learning with EM will prefer as many states as possible.  Note, for
models trained with EM, some other criteria such as AIC/BIC, or held out
test data, could be used to select the correct number of hidden states.
"""

import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import scipy.stats

from sklearn.utils import check_random_state

from hmmlearn import hmm, vhmm
import matplotlib


def gaussian_hinton_diagram(startprob, transmat, means,
                            variances, vmin=0, vmax=1, infer_hidden=True):
    """
    Show the initial state probabilities, the transition probabilities
    as heatmaps, and draw the emission distributions.
    """

    num_states = transmat.shape[0]

    f = plt.figure(figsize=(3*(num_states), 2*num_states))
    grid = gs.GridSpec(3, 3)

    ax = f.add_subplot(grid[0, 0])
    ax.imshow(startprob[None, :], vmin=vmin, vmax=vmax)
    ax.set_title("Initial Probabilities", size=14)

    ax = f.add_subplot(grid[1:, 0])
    ax.imshow(transmat, vmin=vmin, vmax=vmax)
    ax.set_title("Transition Probabilities", size=14)

    ax = f.add_subplot(grid[1:, 1:])
    for i in range(num_states):
        keep = True
        if infer_hidden:
            if np.all(np.abs(transmat[i] - transmat[i][0]) < 1e-4):
                keep = False
        if keep:
            s_min = means[i] - 10 * variances[i]
            s_max = means[i] + 10 * variances[i]
            xx = np.arange(s_min, s_max, (s_max - s_min) / 1000)
            norm = scipy.stats.norm(means[i], np.sqrt(variances[i]))
            yy = norm.pdf(xx)
            keep = yy > .01
            ax.plot(xx[keep], yy[keep], label="State: {}".format(i))
    ax.set_title("Emissions Probabilities", size=14)
    ax.legend(loc="best")
    f.tight_layout()
    return f

np.set_printoptions(formatter={'float_kind': "{:.3f}".format})
rs = check_random_state(2022)
sample_length = 500
num_samples = 1
# With random initialization, it takes a few tries to find the
# best solution
num_inits = 5
num_states = np.arange(1, 7)
verbose = False


# Prepare parameters for a 4-components HMM
# And Sample several sequences from this model
model = hmm.GaussianHMM(4, init_params="")
model.n_features = 4
# Initial population probability
model.startprob_ = np.array([0.25, 0.25, 0.25, 0.25])
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
model.transmat_ = np.array([[0.2, 0.2, 0.3, 0.3],
                            [0.3, 0.2, 0.2, 0.3],
                            [0.2, 0.3, 0.3, 0.2],
                            [0.3, 0.3, 0.2, 0.2]])
# The means and covariance of each component
model.means_ = np.array([[-1.5],
                         [0],
                         [1.5],
                         [3.]])
model.covars_ = np.array([[0.25],
                          [0.25],
                          [0.25],
                          [0.25]])**2

# Generate training data
sequences = []
lengths = []

for i in range(num_samples):
    sequences.extend(model.sample(sample_length, random_state=rs)[0])
    lengths.append(sample_length)
sequences = np.asarray(sequences)

# Train a suite of models, and keep track of the best model for each
# number of states, and algorithm
best_scores = collections.defaultdict(dict)
best_models = collections.defaultdict(dict)
for n in num_states:
    for i in range(num_inits):
        vi = vhmm.VariationalGaussianHMM(n,
                                         n_iter=1000,
                                         covariance_type="full",
                                         implementation="scaling",
                                         tol=1e-6,
                                         random_state=rs,
                                         verbose=verbose)
        vi.fit(sequences, lengths)
        lb = vi.monitor_.history[-1]
        print(f"Training VI({n}) Variational Lower Bound={lb} "
              f"Iterations={len(vi.monitor_.history)} ")
        if best_models["VI"].get(n) is None or best_scores["VI"][n] < lb:
            best_models["VI"][n] = vi
            best_scores["VI"][n] = lb

        em = hmm.GaussianHMM(n,
                             n_iter=1000,
                             covariance_type="full",
                             implementation="scaling",
                             tol=1e-6,
                             random_state=rs,
                             verbose=verbose)
        em.fit(sequences, lengths)
        ll = em.monitor_.history[-1]
        print(f"Training EM({n}) Final Log Likelihood={ll} "
              f"Iterations={len(vi.monitor_.history)} ")
        if best_models["EM"].get(n) is None or best_scores["EM"][n] < ll:
            best_models["EM"][n] = em
            best_scores["EM"][n] = ll

# Display the model likelihood/variational lower bound for each N
# and show the best learned model
for algo, scores in best_scores.items():
    best = max(scores.values())
    best_n, best_score = max(scores.items(), key=lambda x: x[1])
    for n, score in scores.items():
        flag = "* <- Best Model" if score == best_score else ""
        print(f"{algo}({n}): {score:.4f}{flag}")

    print(f"Best Model {algo}")
    best_model = best_models[algo][best_n]
    print(best_model.transmat_)
    print(best_model.means_)
    print(best_model.covars_)

# Also inpsect the VI model with 6 states, to see how it has sparse structure
vi_model = best_models["VI"][6]
em_model = best_models["EM"][6]
print("VI solution for 6 states: Notice sparsity among states 1 and 4")
print(vi_model.transmat_)
print(vi_model.means_)
print(vi_model.covars_)
print("EM solution for 6 states")
print(em_model.transmat_)
print(em_model.means_)
print(em_model.covars_)

f = gaussian_hinton_diagram(
    vi_model.startprob_,
    vi_model.transmat_,
    vi_model.means_.ravel(),
    vi_model.covars_.ravel(),
)
f.suptitle("Variational Inference Solution", size=16)
f = gaussian_hinton_diagram(
    em_model.startprob_,
    em_model.transmat_,
    em_model.means_.ravel(),
    em_model.covars_.ravel(),
)
f.suptitle("Expectation-Maximization Solution", size=16)

plt.show()
