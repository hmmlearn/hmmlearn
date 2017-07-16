# Bug-specific tests

from hmmlearn import hmm
import numpy as np

def test_214():
    model = hmm.GaussianHMM(
        n_components=3, params='st', init_params='',
    )

    model.transmat_ = np.array(
        [[0.8, 0.1, 0.1],
        [0.1, 0.9, 0.0],
        [0.0, 0.0, 1.0]]
    )
    model.means_ = np.array([1.0, 0.0, 0.0]).reshape(-1, 1)
    model.covars_ = np.array([0.001, 0.001, 0.001]).reshape(-1, 1)
    model.startprob_ = np.array([0.8, 0.1, 0.1])

    samples = np.array([
        0.97463307, 0.98580524, 0.96502726, 0.94767963, 0.96783919, 0.02015515,
        0.97280737, 1.05605478, 1.01784615, -0.01791463, 0.02308386, 0.0117951,
        0.01688058, -0.00290845, 0.03611139, -0.03572094, -0.02688102, 0.0303838,
        0.02761991, -0.00352225
    ]).reshape(-1, 1)
    states = [0, 0, 0, 0, 0, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]

    prediction = model.predict(samples)
    np.testing.assert_equal(states, prediction)
