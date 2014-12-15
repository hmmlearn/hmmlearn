from __future__ import print_function

import time
from contextlib import contextmanager

from hmmlearn.hmm import GaussianHMM


@contextmanager
def timed_step(title):
    print(title, end="... ", flush=True)
    start_time = time.clock()
    yield
    end_time = time.clock()
    print("done in {0:.2f}s".format(end_time - start_time))


def bench_gaussian_hmm(size):
    title = "Benchmarking Gaussian HMM on a sample of size {0}".format(size)
    print(title.center(64, " "))
    ghmm = GaussianHMM()
    ghmm.means_ = [[42], [24]]
    ghmm.covars_ = [[1], [1]]

    with timed_step("generating sample"):
        sample, _states = ghmm.sample(size)

    with timed_step("fitting"):
        GaussianHMM(n_components=2).fit([sample])


if __name__ == "__main__":
    bench_gaussian_hmm(2**16)
