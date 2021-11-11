"""
A script for testing / benchmarking HMM Implementations
"""

import argparse
import collections
import logging
import time

import hmmlearn.hmm

import numpy as np

import sklearn.base

LOG = logging.getLogger(__file__)


class Benchmark:
    def __init__(self, repeat, n_iter, verbose):
        self.repeat = repeat
        self.n_iter = n_iter
        self.verbose = verbose

    def benchmark(self, sequences, lengths, model, tag):
        elapsed = []
        for i in range(self.repeat):
            start = time.time()
            cloned = sklearn.base.clone(model)
            cloned.fit(sequences, lengths)
            end = time.time()
            elapsed.append(end-start)
            self.log_one_run(start, end, cloned, tag)
        return np.asarray(elapsed)

    def generate_training_sequences(self):
        pass

    def new_model(self, implementation):
        pass

    def run(self, results_file):
        runtimes = collections.defaultdict(dict)

        sequences, lengths = self.generate_training_sequences()

        for implementation in ["scaling", "log"]:
            model = self.new_model(implementation)
            LOG.info(f"{model.__class__.__name__}: testing {implementation}")
            key = f"{model.__class__.__name__}|EM|hmmlearn-{implementation}"
            elapsed = self.benchmark(sequences, lengths, model, key)
            runtimes[key]["mean"] = elapsed.mean()
            runtimes[key]["std"] = elapsed.std()

        with open(results_file, mode="w") as fd:
            fd.write("configuration,mean,std,n_iterations,repeat\n")
            for key, value in runtimes.items():
                fd.write(f"{key},{value['mean']},{value['std']},{self.n_iter},{self.repeat}\n")

    def log_one_run(self, start, end, model, tag):
        LOG.info(f"Training Took {end-start} seconds {tag}")
        LOG.info(f"startprob={model.startprob_}")
        LOG.info(f"transmat={model.transmat_}")


class GaussianBenchmark(Benchmark):

    def new_model(self, implementation):
        return hmmlearn.hmm.GaussianHMM(
            n_components=4,
            n_iter=self.n_iter,
            covariance_type="full",
            implementation=implementation,
            verbose=self.verbose
        )

    def generate_training_sequences(self):
        sampler = hmmlearn.hmm.GaussianHMM(
            n_components=4,
            covariance_type="full",
            init_params="",
            verbose=self.verbose
        )

        sampler.startprob_ = np.asarray([0, 0, 0, 1])
        sampler.transmat_ = np.asarray([
            [.2, .2, .3, .3],
            [.3, .2, .2, .3],
            [.2, .3, .3, .2],
            [.3, .3, .2, .2],
        ])
        sampler.means_ = np.asarray([
            -1.5,
            0,
            1.5,
            3
        ]).reshape(4, 1)
        sampler.covars_ = np.asarray([
            .5,
            .5,
            .5,
            .5
        ]).reshape(4, 1, 1,)

        sequences, states = sampler.sample(50000)
        lengths = [len(sequences)]
        return sequences, lengths

    def log_one_run(self, start, end, model, tag):
        super().log_one_run(start, end, model, tag)
        LOG.info(f"means={model.means_}")
        LOG.info(f"covars={model.covars_}")


class MultinomialBenchmark(Benchmark):

    def new_model(self, implementation):
        return hmmlearn.hmm.MultinomialHMM(
                n_components=3,
                n_iter=self.n_iter,
                verbose=self.verbose,
                implementation=implementation
            )

    def generate_training_sequences(self):

        sampler = hmmlearn.hmm.MultinomialHMM(n_components=3)
        sampler.startprob_ = np.array([0.6, 0.3, 0.1])
        sampler.transmat_ = np.array([[0.6, 0.2, 0.2],
                             [0.3, 0.5, 0.2],
                             [0.4, 0.3, 0.3]])

        sampler.emissionprob_ = np.array([
            [.1, .5, .1, .3],
            [.1, .2, .4, .3],
            [0, .5, .5, .0],
        ])

        sequences, states = sampler.sample(50000)
        lengths = [len(sequences)]
        return sequences, lengths

    def log_one_run(self, start, end, model, tag):
        super().log_one_run(start, end, model, tag)
        LOG.info(f"emissions={model.emissionprob_}")


class MultivariateGaussianBenchmark(GaussianBenchmark):
    def generate_training_sequences(self):
        sampler = hmmlearn.hmm.GaussianHMM(
            n_components=4,
            covariance_type="full",
            init_params=""
        )

        sampler.startprob_ = np.asarray([0, 0, 0, 1])
        sampler.transmat_ = np.asarray([
            [.2, .2, .3, .3],
            [.3, .2, .2, .3],
            [.2, .3, .3, .2],
            [.3, .3, .2, .2],
        ])
        sampler.means_ = np.asarray([
            [-1.5, 0],
            [0,  0],
            [1.5, 0],
            [3, 0]
        ])
        sampler.covars_ = np.asarray([
            [[.5, 0],
             [0, .5]],
            [[.5, 0],
             [0, 0.5]],

            [[.5, 0],
             [0, .5]],
            [[0.5, 0],
             [0, 0.5]],
        ])

        observed, hidden = sampler.sample(50000)
        lengths = [len(observed)]
        return observed, lengths


class GMMBenchmark(GaussianBenchmark):
    def generate_training_sequences(self):
        sampler = hmmlearn.hmm.GMMHMM(
            n_components=4,
            n_mix=3,
            covariance_type="full",
            init_params=""
        )

        sampler.startprob_ = [.25, .25, .25, .25]
        sampler.transmat_ = [
            [.1, .3, .3, .3],
            [.3, .1, .3, .3],
            [.3, .3, .1, .3],
            [.3, .3, .3, .1],
        ]
        sampler.weights_ = [
            [.2, .2, .6],
            [.6, .2, .2],
            [.2, .6, .2],
            [.1, .1, .8],
        ]
        sampler.means_ = np.asarray([
            [[-10], [-12], [-9]],
            [[-5], [-4], [-3]],
            [[-1.5], [0], [1.5]],
            [[5], [7], [9]],
        ])

        sampler.covars_ = np.asarray([
            [[[.125]], [[.125]], [[.125]]],
            [[[.125]], [[.125]], [[.125]]],
            [[[.125]], [[.125]], [[.125]]],
            [[[.125]], [[.125]], [[.125]]],
        ])

        n_sequences = 10
        length = 5_000
        sequences = []
        for i in range(n_sequences):
            sequences.append(sampler.sample(5000)[0])
        return np.concatenate(sequences), [length] * n_sequences

    def new_model(self, implementation):
        return hmmlearn.hmm.GMMHMM(
            n_components=4,
            n_mix=3,
            n_iter=self.n_iter,
            covariance_type="full",
            verbose=self.verbose,
            implementation=implementation
        )

    def log_one_run(self, start, end, model, tag):
        super().log_one_run(start, end, model, tag)
        LOG.info(f"weights_={model.weights_}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--categorical", action="store_true")
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--multivariate-gaussian", action="store_true")
    parser.add_argument("--gaussian-mixture", action="store_true")
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-iter", type=int, default=100)

    args = parser.parse_args()
    if args.all:
        args.categorical = True
        args.gaussian = True
        args.multivariate_gaussian = True
        args.gaussian_mixture = True

    if args.categorical:
        bench = MultinomialBenchmark(
            repeat=args.repeat,
            n_iter=args.n_iter,
            verbose=args.verbose,
        )
        bench.run("categorical.benchmark.csv")
    if args.gaussian:
        bench = GaussianBenchmark(
            repeat=args.repeat,
            n_iter=args.n_iter,
            verbose=args.verbose,
        )
        bench.run("gaussian.benchmark.csv")
    if args.multivariate_gaussian:
        bench = MultivariateGaussianBenchmark(
            repeat=args.repeat,
            n_iter=args.n_iter,
            verbose=args.verbose,
        )
        bench.run("multivariate_gaussian.benchmark.csv")
    if args.gaussian_mixture:
        bench = GMMBenchmark(
            repeat=args.repeat,
            n_iter=args.n_iter,
            verbose=args.verbose,
        )
        bench.run("gmm.benchmark.csv")


if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )
    main()
