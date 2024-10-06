#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cfenv>
#include <limits>

namespace py = pybind11;
using ssize_t = Py_ssize_t;

double logaddexp(double a, double b)
{
  return
    a == -std::numeric_limits<double>::infinity() ? b
    : b == -std::numeric_limits<double>::infinity() ? a
    : std::max(a, b) + std::log1p(std::exp(-std::abs(b - a)));
}

double logsumexp(double const* v, ssize_t n)
{
  auto max = *std::max_element(v, v + n);
  if (std::isinf(max)) {
    return max;
  }
  auto acc = 0.;
  for (auto i = 0; i < n; ++i) {
    acc += std::exp(v[i] - max);
  }
  return std::log(acc) + max;
}

py::array_t<double> log(
  py::array_t<double, py::array::c_style | py::array::forcecast> x_)
{
  auto n = x_.size();
  auto ptr = x_.data();
  auto log = py::array_t<double>{{n}};
  auto log_ptr = log.mutable_data();
  for (auto i = 0; i < n; ++i) {
    *(log_ptr++) = std::log(*(ptr++));
  }
  if (std::fetestexcept(FE_DIVBYZERO)) {
    std::feclearexcept(FE_DIVBYZERO);  // log(0) = -inf, ignore exception.
  }
  return log.reshape(std::vector<ssize_t>(x_.shape(), x_.shape() + x_.ndim()));
}

std::tuple<double, py::array_t<double>, py::array_t<double>> forward_scaling(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_)
{
  auto min_sum = 1e-300;

  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  auto scaling_ = py::array_t<double>{{ns}};
  auto scaling = scaling_.mutable_unchecked<1>();
  auto log_prob = 0.;
  {
    py::gil_scoped_release nogil;
    std::fill_n(fwd.mutable_data(0, 0), fwd.size(), 0);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) = startprob(i) * frameprob(0, i);
    }
    auto sum = std::accumulate(&fwd(0, 0), &fwd(0, nc), 0.);
    if (sum < min_sum) {
      throw std::range_error{"forward pass failed with underflow; "
                             "consider using implementation='log' instead"};
    }
    auto scale = scaling(0) = 1. / sum;
    log_prob -= std::log(scale);
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) *= scale;
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        for (auto i = 0; i < nc; ++i) {
          fwd(t, j) += fwd(t - 1, i) * transmat(i, j);
        }
        fwd(t, j) *= frameprob(t, j);
      }
      auto sum = std::accumulate(&fwd(t, 0), &fwd(t, nc), 0.);
      if (sum < min_sum) {
        throw std::range_error{"forward pass failed with underflow; "
                               "consider using implementation='log' instead"};
      }
      auto scale = scaling(t) = 1. / sum;
      log_prob -= std::log(scale);
      for (auto j = 0; j < nc; ++j) {
        fwd(t, j) *= scale;
      }
    }
  }
  return {log_prob, fwdlattice_, scaling_};
}

std::tuple<double, py::array_t<double>> forward_log(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto buf = std::vector<double>(nc);
  auto fwdlattice_ = py::array_t<double>{{ns, nc}};
  auto fwd = fwdlattice_.mutable_unchecked<2>();
  {
    py::gil_scoped_release nogil;
    for (auto i = 0; i < nc; ++i) {
      fwd(0, i) = log_startprob(i) + log_frameprob(0, i);
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto j = 0; j < nc; ++j) {
        for (auto i = 0; i < nc; ++i) {
          buf[i] = fwd(t - 1, i) + log_transmat(i, j);
        }
        fwd(t, j) = logsumexp(buf.data(), nc) + log_frameprob(t, j);
      }
    }
  }
  auto log_prob = logsumexp(&fwd(ns - 1, 0), nc);
  return {log_prob, fwdlattice_};
}

py::array_t<double> backward_scaling(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> frameprob_,
  py::array_t<double> scaling_)
{
  auto startprob = startprob_.unchecked<1>();
  auto transmat = transmat_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto scaling = scaling_.unchecked<1>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (startprob.shape(0) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || scaling.shape(0) != ns) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();
  py::gil_scoped_release nogil;
  std::fill_n(bwd.mutable_data(0, 0), bwd.size(), 0);
  for (auto i = 0; i < nc; ++i) {
    bwd(ns - 1, i) = scaling(ns - 1);
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        bwd(t, i) += transmat(i, j) * frameprob(t + 1, j) * bwd(t + 1, j);
      }
      bwd(t, i) *= scaling(t);
    }
  }
  return bwdlattice_;
}

py::array_t<double> backward_log(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto buf = std::vector<double>(nc);
  auto bwdlattice_ = py::array_t<double>{{ns, nc}};
  auto bwd = bwdlattice_.mutable_unchecked<2>();
  py::gil_scoped_release nogil;
  for (auto i = 0; i < nc; ++i) {
    bwd(ns - 1, i) = 0;
  }
  for (auto t = ns - 2; t >= 0; --t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        buf[j] = log_transmat(i, j) + log_frameprob(t + 1, j) + bwd(t + 1, j);
      }
      bwd(t, i) = logsumexp(buf.data(), nc);
    }
  }
  return bwdlattice_;
}

py::array_t<double> compute_scaling_xi_sum(
  py::array_t<double> fwdlattice_,
  py::array_t<double> transmat_,
  py::array_t<double> bwdlattice_,
  py::array_t<double> frameprob_)
{
  auto fwd = fwdlattice_.unchecked<2>();
  auto transmat = transmat_.unchecked<2>();
  auto bwd = bwdlattice_.unchecked<2>();
  auto frameprob = frameprob_.unchecked<2>();
  auto ns = frameprob.shape(0), nc = frameprob.shape(1);
  if (fwd.shape(0) != ns || fwd.shape(1) != nc
      || transmat.shape(0) != nc || transmat.shape(1) != nc
      || bwd.shape(0) != ns || bwd.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto xi_sum_ = py::array_t<double>{{nc, nc}};
  auto xi_sum = xi_sum_.mutable_unchecked<2>();
  std::fill_n(xi_sum.mutable_data(0, 0), xi_sum.size(), 0);
  py::gil_scoped_release nogil;
  for (auto t = 0; t < ns - 1; ++t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        xi_sum(i, j) += fwd(t, i)
                        * transmat(i, j)
                        * frameprob(t + 1, j)
                        * bwd(t + 1, j);
      }
    }
  }
  return xi_sum_;
}

py::array_t<double> compute_log_xi_sum(
  py::array_t<double> fwdlattice_,
  py::array_t<double> transmat_,
  py::array_t<double> bwdlattice_,
  py::array_t<double> log_frameprob_)
{
  auto fwd = fwdlattice_.unchecked<2>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto bwd = bwdlattice_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (fwd.shape(0) != ns || fwd.shape(1) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc
      || bwd.shape(0) != ns || bwd.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto log_prob = logsumexp(&fwd(ns - 1, 0), nc);
  auto log_xi_sum_ = py::array_t<double>{{nc, nc}};
  auto log_xi_sum = log_xi_sum_.mutable_unchecked<2>();
  std::fill_n(log_xi_sum.mutable_data(0, 0), log_xi_sum.size(),
              -std::numeric_limits<double>::infinity());
  py::gil_scoped_release nogil;
  for (auto t = 0; t < ns - 1; ++t) {
    for (auto i = 0; i < nc; ++i) {
      for (auto j = 0; j < nc; ++j) {
        auto log_xi = fwd(t, i)
                      + log_transmat(i, j)
                      + log_frameprob(t + 1, j)
                      + bwd(t + 1, j)
                      - log_prob;
        log_xi_sum(i, j) = logaddexp(log_xi_sum(i, j), log_xi);
      }
    }
  }
  return log_xi_sum_;
}

std::tuple<double, py::array_t<ssize_t>> viterbi(
  py::array_t<double> startprob_,
  py::array_t<double> transmat_,
  py::array_t<double> log_frameprob_)
{
  auto log_startprob_ = log(startprob_);
  auto log_startprob = log_startprob_.unchecked<1>();
  auto log_transmat_ = log(transmat_);
  auto log_transmat = log_transmat_.unchecked<2>();
  auto log_frameprob = log_frameprob_.unchecked<2>();
  auto ns = log_frameprob.shape(0), nc = log_frameprob.shape(1);
  if (log_startprob.shape(0) != nc
      || log_transmat.shape(0) != nc || log_transmat.shape(1) != nc) {
    throw std::invalid_argument{"shape mismatch"};
  }
  auto state_sequence_ = py::array_t<ssize_t>{{ns}};
  auto viterbi_lattice_ = py::array_t<double>{{ns, nc}};
  auto state_sequence = state_sequence_.mutable_unchecked<1>();
  auto viterbi_lattice = viterbi_lattice_.mutable_unchecked<2>();
  {
    py::gil_scoped_release nogil;
    for (auto i = 0; i < nc; ++i) {
      viterbi_lattice(0, i) = log_startprob(i) + log_frameprob(0, i);
    }
    for (auto t = 1; t < ns; ++t) {
      for (auto i = 0; i < nc; ++i) {
        auto max = -std::numeric_limits<double>::infinity();
        for (auto j = 0; j < nc; ++j) {
          max = std::max(max, viterbi_lattice(t - 1, j) + log_transmat(j, i));
        }
        viterbi_lattice(t, i) = max + log_frameprob(t, i);
      }
    }
    auto row = &viterbi_lattice(ns - 1, 0);
    auto prev = state_sequence(ns - 1) = std::max_element(row, row + nc) - row;
    for (auto t = ns - 2; t >= 0; --t) {
      auto max = std::make_pair(-std::numeric_limits<double>::infinity(), 0);
      for (auto i = 0; i < nc; ++i) {
        max = std::max(max, {viterbi_lattice(t, i) + log_transmat(i, prev), i});
      }
      state_sequence(t) = prev = max.second;
    }
  }
  return {viterbi_lattice(ns - 1, state_sequence(ns - 1)), state_sequence_};
}

PYBIND11_MODULE(_hmmc, m) {
  m
    .def("forward_scaling", forward_scaling)
    .def("forward_log", forward_log)
    .def("backward_scaling", backward_scaling)
    .def("backward_log", backward_log)
    .def("compute_scaling_xi_sum", compute_scaling_xi_sum)
    .def("compute_log_xi_sum", compute_log_xi_sum)
    .def("viterbi", viterbi)
    ;
}
