import numpy as np


class DistributionType:
    def __init__(self, name, function, param_names, defaults=None):
        self.name = name
        self.function = function
        self.param_names = param_names
        self.params = {}
        self.defaults = defaults or {}

    def set_params(self, **kwargs):
        self.params = kwargs

    def sample(self):
        return self.function(**self.params)
    
distributions = {
    "Uniform": DistributionType(
        "Uniform",
        lambda low=0.0, high=1.0, **kwargs: np.random.uniform(low, high),
        param_names=["low", "high"], defaults={"low": 0.0, "high": 1.0}
    ),
    "Normal": DistributionType(
        "Normal",
        lambda mean=0.0, stddev=1.0, **kwargs: np.random.normal(mean, stddev),
        param_names=["mean", "stddev"], defaults={"mean": 0.0, "stddev": 1.0}
    ),
    "Exponential": DistributionType(
        "Exponential",
        lambda scale=1.0, **kwargs: np.random.exponential(scale),
        param_names=["scale"],  defaults={"scale": 1.0}
    ),
    "Lognormal": DistributionType(
        "Lognormal",
        lambda mean=0.0, sigma=0.25, **kwargs: np.random.lognormal(mean, sigma),
        param_names=["mean", "sigma"], defaults={"mean": 0.0, "sigma": 0.25}
    ),
    "Gamma": DistributionType(
        "Gamma",
        lambda shape=2.0, scale=1.0, **kwargs: np.random.gamma(shape, scale),
        param_names=["shape", "scale"], defaults={"shape": 2.0, "scale": 1.0}
    ),
    "Poisson": DistributionType(
        "Poisson",
        lambda lam=1.0, **kwargs: np.random.poisson(lam),
        param_names=["lam"], defaults={"lam": 1.0}
    ),
    "Binomial": DistributionType(
        "Binomial",
        lambda n=1, p=0.5, **kwargs: np.random.binomial(n, p),
        param_names=["n", "p"], defaults={"n": 1, "p": 0.5}
    ),
    "Geometric": DistributionType(
        "Geometric",
        lambda p=0.5, **kwargs: np.random.geometric(p),
        param_names=["p"],  defaults={"p": 0.5}
    ),
    "Negative Binomial": DistributionType(
        "Negative Binomial",
        lambda n=1, p=0.5, **kwargs: np.random.negative_binomial(n, p),
        param_names=["n", "p"], defaults={"n": 1, "p": 0.5}
    ),
    "Beta": DistributionType(
        "Beta",
        lambda a=1.0, b=1.0, **kwargs: np.random.beta(a, b),
        param_names=["a", "b"], defaults={"a": 1.0, "b": 1.0}
    ),
}

__all__ = ['DistributionType', 'distributions']