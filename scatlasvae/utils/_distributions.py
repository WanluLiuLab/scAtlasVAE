import torch
from torch.distributions import constraints, NegativeBinomial, Poisson, Beta, Normal
from torch.distributions.utils import (
    broadcast_all,
    lazy_property,
    logits_to_probs,
    probs_to_logits,
)
from torch.nn.functional import softplus
from torch.distributions.distribution import Distribution
from scipy.stats import f
import numpy as np

def multivariate_gaussian_p_value(X, Y):
    # Step 1: Compute the mean vectors for X and Y
    mean_X = np.mean(X, axis=0)
    mean_Y = np.mean(Y, axis=0)

    # Step 2: Compute the covariance matrices for X and Y (use unbiased estimator)
    cov_X = np.cov(X, rowvar=False, bias=False)
    cov_Y = np.cov(Y, rowvar=False, bias=False)

    # Step 3: Compute the sample sizes of X and Y
    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Step 4: Compute the Hotelling's T-square test statistic
    t_square = (n_X * n_Y) / (n_X + n_Y) * np.dot(np.dot((mean_X - mean_Y), np.linalg.inv((cov_X / n_X) + (cov_Y / n_Y))), (mean_X - mean_Y))

    # Step 5: Calculate the degrees of freedom for the F-distribution
    df1 = X.shape[1]  # Number of dimensions
    df2 = n_X + n_Y - df1 - 1

    # Step 6: Calculate the p-value using the cumulative distribution function (CDF) of the F-distribution
    p_value = 1.0 - f.cdf(t_square, df1, df2)

    return p_value


# Pyro Distributions.utils
def broadcast_shape(*shapes, **kwargs):
    """
    Similar to ``np.broadcast()`` but for shapes.
    Equivalent to ``np.broadcast(*map(np.empty, shapes)).shape``.
    :param tuple shapes: shapes of tensors.
    :param bool strict: whether to use extend-but-not-resize broadcasting.
    :returns: broadcasted shape
    :rtype: tuple
    :raises: ValueError
    """
    strict = kwargs.pop("strict", False)
    reversed_shape = []
    for shape in shapes:
        for i, size in enumerate(reversed(shape)):
            if i >= len(reversed_shape):
                reversed_shape.append(size)
            elif reversed_shape[i] == 1 and not strict:
                reversed_shape[i] = size
            elif reversed_shape[i] != size and (size != 1 or strict):
                raise ValueError(
                    "shape mismatch: objects cannot be broadcast to a single shape: {}".format(
                        " vs ".join(map(str, shapes))
                    )
                )
    return tuple(reversed(reversed_shape))

# Pyro Distributions.zero_inflated
class ZeroInflatedDistribution(Distribution):
    """
    Generic Zero Inflated distribution.
    This can be used directly or can be used as a base class as e.g. for
    :class:`ZeroInflatedPoisson` and :class:`ZeroInflatedNegativeBinomial`.
    :param TorchDistribution base_dist: the base distribution.
    :param torch.Tensor gate: probability of extra zeros given via a Bernoulli distribution.
    :param torch.Tensor gate_logits: logits of extra zeros given via a Bernoulli distribution.
    """

    arg_constraints = {
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(self, base_dist, *, gate=None, gate_logits=None, validate_args=None):
        if (gate is None) == (gate_logits is None):
            raise ValueError(
                "Either `gate` or `gate_logits` must be specified, but not both."
            )
        if gate is not None:
            batch_shape = broadcast_shape(gate.shape, base_dist.batch_shape)
            self.gate = gate.expand(batch_shape)
        else:
            batch_shape = broadcast_shape(gate_logits.shape, base_dist.batch_shape)
            self.gate_logits = gate_logits.expand(batch_shape)
        if base_dist.event_shape:
            raise ValueError(
                "ZeroInflatedDistribution expected empty "
                "base_dist.event_shape but got {}".format(base_dist.event_shape)
            )

        self.base_dist = base_dist.expand(batch_shape)
        event_shape = torch.Size()

        super().__init__(batch_shape, event_shape, validate_args)

    @constraints.dependent_property
    def support(self):
        return self.base_dist.support

    @lazy_property
    def gate(self):
        return logits_to_probs(self.gate_logits)

    @lazy_property
    def gate_logits(self):
        return probs_to_logits(self.gate)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)

        if "gate" in self.__dict__:
            gate, value = broadcast_all(self.gate, value)
            log_prob = (-gate).log1p() + self.base_dist.log_prob(value)
            log_prob = torch.where(value == 0, (gate + log_prob.exp()).log(), log_prob)
        else:
            gate_logits, value = broadcast_all(self.gate_logits, value)
            base_log_prob = self.base_dist.log_prob(value)
            log_prob_minus_log_gate = -gate_logits + base_log_prob
            log_gate = -softplus(-gate_logits)
            log_prob = log_prob_minus_log_gate + log_gate
            
            zero_log_prob = softplus(log_prob_minus_log_gate) + log_gate
            log_prob = torch.where(value == 0, zero_log_prob, log_prob)
        return log_prob

    def sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        with torch.no_grad():
            mask = torch.bernoulli(self.gate.expand(shape)).bool()
            samples = self.base_dist.expand(shape).sample()
            samples = torch.where(mask, samples.new_zeros(()), samples)
        return samples

    @lazy_property
    def mean(self):
        return (1 - self.gate) * self.base_dist.mean

    @lazy_property
    def variance(self):
        return (1 - self.gate) * (
            self.base_dist.mean ** 2 + self.base_dist.variance
        ) - (self.mean) ** 2

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(type(self), _instance)
        batch_shape = torch.Size(batch_shape)
        gate = self.gate.expand(batch_shape) if "gate" in self.__dict__ else None
        gate_logits = (
            self.gate_logits.expand(batch_shape)
            if "gate_logits" in self.__dict__
            else None
        )
        base_dist = self.base_dist.expand(batch_shape)
        ZeroInflatedDistribution.__init__(
            new, base_dist, gate=gate, gate_logits=gate_logits, validate_args=False
        )
        new._validate_args = self._validate_args
        return new

# Pyro Distributions.zero_inflated
class ZeroInflatedPoisson(ZeroInflatedDistribution):
    """
    A Zero Inflated Poisson distribution.
    :param torch.Tensor rate: rate of poisson distribution.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        "rate": constraints.positive,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(self, rate, *, gate=None, gate_logits=None, validate_args=None):
        base_dist = Poisson(rate=rate, validate_args=False)
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def rate(self):
        return self.base_dist.rate

# Pyro Distributions.zero_inflated
class ZeroInflatedNegativeBinomial(ZeroInflatedDistribution):
    """
    A Zero Inflated Negative Binomial distribution.
    :param total_count: non-negative number of negative Bernoulli trials.
    :type total_count: float or torch.Tensor
    :param torch.Tensor probs: Event probabilities of success in the closed interval [0, 1].
    :param torch.Tensor logits: Event log-odds for probabilities of success.
    :param torch.Tensor gate: probability of extra zeros.
    :param torch.Tensor gate_logits: logits of extra zeros.
    """

    arg_constraints = {
        "total_count": constraints.greater_than_eq(0), # Should the prob be half-open?
        "probs": constraints.interval(0.0, 1.0),
        "logits": constraints.real,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }
    support = constraints.nonnegative_integer

    def __init__(
        self,
        total_count,
        *,
        probs=None,
        logits=None,
        gate=None,
        gate_logits=None,
        validate_args=None
    ):

        base_dist = NegativeBinomial(
            total_count=total_count,
            probs=probs,
            logits=logits,
            validate_args=False,
        )
        base_dist._validate_args = validate_args

        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )

    @property
    def total_count(self):
        return self.base_dist.total_count

    @property
    def probs(self):
        return self.base_dist.probs

    @property
    def logits(self):
        return self.base_dist.logits

class ZeroInflatedGaussian(ZeroInflatedDistribution):
    arg_constraints = {
        "mean": constraints.real,
        "variance": constraints.greater_than(0),
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real,
    }

    def __init__(
        self,
        mean,
        *,
        variance,
        gate=None,
        gate_logits=None,
        validate_args=None
    ) -> None:
        base_dist = Normal(mean, variance)
        base_dist._validate_args = validate_args
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )
    
    @property
    def mean(self):
        return self.base_dist.mean 
    
    @property
    def variance(self):
        return self.base_dist.variance

class ZeroInflatedBeta(ZeroInflatedDistribution):
    arg_constraints = {
        "concentration1": constraints.real,
        "concentration2": constraints.real,
        "gate": constraints.unit_interval,
        "gate_logits": constraints.real
    }

    def __init__(
        self,
        concentration1,
        concentration2,
        gate=None,
        gate_logits=None,
        validate_args=None
    ) -> None:  
        base_dist = Beta(concentration1, concentration2)
        super().__init__(
            base_dist, gate=gate, gate_logits=gate_logits, validate_args=validate_args
        )
    
    @property
    def mean(self):
        return self.concentration1 / (self.concentration1 + self.concentration2)
    
    @property
    def variance(self):
        return (self.concentration1 * self.concentration2) / ((self.concentration1 + self.concentration2) ** 2 + (self.concentration1 + self.concentration2 + 1))