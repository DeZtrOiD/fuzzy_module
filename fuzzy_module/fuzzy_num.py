from math import log, isinf
from typing import List, Tuple
from .fuzzy_set import ConvexFuzzySet


def create_fuzzy_exp(
        exp_lambda: float = 1.0, shift: float = 0.0, max_alpha: float = 1.0,
        min_alpha: float = 0.01, num_levels: int = 10,
        normalization_exp: bool = True, normalization_set: bool = True,
        reversed: bool = False) -> ConvexFuzzySet:

    if min_alpha <= 0:
        min_alpha = 0.01
    if max_alpha > 1.0:
        max_alpha = 1.0
    if exp_lambda <= 0.0:
        exp_lambda = 1.0
    if num_levels <= 0:
        num_levels = 10
    if isinf(shift):
        raise ValueError('"shift" must be finite.')
    if isinf(exp_lambda):
        raise ValueError('exp_lambda must be finite.')
    if max_alpha <= 0.0:
        raise ValueError('max_alpha is invalid')
    if not (min_alpha < max_alpha):
        raise ValueError('min_alpha is invalid')
    if isinf(num_levels):
        raise ValueError('num_levels must be finite.')

    alpha_levels: List[Tuple[float, float, float]] = []
    coef = 1.0
    max_level = 1.0
    if not normalization_exp:
        # ln(a / lamb): a > lamb -> err
        if max_level > exp_lambda:
            max_level = exp_lambda
        if min_alpha > exp_lambda:
            min_alpha = min(exp_lambda, max_level) / 1.5
        coef = 1.0 / exp_lambda

    for i in range(num_levels):
        alpha = min_alpha + (max_level - min_alpha) * i / (num_levels - 1)
        # alpha = lamb * exp(-lamb * (x - shift)) => x = shift - ln(alpha/lamb)/lamb
        alpha_levels.append((alpha, shift, shift - log(alpha * coef) / exp_lambda))

    support = (shift, float('inf'))
    if reversed:
        support = (-support[1], -support[0])
        for i in range(num_levels):
            alpha_levels[i] = (alpha_levels[i][0], -alpha_levels[i][2], -alpha_levels[i][1])

    return ConvexFuzzySet(alpha_levels, support, max_alpha, normalization_set)
