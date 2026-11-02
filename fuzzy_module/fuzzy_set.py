# Universe is Real numbers

from typing import List, Tuple
from bisect import insort, bisect_left
from math import isnan, isclose
from copy import deepcopy


class ConvexFuzzySet:
    _STR_PRECISION: int = 6
    _ALPHA_LEVEL_EPS: float = 1e-10
    _METRIC_PENALTY: float = 1e-1
    '''penalty_weight'''
    _DISTANCE_INFL: float = 1e50

    def __init__(
            self, alpha_levels: List[Tuple[float, float, float]], support: Tuple[float, float],
            max_alpha: float = 1.0, normalization: bool = True):
        """
        Args:
            alpha_levels: [(alpha, left, right)]
        """

        if max_alpha > 1.0:
            max_alpha = 1.0
        if max_alpha <= 0:
            raise ValueError('Empty set is not supported')

        if support[0] >= support[1]:
            raise ValueError("Incorrect support.")
        self._support = support
        self._defuzzified: float | None = None
        self._alpha_levels: List[Tuple[float, float, float]] = []
        '''(alpha, left, right)'''

        max_level = max(alpha_levels, key=lambda x: x[0])[0]
        if isclose(max_level, 0.0, rel_tol=self._ALPHA_LEVEL_EPS):
            raise ValueError('Empty set is not supported')

        if max_level < max_alpha and not normalization:
            max_alpha = max_level
        self._max_alpha = max_alpha

        for level in alpha_levels:
            curr_alpha = level[0]
            if normalization:
                curr_alpha /= max_level
            # Filter out elements that exceed the max_alpha threshold
            if curr_alpha <= self._max_alpha:
                insort(self._alpha_levels, (curr_alpha, level[1], level[2]), key=lambda x: x[0])
        if len(self._alpha_levels) == 0:
            if alpha_levels[0][0] / max_level > self._max_alpha:
                t = max_level * self._max_alpha / alpha_levels[0][0]
                left = alpha_levels[0][1] + t * (self._support[0] - alpha_levels[-1][1])
                if self._support[0] == alpha_levels[0][1]:  # inf - inf
                    left = alpha_levels[0][1]
                right = alpha_levels[0][2] + t * (self._support[1] - alpha_levels[-1][2])
                if self._support[1] == alpha_levels[0][2]:  # inf - inf
                    right = alpha_levels[0][2]
                insort(self._alpha_levels, (self._max_alpha, left, right), key=lambda x: x[0])
            else:
                raise ValueError('Empty set is not supported')

        # Validate the nestedness and consistency of the alpha-cuts family
        last_left = float('inf')
        last_right = float('-inf')
        for _, left, right in self._alpha_levels:
            if left > last_left or right < last_right:
                raise ValueError(f"Incorrect alpha-cuts indexed family ({last_left, last_right}) < ({left, right}).")
            if left > right:
                raise ValueError(f"Incorrect alpha-cuts indexed family. Left({left}) > right({right}).")
        if self._alpha_levels[0][1] == float('inf') or self._alpha_levels[0][2] == float('-inf'):
            raise ValueError(f"Incorrect alpha-cuts indexed family. Left == Right == {self._alpha_levels[-1][2]}.")
        if self._alpha_levels[0][1] < self._support[0] or self._alpha_levels[0][2] > self._support[1]:
            raise ValueError("Incorrect alpha-cuts indexed family. The support is smaller than a-cuts.")

    def _defuzzify(self) -> float:
        '''Weighted Average Method'''
        weighted_sum = 0.0
        alpha_sum = 0.0

        for alpha, left, right in self._alpha_levels:
            weighted_sum += alpha * (left + right) / 2
            alpha_sum += alpha

        return weighted_sum / alpha_sum

    @property
    def max_alpha(self):
        return self._max_alpha

    @property
    def support(self):
        return self._support

    @property
    def alpha_levels(self):
        return self._alpha_levels

    @property
    def metric_penalty(self):
        return self._METRIC_PENALTY

    @metric_penalty.setter
    def metric_penalty(self, value: float):
        if value < 0:
            raise ValueError("Metric penalty must be greater than 0.")
        self._METRIC_PENALTY = value

    @property
    def distance_infinity(self):
        return self._DISTANCE_INFL

    @distance_infinity.setter
    def distance_infinity(self, value: float):
        if not (0 < value < float('inf')):
            raise ValueError("Distance infinity must be 0 < value < inf")
        self._DISTANCE_INFL = value

    # -------------------------------------------------------------------------
    # Helpers for algebraic operations
    # -------------------------------------------------------------------------

    @classmethod
    def _getResampledAlphaCuts(
            cls, a: ConvexFuzzySet, b: ConvexFuzzySet, max_alpha
            ) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float, float]]]:

        alphas_a = [level[0] for level in a._alpha_levels if level[0] <= max_alpha]
        alphas_b = [level[0] for level in b._alpha_levels if level[0] <= max_alpha]
        common_alphas = sorted(set(alphas_a + alphas_b))

        if common_alphas and abs(common_alphas[-1] - max_alpha) > cls._ALPHA_LEVEL_EPS:
            max_alpha_a = a._alpha_levels[-1][0] if a._alpha_levels else 0
            max_alpha_b = b._alpha_levels[-1][0] if b._alpha_levels else 0

            if max_alpha <= max(max_alpha_a, max_alpha_b):
                common_alphas.append(max_alpha)
                common_alphas.sort()

        resampled_a: List[Tuple[float, float, float]] = []
        resampled_b: List[Tuple[float, float, float]] = []

        for alpha in common_alphas:
            left_a, right_a = a._interpolate(alpha)
            resampled_a.append((alpha, left_a, right_a))

            left_b, right_b = b._interpolate(alpha)
            resampled_b.append((alpha, left_b, right_b))

        return resampled_a, resampled_b

    def _interpolate(self, target_alpha: float) -> Tuple[float, float]:
        al = self._alpha_levels
        if not al:
            raise ValueError("Cannot interpolate from empty alpha_levels")

        alphas = [level[0] for level in al]

        if target_alpha <= alphas[0]:
            return al[0][1], al[0][2]
        if target_alpha >= alphas[-1]:
            return al[-1][1], al[-1][2]

        idx = bisect_left(alphas, target_alpha)

        if idx < len(alphas) and abs(alphas[idx] - target_alpha) < self._ALPHA_LEVEL_EPS:
            return al[idx][1], al[idx][2]

        alpha_low, left_low, right_low = al[idx - 1]
        alpha_high, left_high, right_high = al[idx]

        t = (target_alpha - alpha_low) / (alpha_high - alpha_low)

        left = left_low + t * (left_high - left_low)
        right = right_low + t * (right_high - right_low)

        return left, right

    # -------------------------------------------------------------------------
    # Algebraic operations
    # -------------------------------------------------------------------------

    def __add__(self, other: ConvexFuzzySet) -> ConvexFuzzySet:
        result = ConvexFuzzySet.__new__(ConvexFuzzySet)
        result._max_alpha = min(self._max_alpha, other._max_alpha)
        result._support = (self._support[0] + other._support[0], self._support[1] + other._support[1])
        result._defuzzified = None

        result._alpha_levels, resampled_b = ConvexFuzzySet._getResampledAlphaCuts(
            self, other, result._max_alpha
        )

        for i in range(len(resampled_b)):
            result._alpha_levels[i] = (
                result._alpha_levels[i][0],
                result._alpha_levels[i][1] + resampled_b[i][1],
                result._alpha_levels[i][2] + resampled_b[i][2]
            )

        return result

    def __sub__(self, other: ConvexFuzzySet) -> ConvexFuzzySet:
        result = ConvexFuzzySet.__new__(ConvexFuzzySet)
        result._max_alpha = min(self._max_alpha, other._max_alpha)
        result._defuzzified = None

        left_a, right_a = self._support
        left_b, right_b = other._support
        left_a -= left_b
        right_a -= right_b
        if isnan(left_a) or isnan(right_a):
            raise ValueError('An indeterminate operation involving infinity.')
        if left_a == float('inf'):
            raise ValueError(f'Incorrect support ({left_a}, {right_a})')
        result._support = (min(left_a, right_a), max(left_a, right_a))

        result._alpha_levels, resampled_b = ConvexFuzzySet._getResampledAlphaCuts(
            self, other, result._max_alpha
        )
        for i in range(len(resampled_b)):
            left = result._alpha_levels[i][1] - resampled_b[i][1]
            right = result._alpha_levels[i][2] - resampled_b[i][2]
            result._alpha_levels[i] = (
                result._alpha_levels[i][0],
                min(left, right),
                max(left, right)
            )
        return result

    def __mul__(self, other: ConvexFuzzySet) -> ConvexFuzzySet:
        # if 2 intervals from the same set contain 'inf',
        # and the 2 corresponding intervals from the other set lie on opposite sides of 0,
        # then the graph will have a break
        result = ConvexFuzzySet.__new__(ConvexFuzzySet)
        result._max_alpha = min(self._max_alpha, other._max_alpha)
        result._defuzzified = None

        left_a, right_a = self._support
        left_b, right_b = other._support
        left_a *= left_b
        right_a *= right_b
        if isnan(left_a) or isnan(right_a):
            raise ValueError('An indeterminate operation involving infinity and zero.')
        if left_a == right_a:
            raise ValueError(f'Incorrect support ({left_a}, {right_a})')
        result._support = (min(left_a, right_a), max(left_a, right_a))

        (result._alpha_levels, resampled_b) = ConvexFuzzySet._getResampledAlphaCuts(self, other, result._max_alpha)
        for i in range(len(resampled_b)):
            left = result._alpha_levels[i][1] * resampled_b[i][1]
            right = result._alpha_levels[i][2] * resampled_b[i][2]
            if isnan(left) or isnan(right):
                raise ValueError("alpha-cuts includes 0 and inf. The operation is undefined.")
            result._alpha_levels[i] = (
                result._alpha_levels[i][0],
                min(left, right),
                max(left, right)
            )
        return result

    def __truediv__(self, other: ConvexFuzzySet) -> ConvexFuzzySet:
        if other._support[0] <= 0 and other._support[1] >= 0:
            raise ValueError("Support includes 0. The operation is undefined.")
        result = ConvexFuzzySet.__new__(ConvexFuzzySet)
        result._max_alpha = min(self._max_alpha, other._max_alpha)
        result._defuzzified = None

        left_a, right_a = self._support
        left_b, right_b = other._support
        left_a = min(left_a / left_b, right_a / left_b, left_a / right_b, right_a / right_b)
        right_a = max(left_a / left_b, right_a / left_b, left_a / right_b, right_a / right_b)
        if isnan(left_a) and isnan(right_a):
            raise ValueError('0 or inf in divisor')

        result._support = (left_a, right_a)

        (result._alpha_levels, resampled_b) = ConvexFuzzySet._getResampledAlphaCuts(self, other, result._max_alpha)
        for i in range(len(resampled_b)):
            left = min(
                result._alpha_levels[i][1] / resampled_b[i][1], result._alpha_levels[i][2] / resampled_b[i][1],
                result._alpha_levels[i][1] / resampled_b[i][2], result._alpha_levels[i][2] / resampled_b[i][2]
            )
            right = max(
                result._alpha_levels[i][1] / resampled_b[i][1], result._alpha_levels[i][2] / resampled_b[i][1],
                result._alpha_levels[i][1] / resampled_b[i][2], result._alpha_levels[i][2] / resampled_b[i][2]
            )
            result._alpha_levels[i] = (result._alpha_levels[i][0], left, right)
        return result

    def __pos__(self):
        return self

    def __neg__(self):
        result = deepcopy(self)
        for i in range(len(result.alpha_levels)):
            result._alpha_levels[i] = (
                result._alpha_levels[i][0],
                -result._alpha_levels[i][2],
                -result._alpha_levels[i][1]
            )
        result._support = (-result._support[1], -result._support[0])
        return result
    # -------------------------------------------------------------------------
    # Comparison via defuzzification (Center of Gravity)
    # -------------------------------------------------------------------------

    def defuzzify(self) -> float:
        if self._defuzzified is None:
            self._defuzzified = self._defuzzify()
        return self._defuzzified

    def __lt__(self, other): return self.defuzzify() < other.defuzzify()
    def __le__(self, other): return self.defuzzify() <= other.defuzzify()
    def __gt__(self, other): return self.defuzzify() > other.defuzzify()
    def __ge__(self, other): return self.defuzzify() >= other.defuzzify()
    def __eq__(self, other): return isclose(self.defuzzify(), other.defuzzify())

    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------

    def get_alpha_level(self, alpha_level: float) -> Tuple[float, float] | None:
        if self._max_alpha < alpha_level or alpha_level < 0:
            return None
        if isclose(alpha_level, 0.0, abs_tol=self._ALPHA_LEVEL_EPS):
            return self._support

        return self._interpolate(alpha_level)

    # -------------------------------------------------------------------------
    # String representation
    # -------------------------------------------------------------------------

    @classmethod
    def set_str_precision(cls, n: int):
        if n < 0:
            raise ValueError("Precision must be non-negative")
        cls._STR_PRECISION = n

    def __str__(self):
        p = ConvexFuzzySet._STR_PRECISION
        lines = []
        line_items = []

        for alpha, left, right in self._alpha_levels:
            item = f"({alpha:.{p}f}, [{left:.{p}f}, {right:.{p}f}])"
            line_items.append(item)

            if len(line_items) == 3:
                lines.append(", ".join(line_items))
                line_items = []

        if line_items:
            lines.append(", ".join(line_items))

        return "\n".join(lines)

    @staticmethod
    def distance(a: ConvexFuzzySet, b: ConvexFuzzySet) -> float:
        """
        L2*(a, b) + penalty*|a.max_alpha - b.max_alpha|.
        """
        max_alpha = min(a._max_alpha, b._max_alpha)
        a_resampled, b_resampled = ConvexFuzzySet._getResampledAlphaCuts(a, b, max_alpha)

        distance_sq = 0.0
        for (_, l_a, r_a), (_, l_b, r_b) in zip(a_resampled, b_resampled):
            left = l_a - l_b
            if isnan(left):
                left = 0
            right = r_a - r_b
            if isnan(right):
                right = 0
            distance_sq += (left)**2 + (right)**2
        distance = distance_sq**0.5

        penalty = ConvexFuzzySet._METRIC_PENALTY * abs(a._max_alpha - b._max_alpha)
        result = distance + penalty
        if result == float('inf'):
            result = ConvexFuzzySet._DISTANCE_INFL
        return result
