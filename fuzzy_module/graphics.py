import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from typing import List, Optional
from .fuzzy_set import ConvexFuzzySet
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from math import isinf


def plot_fuzzy_sets(
    fuzzy_sets: List[ConvexFuzzySet],
    ax: Optional[Axes] = None,
    labels: Optional[List[str]] = None,
    colors: Optional[List[str]] | Optional[List[tuple[float, float, float, float]]] = None,
    show_alpha_lines: bool = True,
    fill_alpha: float = 0.2,
    title: str = "Fuzzy sets >_<",
    xlabel: str = "x",
    ylabel: str = "μ(x)",
    infinity_coef: float = 1.5
) -> Axes:
    """
    Args:
        fuzzy_sets: list of fuzzy sets for display
        ax: an Axes object for drawing. If None, a new one is created
        labels: labels for the legend (default: "Set_0", "Set_1", ...)
        colors: colors for each set (default: from a palette)
        show_alpha_lines: whether to draw horizontal lines for alpha-levels
        fill_alpha: transparency of the fill under the curve
        title, xlabel, ylabel: axis titles
    Returns:
        Axes on which the plot is built
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if fuzzy_sets is None:
        return ax

    if colors is None:
        cmap = plt.get_cmap("tab20")
        colors = [cmap(i) for i in range(cmap.N)]

    if labels is None:
        labels = [f"Set_{i}" for i in range(len(fuzzy_sets))]

    for i, fs in enumerate(fuzzy_sets):
        color = colors[i % len(colors)]
        label = labels[i]

        left_x = []
        left_y = []
        right_x = []
        right_y = []

        support_left, support_right = fs.support
        left_x.append(support_left)
        left_y.append(0.0)
        right_x.append(support_right)
        right_y.append(0.0)

        max_left = fs.alpha_levels[-1][1]
        max_right = fs.alpha_levels[-1][2]

        for alpha, left, right in fs.alpha_levels:
            if left < max_left and not isinf(left):
                max_left = left
            if right > max_right and not isinf(right):
                max_right = right
            left_x.append(left)
            left_y.append(alpha)
            right_x.append(right)
            right_y.append(alpha)

        if max_left > 0:
            max_left /= infinity_coef
        else:
            max_left *= infinity_coef
        if max_right < 0:
            max_right /= infinity_coef
        else:
            max_right *= infinity_coef

        for i in range(len(left_x)):
            if isinf(left_x[i]):
                left_x[i] = max_left
        for i in range(len(right_x)):
            if isinf(right_x[i]):
                right_x[i] = max_right

        # closed shape for filling
        contour_x = left_x + right_x[::-1]
        contour_y = left_y + right_y[::-1]

        ax.plot(contour_x, contour_y, color=color, label=label, linewidth=2)

        ax.fill(contour_x, contour_y, color=color, alpha=fill_alpha)

        if show_alpha_lines:
            for alpha, left, right in fs.alpha_levels:
                ax.plot([left, right], [alpha, alpha], color=color, linestyle='--', alpha=0.3, linewidth=0.5)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    max_alpha = max(fs.max_alpha for fs in fuzzy_sets)
    ax.set_ylim(0, max_alpha * 1.05)

    return ax


def plot_fuzzy_sets_3d(
    fuzzy_sets: List['ConvexFuzzySet'],
    labels: List[int],
    ax: Optional[Axes3D] = None,
    title: str = "Fuzzy sets in 3D (L, R, α)",
    cmap_name: str = "tab10",
    linewidth: float = 1.5,
    alpha_line: float = 0.85,
    infinity_coef: float = 1.5,
    min_alpha: float = 0
) -> Axes:
    """
    3D-coordinates (L(α), R(α), α).
    Args:
        labels: list of cluster labels (one per set, in the same order)
        min_alpha: cuts off everything below this level
    Returns:
        Axes on which the plot is built
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

    unique_labels = sorted(set(labels))
    n_clusters = len(unique_labels)
    cmap = plt.get_cmap(cmap_name)

    if n_clusters == 1:
        label_to_color = {unique_labels[0]: cmap(0.5)}
    else:
        label_to_color = {
            lbl: cmap(i / (n_clusters - 1))
            for i, lbl in enumerate(unique_labels)
        }

    for fs, lbl in zip(fuzzy_sets, labels):
        color = label_to_color[lbl]

        xs: List[float] = []  # L(α)
        ys: List[float] = []  # R(α)
        zs: List[float] = []  # α

        sup_l, sup_r = fs.support
        if min_alpha <= 0:
            xs.append(sup_l)
            ys.append(sup_r)
            zs.append(0.0)

        max_left = fs.alpha_levels[-1][1]
        max_right = fs.alpha_levels[-1][2]

        for alpha, left, right in reversed(fs.alpha_levels):
            if alpha >= min_alpha:
                if left < max_left and not isinf(left):
                    max_left = left
                if right > max_right and not isinf(right):
                    max_right = right
                xs.append(left)
                ys.append(right)
                zs.append(alpha)
            else:
                break

        if min_alpha > 0 and len(xs) != 0:
            left, right = fs._interpolate(min_alpha)
            xs.append(left)
            ys.append(right)
            zs.append(min_alpha)

        if max_left > 0:
            max_left /= infinity_coef
        else:
            max_left *= infinity_coef
        if max_right < 0:
            max_right /= infinity_coef
        else:
            max_right *= infinity_coef
        for i in range(len(xs)):
            if isinf(xs[i]):
                xs[i] = max_left
        for i in range(len(ys)):
            if isinf(ys[i]):
                ys[i] = max_right

        ax.plot(xs, ys, zs, color=color, linewidth=linewidth, alpha=alpha_line)

    ax.set_xlabel('L(α) — Left bound')
    ax.set_ylabel('R(α) — Right bound')
    ax.set_zlabel('α')
    ax.set_title(title)

    legend_elements = [
        Line2D([0], [0], color=label_to_color[lbl], lw=2.5, label=f'Cluster {lbl}')
        for lbl in unique_labels
    ]
    ax.legend(handles=legend_elements, loc='best')

    return ax
