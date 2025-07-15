from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from scipy.ndimage.filters import gaussian_filter1d
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, r2_score

from matplotlib.ticker import MultipleLocator


def start_figure(width: int, height: int,
                 theme: str = 'talk',
                 font: str = None,
                 style: str = 'ticks',
                 custom_params: dict = {}) -> plt.Figure:
    if theme == 'paper' and font is None:
        font = 'serif'
    else:
        font = 'sans'

    sns.set_theme(theme, style=style, font=font, rc=custom_params)

    fig = plt.figure(figsize=(width, height))
    return fig


def add_grid(ax: plt.Axes, grid_spacing: float = 0.01,
             in_axis: str = 'x'):
    minorLocator = MultipleLocator(grid_spacing)
    # Set minor tick locations.
    if in_axis == 'x':
        ax.xaxis.set_minor_locator(minorLocator)
    elif in_axis == 'y':
        ax.yaxis.set_minor_locator(minorLocator)
    elif in_axis == 'both':
        ax.xaxis.set_minor_locator(minorLocator)
        ax.yaxis.set_minor_locator(minorLocator)

    # Set grid to use minor tick locations.
    ax.grid(which='both', axis=in_axis)
    ax.set_axisbelow(True)
    return ax


def plot_rec(preds: np.ndarray, truth: np.ndarray,
             ax: plt.Axes = None, 
             figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)

    errors = np.abs(truth - preds)

    # 5. Define error thresholds for REC curve
    error_thresholds = np.linspace(0, errors.max(), 200)
    fraction_within_threshold = [(errors <= t).mean() for t in error_thresholds]

    ax.plot(error_thresholds, fraction_within_threshold, label='REC Curve')
    ax.set_xlabel('Error Tolerance (ε)')
    ax.set_ylabel('Fraction of Predictions with Error ≤ ε')
    ax.set_title('Regression Error Characteristic (REC) Curve')

    add_grid(ax, grid_spacing=(truth.max() - truth.min()) / 20, in_axis='x')
    add_grid(ax, grid_spacing=(truth.max() - truth.min()) / 20, in_axis='y')

    return ax


def plot_actual_vs_pred(preds: np.ndarray, truth: np.ndarray,
                        ax: plt.Axes = None, 
                        figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)

    r2 = r2_score(truth, preds)
    ax.scatter(truth, preds, alpha=0.7, edgecolor='k')
    ax.plot([truth.min(), truth.max()], [truth.min(), truth.max()], color='red', linestyle='--', label='Ideal (y = x)')
    add_grid(ax, grid_spacing=(truth.max() - truth.min()) / 20, in_axis='x')
    add_grid(ax, grid_spacing=(truth.max() - truth.min()) / 20, in_axis='y')
    ax.set_title(f'Actual vs Predicted Values\n$R^2$ = {r2:.2f}')
    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    return ax


def plot_calibration_curve(preds: np.ndarray, truth: np.ndarray,
                           ax: plt.Axes = None, 
                           figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)
    prob_true, prob_pred = calibration_curve(truth, preds, n_bins=10, strategy='uniform')
    ax.plot(prob_pred, prob_true, marker='o', label='Model')
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated', color='gray')
    add_grid(ax, grid_spacing=0.1, in_axis='x')
    add_grid(ax, grid_spacing=0.1, in_axis='y')
    ax.set_title('Calibration Curve')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('True Probability')


def plot_pr_curve(preds: np.ndarray, truth: np.ndarray,
                  ax: plt.Axes = None, figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)

    precision, recall, thresholds = precision_recall_curve(truth, preds)
    avg_precision = average_precision_score(truth, preds)
    ax.plot(recall, precision, color='blue', label=f"PR curve\n(Average Precision = {avg_precision:.2f})")
    # ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')

    ax.set_title('Precision-Recall curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(bbox_to_anchor=(2.2, 0.5), title="Legend")
    add_grid(ax, grid_spacing=0.1, in_axis='x')
    add_grid(ax, grid_spacing=0.1, in_axis='y')

    return ax


def plot_roc_curve(preds: np.ndarray, truth: np.ndarray,
                   ax: plt.Axes = None, figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)

    fpr, tpr, thresholds = roc_curve(truth, preds)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color='blue', label=f"ROC curve (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')

    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    add_grid(ax, grid_spacing=0.1, in_axis='x')
    add_grid(ax, grid_spacing=0.1, in_axis='y')
    ax.legend(bbox_to_anchor=(2.2, 0.5), title="Legend")

    return ax


def plot_optimization_history(history: pd.DataFrame, ax: plt.Axes = None,
                              metric: str = 'mcc',
                              figsize: Tuple[int, int] = (5, 5)) -> plt.Axes:
    if ax is None:
        fig = start_figure(figsize[0], figsize[1])
        ax = fig.subplots(1, 1)

    y_pos = history['run'].unique()
    Final_array = np.array([[v[metric].mean(), v[metric].std()]
                            for g, v in history.groupby('run')])
    max_points = np.argsort(Final_array[:, 0])[-10:]
    sigma = 4

    Final_array_smooth = gaussian_filter1d(Final_array[:, 0], sigma=sigma)
    upper_err = gaussian_filter1d(Final_array[:, 0] + (Final_array[:, 1] / 2),
                                  sigma=sigma)
    lower_err = gaussian_filter1d(Final_array[:, 0] - (Final_array[:, 1] / 2),
                                  sigma=sigma)

    line_1, = ax.plot(y_pos, Final_array_smooth, label='Trajectory')
    line_2 = ax.scatter(y_pos[max_points], Final_array[max_points, 0],
                        color='r', label="Top-10 trials")
    line_3 = ax.scatter(y_pos, Final_array[:, 0],
                        color='b', alpha=0.2,
                        label='All trials')
    line_4 = ax.fill_between(y_pos, upper_err, lower_err, color='crimson',
                             alpha=0.2, label='Standard error')
    ax.set_title("Optimization history")
    ax.set_xlabel("Trial")
    ax.set_ylabel(f"{metric.upper()}")
    add_grid(ax, grid_spacing=5, in_axis='x')
    add_grid(ax, grid_spacing=.1, in_axis='y')
    ax.set_xlim(0, history['run'].max()+3)

    plt.legend(handles=[line_3, line_2, line_1, line_4],
               bbox_to_anchor=(2, 0.5), title="Legend",
            #    loc="right center",
               ncols=2,
               title_fontproperties={"weight": "bold",
                                     "size": figsize[1] * 3},
               prop={"size": figsize[1] * 2.5})

    return ax


def plot_model_vs_rep(history: pd.DataFrame, ax: plt.Axes = None,
                      metric: str = 'mcc',
                      figsize: Tuple[int, int] = (5, 5)):

    if ax is None:
        fig = start_figure(
            figsize[0]*len(history.run.unique())//20, figsize[1]
        )
        ax = fig.subplots(1, 1)

    history['Representation'] = history['representation']
    history['Model'] = history['name']
    history['Run'] = history['run']
    df = history

    agg_df = (
        df.groupby(['Model', 'Run', 'Representation'])
        .agg(avg_metric=(metric, 'mean'))
        .reset_index()
    )

    performance_df = (
        agg_df.groupby(['Model', 'Representation']).agg(
            mean_metric=('avg_metric', 'mean'),
            num_trials=('avg_metric', 'count')
        ).reset_index()
    )

    heatmap_data = performance_df.pivot(
        index='Model', columns='Representation', values='mean_metric'
    )
    annotations = performance_df.pivot(
        index='Model', columns='Representation',
        values='num_trials')

    sns.heatmap(heatmap_data, annot=annotations, cmap='YlGnBu', linewidths=0.5,
                cbar_kws={'label': metric}, ax=ax)
    ax.set_title('Model vs Representation (Annotated with Trial Count)')
    ax.set_ylabel('Model')
    ax.set_xlabel('Representation')
    plt.tight_layout()
    return ax
