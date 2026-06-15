"""
visualize.py
------------
Reusable plotting utilities for time series results.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_actual_vs_predicted(y_true, y_pred, title: str = "Actual vs Predicted",
                             save_path: str = None) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label="Actual",    linewidth=1.5, color="#1f77b4")
    ax.plot(y_pred, label="Predicted", linewidth=1.5, color="#ff7f0e", linestyle="--")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_residuals(y_true, y_pred, save_path: str = None) -> None:
    residuals = np.array(y_true) - np.array(y_pred)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(residuals, color="#2ca02c", linewidth=1)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Residual")

    axes[1].hist(residuals, bins=30, edgecolor="black", color="#9467bd")
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(model, feature_names: list, save_path: str = None) -> None:
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(importances)), importances[idx], color="#17becf")
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
    ax.set_title("XGBoost Feature Importance")
    ax.set_ylabel("Importance Score")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()