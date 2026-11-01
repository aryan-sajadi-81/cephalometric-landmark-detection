"""Thesis-readable plots for the clinical angle analysis."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from .clinical_analysis import CATEGORY_ORDER, MEASURES
except ImportError:
    from clinical_analysis import CATEGORY_ORDER, MEASURES

COLORS = {"reference": "#2f6f8f", "predicted": "#d97732", "agreement": "#557a46"}


def save_angular_figures(rows, output_dir, stem="clinical_angles"):
    """Save paired-vs-reference scatter, Bland–Altman, and signed/absolute errors."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True); files = []
    for measure in MEASURES:
        ref = np.asarray([row[f"reference_{measure.lower()}_deg"] for row in rows], dtype=float)
        pred = np.asarray([row[f"predicted_{measure.lower()}_deg"] for row in rows], dtype=float)
        diff = pred - ref; average = (pred + ref) / 2.0
        bias = float(np.mean(diff)) if len(diff) else 0.0; sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        if len(ref):
            low = min(float(ref.min()), float(pred.min())); high = max(float(ref.max()), float(pred.max())); pad = max((high - low) * .06, 1.0)
            axes[0].scatter(ref, pred, color=COLORS["predicted"], edgecolor="white", linewidth=.5, alpha=.85)
            axes[0].plot([low-pad, high+pad], [low-pad, high+pad], color="#333", linestyle="--", label="Identity")
            axes[0].set(xlim=(low-pad, high+pad), ylim=(low-pad, high+pad), xlabel="Reference angle (degrees)", ylabel="Predicted angle (degrees)", title=f"{measure}: paired measurements (n={len(ref)})")
            axes[0].legend(frameon=False)
            axes[1].scatter(average, diff, color=COLORS["reference"], edgecolor="white", linewidth=.5, alpha=.85)
            axes[1].axhline(bias, color="#333", label=f"Bias {bias:.2f}°")
            if len(diff) > 1:
                axes[1].axhline(bias + 1.96*sd, color="#aa4b4b", linestyle="--", label="95% limits of agreement")
                axes[1].axhline(bias - 1.96*sd, color="#aa4b4b", linestyle="--")
            axes[1].set(xlabel="Mean of paired angles (degrees)", ylabel="Predicted − reference (degrees)", title=f"{measure}: Bland–Altman")
            axes[1].legend(frameon=False, fontsize=8)
            bins = np.histogram_bin_edges(diff, bins="auto") if len(diff) > 1 else 1
            axes[2].hist(diff, bins=bins, color=COLORS["agreement"], alpha=.82, edgecolor="white", label="Signed difference")
            axes[2].axvline(0, color="#333", linestyle="--", label="No difference")
            axes[2].set(xlabel="Predicted − reference (degrees)", ylabel="Patients", title=f"{measure}: signed error (n={len(diff)})")
            axes[2].legend(frameon=False, fontsize=8)
        else:
            for ax in axes: ax.text(.5, .5, "No valid paired measurements", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(f"{measure} agreement; validation/model-selection cohort", y=1.02); fig.tight_layout()
        base = output_dir / f"{stem}_{measure.lower()}"
        fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
        files.extend([str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))])
    return files


def save_anb_category_figure(rows, output_dir, stem="anb_category_agreement"):
    """Plot paired signed ANB with approved 0° and 4° boundaries on both axes."""
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    colors = {"Class III": "#2f6f8f", "Class I": "#648c55", "Class II": "#d97732"}
    ref = np.asarray([row["reference_anb_deg"] for row in rows], dtype=float); pred = np.asarray([row["predicted_anb_deg"] for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for label in CATEGORY_ORDER:
        mask = np.asarray([row["reference_category"] == label for row in rows])
        if mask.any(): ax.scatter(ref[mask], pred[mask], label=f"Reference {label}", color=colors[label], edgecolor="white", linewidth=.6, alpha=.85)
    if len(ref):
        low = min(float(ref.min()), float(pred.min()), -1.) - .5; high = max(float(ref.max()), float(pred.max()), 5.) + .5
        ax.plot([low, high], [low, high], color="#333", linestyle="--", label="Identity")
        for boundary in (0., 4.): ax.axvline(boundary, color="#777", linestyle=":", linewidth=1.2); ax.axhline(boundary, color="#777", linestyle=":", linewidth=1.2)
        ax.set_xlim(low, high); ax.set_ylim(low, high)
    else: ax.text(.5, .5, "No valid paired measurements", ha="center", va="center", transform=ax.transAxes)
    ax.set(xlabel="Reference-derived signed ANB (degrees)", ylabel="Prediction-derived signed ANB (degrees)", title=f"ANB-derived category agreement (n={len(rows)})")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2); fig.tight_layout()
    base = output_dir / stem; fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight"); fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight"); plt.close(fig)
    return [str(base.with_suffix(".png")), str(base.with_suffix(".pdf"))]
