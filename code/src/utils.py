import numpy as np
import matplotlib.pyplot as plt

def compute_pixel_discrepancy(original: np.ndarray, reconstruction: np.ndarray) -> np.ndarray:
    """Computes per-pixel Mean Squared Error across RGB channels."""
    return np.mean((original - reconstruction) ** 2, axis=-1)

def plot_anomaly_results(samples: list, save_path: str = "evaluation_grid.png"):
    """
    Generates a 3-column evaluation figure: Original | Reconstruction | Anomaly Heatmap
    """
    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 3, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx, (orig, recon, label) in enumerate(samples):
        diff_map = compute_pixel_discrepancy(orig, recon)
        label_text = "Defect" if label != 0 else "Normal"

        axes[idx, 0].imshow(orig)
        axes[idx, 0].set_title(f"Original ({label_text})")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(recon)
        axes[idx, 1].set_title("Reconstruction")
        axes[idx, 1].axis("off")

        heatmap = axes[idx, 2].imshow(diff_map, cmap="jet")
        axes[idx, 2].set_title("Residual Error Map")
        axes[idx, 2].axis("off")
        fig.colorbar(heatmap, ax=axes[idx, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()