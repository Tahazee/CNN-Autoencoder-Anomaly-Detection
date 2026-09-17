import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F

from src.model import ConvAutoencoder
from src.dataset import get_dataloaders
from src.evaluate import evaluate_model
from src.utils import plot_anomaly_results

def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["paths"]["weights_dir"], exist_ok=True)

    # 1. Load Data
    train_loader, test_loader = get_dataloaders(
        root=cfg["data"]["root"],
        category=cfg["data"]["category"],
        train_batch_size=cfg["data"]["train_batch_size"],
        eval_batch_size=cfg["data"]["eval_batch_size"]
    )

    # 2. Setup Model, Criterion, Optimizer
    model = ConvAutoencoder(
        in_channels=cfg["model"]["in_channels"],
        latent_dim=cfg["model"]["latent_dim"]
    ).to(device)

    loss_fn = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    # 3. Training Loop
    print(f"Starting training on {device} for {cfg['training']['epochs']} epochs...")
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch.image.to(device)
            inputs = F.interpolate(inputs, size=tuple(cfg["data"]["image_size"]))

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{cfg['training']['epochs']}] - Loss: {epoch_loss:.6f}")

    # 4. Save Weights (State Dict Only)
    save_path = os.path.join(cfg["paths"]["weights_dir"], cfg["paths"]["checkpoint_name"])
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    # 5. Validation & Plotting
    avg_test_loss, normal_scores, defect_scores, sample_visuals = evaluate_model(
        model, test_loader, loss_fn, device
    )
    print(f"Test Loss: {avg_test_loss:.6f}")
    plot_anomaly_results(sample_visuals, save_path="residual_predictions.png")

if __name__ == "__main__":
    main()