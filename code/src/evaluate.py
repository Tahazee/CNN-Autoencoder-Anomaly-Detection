import torch
import torch.nn.functional as F

def evaluate_model(model, test_loader, loss_fn, device):
    """
    Evaluates reconstruction error across normal and anomalous splits.
    """
    model.eval()
    test_loss = 0.0
    normal_scores = []
    defect_scores = []
    sample_visuals = []

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.image.to(device)
            inputs = F.interpolate(inputs, size=(256, 256))
            labels = batch.gt_label

            outputs = model(inputs)
            batch_loss = loss_fn(outputs, inputs)
            test_loss += batch_loss.item()

            # Error map shape: [B, H, W]
            error_maps = ((inputs - outputs) ** 2).mean(dim=1)

            for i in range(len(error_maps)):
                max_score = error_maps[i].max().item()
                if labels[i] == 0:
                    normal_scores.append(max_score)
                else:
                    defect_scores.append(max_score)

            if len(sample_visuals) < 4:
                for b in range(min(4 - len(sample_visuals), inputs.size(0))):
                    sample_visuals.append((
                        inputs[b].cpu().permute(1, 2, 0).numpy(),
                        outputs[b].cpu().permute(1, 2, 0).numpy(),
                        labels[b].item()
                    ))

    avg_loss = test_loss / len(test_loader)
    return avg_loss, normal_scores, defect_scores, sample_visuals