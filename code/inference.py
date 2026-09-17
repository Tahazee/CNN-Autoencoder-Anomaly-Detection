import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model import ConvAutoencoder

def predict_single_image(image_path: str, weights_path: str, threshold: float = 0.05):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Architecture & Weights
    model = ConvAutoencoder(latent_dim=128).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    # Forward
    with torch.no_grad():
        reconstruction = model(tensor)
        error_map = ((tensor - reconstruction) ** 2).mean(dim=1).squeeze(0)
        max_error = error_map.max().item()

    is_anomaly = max_error > threshold
    return {
        "anomaly_score": max_error,
        "is_defective": is_anomaly,
        "reconstruction": reconstruction.squeeze(0).cpu().permute(1, 2, 0).numpy()
    }

if __name__ == "__main__":
    result = predict_single_image("sample.png", "weights/autoencoder_mvtec.pt")
    print(f"Defect Detected: {result['is_defective']} | Max Error: {result['anomaly_score']:.5f}")