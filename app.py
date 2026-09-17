import glob
import html
import os
from typing import List, Optional

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

plt.switch_backend("Agg")

# 1. Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Exact Model Architecture (Matches Colab Checkpoint)
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 256x256 -> 128x128
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 128x128 -> 64x64
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), # 64x64 -> 32x32
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1) # 32x32 -> 16x16
        )
        self.flatten = nn.Flatten()
        self.fc_encode = nn.Linear(256 * 16 * 16, 128)
        self.fc_decode = nn.Linear(128, 256 * 16 * 16)
        self.unflatten = nn.Unflatten(1, (256, 16, 16))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), # 16x16 -> 32x32
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),   # 64x64 -> 128x128
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),    # 128x128 -> 256x256
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        bottleneck = self.fc_encode(x)
        x = self.fc_decode(bottleneck)
        x = self.unflatten(x)
        return self.decoder_conv(x)

# 3. Model Weight Loading
MODEL_PATH = "autoencoder_mvtec.pth"
encoder = Autoencoder().to(device)
encoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
encoder.eval()

transform_pipeline = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# 4. Dataset Catalog Discovery
base_test_dir = os.path.join("bottle", "test")
categories = [
    d for d in os.listdir(base_test_dir)
    if os.path.isdir(os.path.join(base_test_dir, d))
] if os.path.exists(base_test_dir) else []
categories.sort()

def get_category_images(category_name):
    if not category_name:
        return []
    cat_dir = os.path.join(base_test_dir, category_name)
    return sorted(glob.glob(os.path.join(cat_dir, "*.png")) + glob.glob(os.path.join(cat_dir, "*.jpg")))

# 5. Diagnostic Charts Generation
def generate_full_diagnostics(orig_np, recon_np, error_map, threshold, current_category):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=120)
    fig.patch.set_facecolor("#ffffff")

    flat_errors = error_map.flatten()
    current_max = float(np.max(error_map))
    current_mean = float(np.mean(error_map))

    for row in axes:
        for ax in row:
            ax.set_facecolor("#fafbfc")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")

    # 1. ECDF
    sorted_err = np.sort(flat_errors)
    y_vals = np.arange(len(sorted_err)) / float(len(sorted_err))
    axes[0, 0].plot(sorted_err, y_vals, color="#2563eb", linewidth=2)
    axes[0, 0].axvline(threshold, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Cutoff (τ={threshold:.2f})")
    axes[0, 0].set_title("1. Cumulative Defect Penetration (ECDF)", fontsize=10, fontweight="bold")
    axes[0, 0].set_ylabel("Fraction of Pixels")
    axes[0, 0].legend(loc="lower right", fontsize=8)
    axes[0, 0].grid(True, linestyle=":", alpha=0.5)

    # 2. Dynamic Metric Bar Comparison
    labels = ["Sample Mean", "Sample 99th %ile", "Sample Max", "Cutoff Line"]
    vals = [current_mean, float(np.percentile(flat_errors, 99)), current_max, threshold]
    colors = ["#3b82f6", "#f59e0b", "#dc2626" if current_max > threshold else "#10b981", "#64748b"]
    bars = axes[0, 1].bar(labels, vals, color=colors, width=0.5)
    axes[0, 1].set_title(f"2. Dynamic Error Metrics ({current_category})", fontsize=10, fontweight="bold")
    axes[0, 1].set_ylabel("MSE Error Level")
    axes[0, 1].grid(axis="y", linestyle=":", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width() / 2.0, h + 0.008, f"{h:.3f}", ha="center", va="bottom", fontsize=8)

    # 3. Channel breakdown
    channel_errors = np.mean((orig_np - recon_np) ** 2, axis=(0, 1))
    axes[1, 0].bar(["Red", "Green", "Blue"], channel_errors, color=["#ef4444", "#22c55e", "#3b82f6"], width=0.45)
    axes[1, 0].set_title("3. Color Degradation Breakdown", fontsize=10, fontweight="bold")
    axes[1, 0].set_ylabel("Channel MSE")
    axes[1, 0].grid(axis="y", linestyle=":", alpha=0.5)
    for idx, val in enumerate(channel_errors):
        axes[1, 0].text(idx, val + (val * 0.05 + 1e-5), f"{val:.4f}", ha="center", va="bottom", fontsize=8)

    # 4. Boxplot
    axes[1, 1].boxplot(
        flat_errors, vert=False, patch_artist=True,
        boxprops=dict(facecolor="#e0e7ff", color="#4338ca"),
        medianprops=dict(color="#dc2626", linewidth=1.5),
        flierprops=dict(marker="o", markersize=2, markerfacecolor="#f87171", alpha=0.3)
    )
    axes[1, 1].axvline(threshold, color="#dc2626", linestyle="--", linewidth=1.5, label=f"Cutoff ({threshold:.2f})")
    axes[1, 1].set_title("4. Pixel Spread & Anomaly Outliers", fontsize=10, fontweight="bold")
    axes[1, 1].set_xlabel("Reconstruction MSE")
    axes[1, 1].set_yticks([])
    axes[1, 1].legend(loc="upper right", fontsize=8)
    axes[1, 1].grid(axis="x", linestyle=":", alpha=0.5)

    plt.tight_layout()
    fig.canvas.draw()
    rgba = fig.canvas.buffer_rgba()
    plot_img = np.asarray(rgba)[:, :, :3].copy()
    plt.close(fig)
    return plot_img

# 6. Core Inspection Logic
def inspect_image(image_path: Optional[str], threshold: float, current_category: str):
    if not image_path:
        return None, None, None, None, "Select an image from the catalog below."

    pil_img = Image.open(image_path).convert("RGB")
    tensor_input = transform_pipeline(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        recon_tensor = encoder(tensor_input)
        error_maps = ((tensor_input - recon_tensor) ** 2).mean(dim=1)
        max_error = float(error_maps[0].max().item())
        mean_error = float(error_maps[0].mean().item())

    orig_np = tensor_input.squeeze(0).cpu().permute(1, 2, 0).numpy()
    recon_np = recon_tensor.squeeze(0).cpu().permute(1, 2, 0).numpy()
    diff_map = error_maps[0].cpu().numpy()

    # Heatmap colorization
    norm_map = (diff_map - diff_map.min()) / (diff_map.max() - diff_map.min() + 1e-8)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * norm_map), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted((orig_np * 255).astype(np.uint8), 0.65, heatmap_colored, 0.35, 0)

    anomaly_mask = (diff_map > threshold).astype(np.uint8) * 255
    defect_pixels = int(np.sum(diff_map > threshold))

    is_defect = max_error > threshold
    verdict = "DEFECT DETECTED" if is_defect else "NORMAL (CLEAN)"

    metrics_text = (
        f"DIAGNOSTIC AUDIT LOG\n"
        f"─────────────────────────────────────\n"
        f"• Verdict           : {verdict}\n"
        f"• Max Local Error   : {max_error:.5f}\n"
        f"• Cutoff Threshold  : {threshold:.5f}\n"
        f"• Mean Image MSE    : {mean_error:.5f}\n"
        f"• Defective Pixels  : {defect_pixels:,} / 65,536\n"
        f"• Category Folder   : {current_category}"
    )

    plot_img = generate_full_diagnostics(orig_np, recon_np, diff_map, threshold, current_category)
    return (recon_np * 255).astype(np.uint8), overlay, anomaly_mask, plot_img, metrics_text

# 7. Unified Swarm Three.js Topology (64 balls continuously collapsing to 1 ball)
_THREEJS_DOCUMENT = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<style>
  html, body { margin: 0; padding: 0; background: #0b1120; overflow: hidden; font-family: sans-serif; }
  #three-view { width: 100vw; height: 100vh; position: relative; }
  #mesh-info { position: absolute; top: 12px; right: 16px; background: rgba(15, 23, 42, 0.95);
               color: #38bdf8; padding: 10px 14px; border-radius: 6px; font-family: monospace;
               font-size: 12px; border: 1px solid #1e293b; pointer-events: none; z-index: 10; }
  #legend { position: absolute; bottom: 12px; left: 16px; color: #94a3b8; font-size: 11px; pointer-events: none; }
  canvas { width: 100%; height: 100%; display: block; }
</style>
</head>
<body>
<div id="three-view">
  <div id="mesh-info">Hover over any layer block to inspect spatial resolution</div>
  <div id="legend">🟢 Input/Output (65,536 px | 64 Swarm Balls) &nbsp;|&nbsp; 🟣 Bottleneck (128D | 1 Core Ball)</div>
  <canvas id="three-canvas"></canvas>
</div>

<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
(function() {
    const canvas = document.getElementById('three-canvas');
    const info = document.getElementById('mesh-info');

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.set(0, 16, 46);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.9);
    dirLight.position.set(20, 30, 20);
    scene.add(dirLight);

    const layers = [
        { w: 4.8, h: 4.8, d: 0.3, col: 0x10b981, name: "Input Image", pixels: "65,536 px", shape: "256x256x3" },
        { w: 3.8, h: 3.8, d: 0.6, col: 0x3b82f6, name: "Conv1 (Down)", pixels: "16,384 px", shape: "128x128x32" },
        { w: 2.8, h: 2.8, d: 1.0, col: 0x3b82f6, name: "Conv2 (Down)", pixels: "4,096 px", shape: "64x64x64" },
        { w: 1.9, h: 1.9, d: 1.5, col: 0x3b82f6, name: "Conv3 (Down)", pixels: "1,024 px", shape: "32x32x128" },
        { w: 1.2, h: 1.2, d: 2.2, col: 0x3b82f6, name: "Conv4 (Down)", pixels: "256 px", shape: "16x16x256" },
        { w: 0.4, h: 0.4, d: 4.2, col: 0xa855f7, name: "Bottleneck", pixels: "128 Values", shape: "128D Latent" },
        { w: 1.9, h: 1.9, d: 1.5, col: 0xf97316, name: "Deconv1 (Up)", pixels: "1,024 px", shape: "32x32x128" },
        { w: 2.8, h: 2.8, d: 1.0, col: 0xf97316, name: "Deconv2 (Up)", pixels: "4,096 px", shape: "64x64x64" },
        { w: 3.8, h: 3.8, d: 0.6, col: 0xf97316, name: "Deconv3 (Up)", pixels: "16,384 px", shape: "128x128x32" },
        { w: 4.8, h: 4.8, d: 0.3, col: 0x10b981, name: "Output Image", pixels: "65,536 px", shape: "256x256x3" }
    ];

    function makeTextSprite(message, colorStr) {
        const canvasText = document.createElement('canvas');
        canvasText.width = 340;
        canvasText.height = 120;
        const context = canvasText.getContext('2d');
        context.font = "Bold 34px monospace";
        context.fillStyle = "rgba(15, 23, 42, 0.85)";
        context.strokeStyle = colorStr;
        context.lineWidth = 3;
        context.roundRect(10, 10, 320, 100, 16);
        context.fill();
        context.stroke();

        context.fillStyle = colorStr;
        context.textAlign = "center";
        context.textBaseline = "middle";
        context.fillText(message, 170, 60);

        const texture = new THREE.Texture(canvasText);
        texture.needsUpdate = true;
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture }));
        sprite.scale.set(3.4, 1.2, 1.0);
        return sprite;
    }

    const meshes = [];
    const spacing = 3.8;
    const startX = -((layers.length - 1) * spacing) / 2;

    layers.forEach((l, i) => {
        const x = startX + (i * spacing);
        const geom = new THREE.BoxGeometry(l.d, l.h, l.w);
        const mat = new THREE.MeshStandardMaterial({
            color: l.col, roughness: 0.25, metalness: 0.15, transparent: true, opacity: 0.85
        });
        const mesh = new THREE.Mesh(geom, mat);
        mesh.position.set(x, 0, 0);
        mesh.userData = l;
        scene.add(mesh);
        meshes.push(mesh);

        const wire = new THREE.LineSegments(
            new THREE.EdgesGeometry(geom),
            new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.3 })
        );
        mesh.add(wire);

        const colorHex = '#' + l.col.toString(16).padStart(6, '0');
        const labelSprite = makeTextSprite(l.pixels, colorHex);
        labelSprite.position.set(x, l.h / 2 + 1.2, 0);
        scene.add(labelSprite);
    });

    // Unified 64 Particle Swarm
    const TOTAL_BALLS = 64;
    const ballMeshes = [];
    const totalSpan = (layers.length - 1) * spacing;
    const baseBallGeo = new THREE.SphereGeometry(0.14, 12, 12);

    for (let i = 0; i < TOTAL_BALLS; i++) {
        const isCore = (i === 0);
        const mat = new THREE.MeshBasicMaterial({
            color: isCore ? 0xf43f5e : 0x38bdf8,
            transparent: true,
            opacity: 1.0
        });
        const mesh = new THREE.Mesh(baseBallGeo, mat);
        
        // Spherical distribution around cylinder
        const theta = Math.random() * Math.PI * 2;
        const radialFraction = isCore ? 0.0 : Math.sqrt(Math.random());
        
        // Layer index cutoff for dynamic extinction:
        let extinctionThreshold = 1.0;
        if (i >= 32) extinctionThreshold = 0.11;     // Drop after Conv1
        else if (i >= 16) extinctionThreshold = 0.22; // Drop after Conv2
        else if (i >= 8)  extinctionThreshold = 0.33; // Drop after Conv3
        else if (i >= 1)  extinctionThreshold = 0.44; // Drop after Conv4 (Leave 1 in Bottleneck)

        mesh.userData = {
            id: i,
            isCore: isCore,
            theta: theta,
            radialFraction: radialFraction,
            extinctionThreshold: extinctionThreshold,
            offset: i / TOTAL_BALLS
        };

        scene.add(mesh);
        ballMeshes.push(mesh);
    }

    let isDragging = false;
    let prevX = 0, prevY = 0;
    window.addEventListener('mousedown', (e) => { isDragging = true; prevX = e.clientX; prevY = e.clientY; });
    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', (e) => {
        if (isDragging) {
            scene.rotation.y += (e.clientX - prevX) * 0.008;
            scene.rotation.x += (e.clientY - prevY) * 0.008;
            prevX = e.clientX;
            prevY = e.clientY;
        }

        const mouse = new THREE.Vector2((e.clientX / window.innerWidth) * 2 - 1, -(e.clientY / window.innerHeight) * 2 + 1);
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(meshes);
        if (hits.length > 0) {
            const d = hits[0].object.userData;
            info.innerHTML = `<strong>${d.name}</strong><br>Pixel Count: ${d.pixels}<br>Tensor: ${d.shape}`;
        }
    });

    window.addEventListener('wheel', (e) => {
        camera.position.z = Math.max(16, Math.min(75, camera.position.z + e.deltaY * 0.04));
    }, { passive: true });

    window.addEventListener('resize', () => {
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();
        renderer.setSize(window.innerWidth, window.innerHeight);
    });

    function animate(time) {
        requestAnimationFrame(animate);

        // Constant forward velocity
        const baseProgress = (time * 0.00015);

        ballMeshes.forEach((mesh) => {
            const u = mesh.userData;
            const p = (baseProgress + u.offset) % 1.0;

            // X-axis linear transit across the architecture
            mesh.position.x = startX + (p * totalSpan);

            // Calculate active layer spatial envelope
            const normalizedDistToCenter = Math.abs(p - 0.5) * 2.0; // 0.0 at center, 1.0 at ends
            const currentEnvelopeRadius = 0.2 + (normalizedDistToCenter * 1.8);

            // Calculate survival opacity based on bottleneck proximity
            if (u.isCore) {
                mesh.material.opacity = 1.0;
                mesh.scale.set(1.4, 1.4, 1.4);
                mesh.position.y = Math.sin(p * Math.PI * 8) * 0.15;
                mesh.position.z = Math.cos(p * Math.PI * 8) * 0.15;
            } else {
                let opacity = 1.0;
                if (normalizedDistToCenter < u.extinctionThreshold) {
                    opacity = THREE.MathUtils.clamp((normalizedDistToCenter - (u.extinctionThreshold - 0.08)) / 0.08, 0.0, 1.0);
                }
                mesh.material.opacity = opacity;

                const r = u.radialFraction * currentEnvelopeRadius;
                mesh.position.y = Math.sin(u.theta + p * 4.0) * r;
                mesh.position.z = Math.cos(u.theta + p * 4.0) * r;
                
                const s = 0.7 + normalizedDistToCenter * 0.5;
                mesh.scale.set(s, s, s);
            }
        });

        renderer.render(scene, camera);
    }
    animate(0);
})();
</script>
</body>
</html>
"""

threejs_interactive_html = (
    f'<iframe srcdoc="{html.escape(_THREEJS_DOCUMENT)}" '
    f'style="width:100%; height:540px; border:none; border-radius:12px; '
    f'box-shadow: 0 1px 3px rgba(0,0,0,0.15);" '
    f'title="Autoencoder 3D Architecture Viewer"></iframe>'
)

# 8. Gradio User Interface with Fixed Tab Navigation CSS
custom_css = """
.gradio-container { max-width: 1320px !important; margin: auto !important; padding: 18px !important; }
.panel-header { font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 6px; }

/* Keep tab buttons aligned without layout shifts */
div[role="tablist"] {
    display: flex !important;
    gap: 8px !important;
    border-bottom: 2px solid #e2e8f0 !important;
    margin-bottom: 16px !important;
}

div[role="tablist"] button {
    padding: 10px 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
    background: transparent !important;
    color: #64748b !important;
    transition: all 0.2s ease !important;
}

div[role="tablist"] button[aria-selected="true"] {
    color: #2563eb !important;
    border-bottom: 2px solid #2563eb !important;
    background: #eff6ff !important;
}
"""

with gr.Blocks(title="Surface Defect Inspection Platform") as demo:
    gr.Markdown(
        """
        # Industrial Visual Defect Detection & Diagnostic Studio
        ### Unsupervised Surface Defect Localization via Convolutional Compression
        """
    )

    with gr.Tabs():
        with gr.TabItem("Live Diagnostic Dashboard"):
            with gr.Group():
                gr.Markdown("<div class='panel-header'>Visual Stream Analysis</div>")
                with gr.Row():
                    selected_preview = gr.Image(label="1. Inspected Bottle", interactive=False)
                    out_recon = gr.Image(label="2. Latent Reconstruction", interactive=False)
                    out_overlay = gr.Image(label="3. Localized Heatmap", interactive=False)
                    out_mask = gr.Image(label="4. Segmented Defect Mask", interactive=False)

            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    gr.Markdown("<div class='panel-header'>Sensitivity Control & Logs</div>")
                    threshold_slider = gr.Slider(
                        0.05, 0.50, value=0.23, step=0.01,
                        label="Anomaly Decision Cutoff (τ)",
                        info="Default cutoff calibrated at 0.23 (Normal parts stay < 0.15; defects cross 0.25+)"
                    )
                    metrics_box = gr.Textbox(label="Diagnostic Telemetry", lines=8, interactive=False)

                with gr.Column(scale=2):
                    gr.Markdown("<div class='panel-header'>Statistical Analytics (Dynamic Per Image)</div>")
                    out_plot = gr.Image(label="Real-Time Analytics Suite", interactive=False)

            gr.Markdown("---")
            with gr.Group():
                gr.Markdown("<div class='panel-header'>Dataset Catalog Browser</div>")
                selected_cat = gr.Dropdown(
                    choices=categories,
                    value=categories[0] if categories else None,
                    label="Test Category Folder",
                    interactive=True
                )

                initial_imgs = get_category_images(categories[0]) if categories else []
                gallery = gr.Gallery(
                    value=initial_imgs,
                    label="Inspection Catalog (Click any image to evaluate)",
                    columns=6,
                    rows=2,
                    height=280,
                    object_fit="contain",
                    interactive=False
                )

            selected_cat.change(
                fn=get_category_images,
                inputs=[selected_cat],
                outputs=[gallery]
            )

            def on_sample_click(evt: gr.SelectData, thresh: float, cat: str):
                path = evt.value["image"]["path"]
                rec, over, mask, plot, rep = inspect_image(path, thresh, cat)
                return path, rec, over, mask, plot, rep

            gallery.select(
                fn=on_sample_click,
                inputs=[threshold_slider, selected_cat],
                outputs=[selected_preview, out_recon, out_overlay, out_mask, out_plot, metrics_box]
            )

        with gr.TabItem("Architecture & Connectivity (3D)"):
            gr.Markdown(
                """
                ### Interactive Volumetric Network Topology
                * **Drag** to rotate the network | **Scroll** to zoom.
                * Floating billboards track spatial pixel counts ($H \\times W$) as data compresses from **65,536 px** down to the **128D bottleneck**, then expands back out to **65,536 px**.
                * Animated particle swarms physically downsample in density ($64 \\rightarrow 32 \\rightarrow 16 \\rightarrow 8 \\rightarrow 1 \\rightarrow 8 \\dots$) reflecting active feature dimension scaling.
                """
            )
            gr.HTML(threejs_interactive_html)

        with gr.TabItem("Technology Stack & Algorithm Details"):
            gr.Markdown(
                r"""
                ### Core Mathematical Formulation

                * **Unsupervised Reconstruction Loss (MSE):** The network trains strictly on defect-free samples ($\mathcal{X}_{\text{normal}}$) to minimize Mean Squared Error:
                  $$\mathcal{L}_{\text{MSE}}(x, \hat{x}) = \frac{1}{C \cdot H \cdot W} \sum_{c=1}^{C} \sum_{i=1}^{H} \sum_{j=1}^{W} \left( x_{c,i,j} - \hat{x}_{c,i,j} \right)^2$$

                * **Per-Pixel Anomaly Discrepancy Map:** During test inference, reconstruction discrepancy is averaged across the RGB color channels:
                  $$\mathcal{E}(i, j) = \frac{1}{3} \sum_{c=1}^{3} \left( x_{c,i,j} - \hat{x}_{c,i,j} \right)^2$$

                * **Binary Segmentation & Operational Decision:**
                  $$\mathcal{M}(i, j) = \begin{cases} 255 & \text{if } \mathcal{E}(i, j) > \tau \\ 0 & \text{otherwise} \end{cases}$$
                  Where $\tau = 0.23$ defines the calibrated operational cutoff (normal parts stay $< 0.15$; defects cross $0.25+$).

                ---

                ### System Architecture & Technology Stack

                | Layer / Domain | Technology | Implementation Details |
                | :--- | :--- | :--- |
                | **Deep Learning Framework** | **PyTorch 2.x** (`torch.nn`) | Asymmetric 4-layer Conv2D encoder, 128D latent bottleneck, 4-layer ConvTranspose2D decoder |
                | **Benchmark Dataset** | **Intel Anomalib / MVTec AD** | Industrial visual inspection benchmark (`bottle` category: 209 train, 83 test) |
                | **Optimization & Training** | **Adam Optimizer** | Trained for 100 epochs ($\eta = 10^{-3}$, batch size = 8, MSE loss objective) |
                | **Computer Vision Engine** | **Torchvision & OpenCV** | Strict `Resize((256, 256))`, `ToTensor()` normalization $[0, 1]$, and Jet colormap overlay blending |
                | **Diagnostic Telemetry** | **Matplotlib 3.8+ (`Agg`)** | Headless real-time generation of ECDF curves, 99th-percentile bars, RGB error profiles, and outlier boxplots |
                | **3D Volumetric Engine** | **Three.js (WebGL r128)** | Client-side 3D neural topology with stage-by-stage particle downsampling ($64 \rightarrow 1 \rightarrow 64$) |
                | **UI & Reactive Serving** | **Gradio 6.0 (Blocks API)** | Asynchronous browser-based dashboard with responsive catalog filtering and direct tensor execution |
                | **Hardware Acceleration** | **CUDA / cuDNN** | Automatic fallback to CPU if no discrete NVIDIA GPU is available |
                """
            )


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(), css=custom_css, share=True)