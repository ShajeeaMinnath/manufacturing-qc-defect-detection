
# Full corrected production-ready Streamlit app.py
# ResNet-50 + Autoencoder + Grad-CAM + Hybrid Prediction

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from torchvision import models, transforms
import warnings
import ast
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================
# Page Config
# ============================================================
st.set_page_config(
    page_title="Defect Detection — Quality Control",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CSS
# ============================================================
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a1a2e;
    text-align: center;
    margin-bottom: 0.5rem;
}
.sub-header {
    font-size: 1rem;
    color: #666;
    text-align: center;
    margin-bottom: 2rem;
}
.result-normal {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-left: 5px solid #28a745;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    font-size: 1.3rem;
    font-weight: 600;
    color: #155724;
}
.result-defective {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border-left: 5px solid #dc3545;
    padding: 1rem 1.5rem;
    border-radius: 8px;
    font-size: 1.3rem;
    font-weight: 600;
    color: #721c24;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# MODEL DEFINITIONS
# ============================================================
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, latent_dim), nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 14 * 14), nn.ReLU(),
            nn.Unflatten(1, (256, 14, 14)),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


# ============================================================
# GRAD-CAM
# ============================================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def generate(self, img_tensor, target_class):
        self.model.eval()
        img_tensor = img_tensor.requires_grad_(True)
        output = self.model(img_tensor)
        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam


# ============================================================
# LOAD MODELS + CONFIG
# ============================================================
@st.cache_resource
def load_models():
    device = torch.device("cpu")
    save_dir = Path(__file__).parent / "saved_models"

    required_files = [
        save_dir / "resnet50_best.pth",
        save_dir / "autoencoder_best.pth",
        save_dir / "ae_threshold.txt",
        save_dir / "class_mapping.txt"
    ]

    for file in required_files:
        if not file.exists():
            st.error(f"Missing required file: {file}")
            st.stop()

    resnet = ResNet50Classifier(2)
    resnet.load_state_dict(torch.load(save_dir / "resnet50_best.pth", map_location=device))
    resnet.eval()

    autoencoder = ConvAutoencoder(512)
    autoencoder.load_state_dict(torch.load(save_dir / "autoencoder_best.pth", map_location=device))
    autoencoder.eval()

    with open(save_dir / "ae_threshold.txt", "r") as f:
        ae_threshold = float(f.read())

    with open(save_dir / "class_mapping.txt", "r") as f:
        class_mapping = ast.literal_eval(f.read())

    return resnet, autoencoder, ae_threshold, class_mapping


# ============================================================
# PREPROCESSING (FIXED)
# ============================================================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ResNet preprocessing
resnet_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
])

# Autoencoder preprocessing (NO normalization)
ae_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess_image(img):
    rgb = img.convert("RGB")

    resnet_tensor = resnet_transform(rgb).unsqueeze(0)

    ae_tensor = ae_transform(rgb).unsqueeze(0)

    return resnet_tensor, ae_tensor, rgb


def overlay_gradcam(original, heatmap, alpha=0.45):
    heatmap_resized = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap_uint8 = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * original.astype(np.float32) + alpha * heatmap_colored.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


# ============================================================
# MAIN APP
# ============================================================
def main():
    st.markdown('<div class="main-header">🔍 Manufacturing Defect Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">ResNet-50 + Autoencoder + Grad-CAM Hybrid System</div>', unsafe_allow_html=True)

    resnet, autoencoder, AE_THRESHOLD, class_mapping = load_models()

    DEFECTIVE_CLASS = class_mapping['defective']
    GOOD_CLASS = class_mapping['good']

    with st.sidebar:
        st.header("⚙️ Settings")
        threshold = st.slider(
            "Defect Classification Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.35,
            step=0.05
        )
        show_gradcam = st.checkbox("Show Grad-CAM", value=True)
        st.markdown("---")
        st.write(f"AE Threshold: {AE_THRESHOLD:.6f}")

    uploaded = st.file_uploader("Upload Product Image", type=["png", "jpg", "jpeg"])

    if uploaded:
        img = Image.open(uploaded)
        resnet_tensor, ae_tensor, rgb = preprocess_image(img)
        img_array = np.array(rgb.resize((224, 224)))

        st.image(rgb, caption="Uploaded Image", width=300)

        # ResNet Prediction
        with torch.no_grad():
            output = resnet(resnet_tensor)
            probs = torch.softmax(output, dim=1)[0].cpu().numpy()

        defective_prob = probs[DEFECTIVE_CLASS]
        good_prob = probs[GOOD_CLASS]

        # Autoencoder
        with torch.no_grad():
            recon, _ = autoencoder(ae_tensor)
            mse = torch.nn.functional.mse_loss(recon, ae_tensor).item()

        # Hybrid Decision
        is_defective = (mse > AE_THRESHOLD) or (defective_prob >= threshold)

        # GradCAM
        if show_gradcam:
            gc = GradCAM(resnet, resnet.backbone.layer4[-1])
            heatmap = gc.generate(resnet_tensor, DEFECTIVE_CLASS)
            overlay = overlay_gradcam(img_array, heatmap)

            col1, col2 = st.columns(2)
            with col1:
                st.image((plt.cm.jet(cv2.resize(heatmap, (224,224)))[:, :, :3] * 255).astype(np.uint8), caption="Grad-CAM")
            with col2:
                st.image(overlay, caption="Defect Overlay")

        # Final Result
        st.markdown("---")
        if is_defective:
            st.markdown(
                f'<div class="result-defective">⚠️ DEFECTIVE — ResNet: {defective_prob*100:.2f}% | AE MSE: {mse:.6f}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-normal">✅ NORMAL — Good: {good_prob*100:.2f}% | AE MSE: {mse:.6f}</div>',
                unsafe_allow_html=True
            )

        # Debug Metrics
        st.markdown("---")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("P(Defective)", f"{defective_prob*100:.2f}%")
        with c2:
            st.metric("P(Good)", f"{good_prob*100:.2f}%")
        with c3:
            st.metric("AE MSE", f"{mse:.6f}")


if __name__ == "__main__":
    main()
