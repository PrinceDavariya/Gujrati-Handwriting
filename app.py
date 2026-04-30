import streamlit as st
import torch
import torch.nn as nn
import json
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms
from streamlit_drawable_canvas import st_canvas

# ── Model Definition ──────────────────────────────────────
class GujaratiCNN(nn.Module):
    def __init__(self, num_classes):
        super(GujaratiCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load Model ────────────────────────────────────────────
@st.cache_resource
def load_model():
    classes = json.load(open('classes.json', encoding='utf-8'))
    model = GujaratiCNN(num_classes=len(classes))
    model.load_state_dict(torch.load('gujarati_model.pth', map_location='cpu'))
    model.eval()
    return model, classes

model, classes = load_model()

transform = transforms.Compose([
    transforms.Grayscale(1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ── UI ────────────────────────────────────────────────────
st.set_page_config(page_title="ગુજરાતી OCR", page_icon="✍️")
st.title("✍️ ગુજરાતી હસ્તલેખન ઓળખ")
st.write("Draw a Gujarati character below and click **Recognize**")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Draw here:")
    canvas = st_canvas(
        fill_color="white",
        stroke_width=10,
        stroke_color="black",
        background_color="white",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas"
    )

with col2:
    st.subheader("Result:")
    if st.button("🔍 Recognize", use_container_width=True):
        if canvas.image_data is not None:
            img = Image.fromarray(canvas.image_data.astype(np.uint8))
            img = ImageOps.invert(img.convert("RGB"))
            tensor = transform(img).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                probs  = torch.softmax(output, dim=1)[0]
                top5   = torch.topk(probs, 5)

            top_char = classes[top5.indices[0].item()]
            top_prob = top5.values[0].item() * 100

            st.metric("Recognized Character", top_char)
            st.metric("Confidence", f"{top_prob:.1f}%")

            st.write("**Top 5 Predictions:**")
            for i in range(5):
                idx  = top5.indices[i].item()
                prob = top5.values[i].item() * 100
                st.progress(int(prob), text=f"{classes[idx]} — {prob:.1f}%")

    if st.button("🗑️ Clear", use_container_width=True):
        st.rerun()

st.divider()
st.caption(f"Trained on {len(classes)} Gujarati character classes · 53,043 images · 30 epochs · Val Accuracy ~100%")