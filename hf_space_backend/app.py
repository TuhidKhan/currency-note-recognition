import io
import json
import os
import numpy as np
from PIL import Image
import cv2
import spaces
import torch
import torch.nn as nn
from torchvision import models, transforms
from fastapi.middleware.cors import CORSMiddleware
import gradio as gr

# Calibrated Thresholds
MIN_CLASS_CONFIDENCE = 40.0
MIN_COUNTRY_CONFIDENCE = 65.0
MIN_MARGIN = 15.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("class_names.json", "r") as f:
    class_names = json.load(f)

# Load ResNet-50
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(num_ftrs, len(class_names))
)

model.load_state_dict(torch.load("model.pth", map_location=device, weights_only=False))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_banknote_crop(img_pil):
    """Isolates banknote foreground by detecting rectangular contours."""
    img_np = np.array(img_pil)
    if img_np.ndim == 2:
        gray = img_np
    else:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 130)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    largest_c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_c)

    img_h, img_w = gray.shape[:2]
    area_ratio = (w * h) / (img_w * img_h)

    if 0.15 <= area_ratio <= 0.96:
        pad_x = int(w * 0.04)
        pad_y = int(h * 0.04)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)
        return img_pil.crop((x1, y1, x2, y2))

    return None


def _eval_tensor(pil_img):
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0]
    best_p, _ = torch.max(probs, dim=0)
    return best_p.item(), probs


@spaces.GPU
def predict_core(image: Image.Image):
    image = image.convert("RGB")

    candidate_crops = [image]

    # Handle phone portrait captures
    if image.height > image.width:
        candidate_crops.append(image.rotate(90, expand=True))
        candidate_crops.append(image.rotate(270, expand=True))

    # OpenCV contour isolation
    note_crop = extract_banknote_crop(image)
    if note_crop is not None:
        candidate_crops.append(note_crop)
        if note_crop.height > note_crop.width:
            candidate_crops.append(note_crop.rotate(90, expand=True))
            candidate_crops.append(note_crop.rotate(270, expand=True))

    # Center crop fallback
    w, h = image.size
    center_crop = image.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))
    candidate_crops.append(center_crop)
    if center_crop.height > center_crop.width:
        candidate_crops.append(center_crop.rotate(90, expand=True))

    best_overall_prob = -1.0
    best_probs_tensor = None

    for cand in candidate_crops:
        p, probs_tensor = _eval_tensor(cand)
        if p > best_overall_prob:
            best_overall_prob = p
            best_probs_tensor = probs_tensor

    # Country Aggregation
    pkr_sum = 0.0
    afn_sum = 0.0
    for idx, cls in enumerate(class_names):
        val = best_probs_tensor[idx].item() * 100
        if cls.startswith("PKR"):
            pkr_sum += val
        elif cls.startswith("Afghani"):
            afn_sum += val

    top_country_prob = max(pkr_sum, afn_sum)

    top_probs, top_indices = torch.topk(best_probs_tensor, k=min(3, len(class_names)))
    best_idx = top_indices[0].item()
    best_prob = round(top_probs[0].item() * 100, 2)
    best_class = class_names[best_idx]
    currency, denom = best_class.split("_")

    # STAGE 1: Check if genuine banknote or random object (car, background)
    second_prob = round(top_probs[1].item() * 100, 2) if len(top_probs) > 1 else 0.0
    margin = best_prob - second_prob
    is_banknote = (best_prob >= MIN_CLASS_CONFIDENCE) and (top_country_prob >= MIN_COUNTRY_CONFIDENCE or margin >= MIN_MARGIN)

    top_preds = [
        {"class": class_names[idx.item()], "confidence": round(prob.item() * 100, 2)}
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "is_valid": is_banknote,
        "currency": currency if is_banknote else "Unknown",
        "denomination": denom if is_banknote else "Unknown",
        "class_name": best_class if is_banknote else "Not a Banknote",
        "confidence": best_prob,
        "threshold": MIN_CLASS_CONFIDENCE,
        "country_confidence": round(top_country_prob, 2),
        "top_predictions": top_preds
    }


# 2. Gradio Interface
@spaces.GPU
def gradio_predict(img):
    if img is None:
        return {}
    res = predict_core(Image.fromarray(img))

    # If rejected as non-currency object
    if not res["is_valid"]:
        return {
            f"Not a Banknote (Low Confidence: {res['confidence']}%)": 1.0
        }

    return {p["class"]: p["confidence"] / 100.0 for p in res["top_predictions"]}


demo = gr.Interface(
    fn=gradio_predict,
    inputs=gr.Image(type="numpy", label="Upload Banknote"),
    outputs=gr.Label(num_top_classes=3, label="Recognition Result"),
    title="Currency Note Recognition API"
)

# Enable CORS for Netlify
demo.app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
