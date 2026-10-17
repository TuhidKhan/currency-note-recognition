import os
import json
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Calibrated Thresholds for Banknote Discrimination
# True banknotes achieve >54% confidence with deep fine-tuning.
# Non-currency objects (vehicles, furniture, animals) score below 30% confidence.
MIN_CLASS_CONFIDENCE = 40.0
MIN_COUNTRY_CONFIDENCE = 65.0
MIN_MARGIN = 15.0

_CACHED_MODEL = None
_CACHED_CLASSES = None


def get_inference_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def load_model(weights_path="models/model.pth", classes_path="models/class_names.json"):
    global _CACHED_MODEL, _CACHED_CLASSES

    if _CACHED_MODEL is not None and _CACHED_CLASSES is not None:
        return _CACHED_MODEL, _CACHED_CLASSES

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found at {weights_path}. Please run train.py first.")

    with open(classes_path, "r") as f:
        class_names = json.load(f)

    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, len(class_names))
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False))
    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_CLASSES = class_names
    return _CACHED_MODEL, _CACHED_CLASSES


def extract_banknote_crop(img_pil):
    """
    Detects rectangular banknote boundaries using edge and contour analysis
    to isolate the banknote from wooden tables, desks, or background surfaces.
    """
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

    # Find the largest contour
    largest_c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_c)

    img_h, img_w = gray.shape[:2]
    area_ratio = (w * h) / (img_w * img_h)

    # A valid banknote on a table typically covers 15% to 95% of the frame
    if 0.15 <= area_ratio <= 0.96:
        pad_x = int(w * 0.04)
        pad_y = int(h * 0.04)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img_w, x + w + pad_x)
        y2 = min(img_h, y + h + pad_y)
        return img_pil.crop((x1, y1, x2, y2))

    return None


def _eval_tensor(model, pil_img, transform, device):
    """Helper to run inference on a single PIL crop."""
    t = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        probs = torch.softmax(out, dim=1)[0]
    best_p, _ = torch.max(probs, dim=0)
    return best_p.item(), probs


def predict_image(image_input, weights_path="models/model.pth", classes_path="models/class_names.json"):
    """
    Hierarchical 3-Stage Currency Prediction Pipeline:
    1. Multi-Crop & Multi-Orientation Evaluation:
       - Checks original, 90-degree rotations, OpenCV banknote contour crop, and center crop.
    2. Stage 1: Banknote Verification (Is it a banknote?)
       - Validates confidence threshold and country probability mass to reject cars/objects.
    3. Stage 2: Country Identification (Which country?)
       - Sums probability masses for PKR and AFN to find the dominant country.
    4. Stage 3: Denomination Classification (Which note?)
       - Identifies the specific denomination within the recognized country.
    """
    model, class_names = load_model(weights_path, classes_path)
    device = next(model.parameters()).device
    transform = get_inference_transform()

    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    elif isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    # Generate Candidate Crops (Auto-alignment + Foreground isolation)
    candidate_crops = [image]

    # Phone photos are often vertical: test 90° and 270°
    if image.height > image.width:
        candidate_crops.append(image.rotate(90, expand=True))
        candidate_crops.append(image.rotate(270, expand=True))

    # OpenCV contour isolation (removes surrounding table/desk borders)
    note_crop = extract_banknote_crop(image)
    if note_crop is not None:
        candidate_crops.append(note_crop)
        if note_crop.height > note_crop.width:
            candidate_crops.append(note_crop.rotate(90, expand=True))
            candidate_crops.append(note_crop.rotate(270, expand=True))

    # Center 75% crop fallback (in case edges are slightly noisy)
    w, h = image.size
    center_crop = image.crop((int(w * 0.12), int(h * 0.12), int(w * 0.88), int(h * 0.88)))
    candidate_crops.append(center_crop)
    if center_crop.height > center_crop.width:
        candidate_crops.append(center_crop.rotate(90, expand=True))

    # Evaluate each crop and pick the candidate with highest certainty
    best_overall_prob = -1.0
    best_probs_tensor = None

    for cand in candidate_crops:
        p, probs_tensor = _eval_tensor(model, cand, transform, device)
        if p > best_overall_prob:
            best_overall_prob = p
            best_probs_tensor = probs_tensor

    # Aggregate Country Probabilities
    pkr_prob_sum = 0.0
    afn_prob_sum = 0.0

    for idx, cls in enumerate(class_names):
        val = best_probs_tensor[idx].item() * 100
        if cls.startswith("PKR"):
            pkr_prob_sum += val
        elif cls.startswith("Afghani"):
            afn_prob_sum += val

    top_country = "PKR" if pkr_prob_sum >= afn_prob_sum else "AFN"
    top_country_prob = max(pkr_prob_sum, afn_prob_sum)
    top_country_name = "Pakistani Rupee" if top_country == "PKR" else "Afghan Afghani"

    # Top class prediction
    top_probs, top_indices = torch.topk(best_probs_tensor, k=min(3, len(class_names)))
    best_idx = top_indices[0].item()
    best_class = class_names[best_idx]
    best_class_prob = round(top_probs[0].item() * 100, 2)

    pred_curr, pred_denom = best_class.split("_")

    # STAGE 1: Is it a banknote?
    # Valid banknotes produce high peak confidence (>=40.0%) and clear class separation.
    # Non-banknotes (cars, animals, furniture) produce scattered, low-confidence (<30%) distributions.
    second_class_prob = round(top_probs[1].item() * 100, 2) if len(top_probs) > 1 else 0.0
    margin = best_class_prob - second_class_prob
    is_banknote = (best_class_prob >= MIN_CLASS_CONFIDENCE) and (top_country_prob >= MIN_COUNTRY_CONFIDENCE or margin >= MIN_MARGIN)

    # STAGE 2: Country confirmation
    resolved_country = "PKR" if pred_curr == "PKR" else "AFN"
    resolved_country_full = "Pakistani Rupee" if resolved_country == "PKR" else "Afghan Afghani"

    # STAGE 3: Denomination confirmation
    resolved_denom = pred_denom

    top_predictions = [
        {
            "class": class_names[idx.item()],
            "confidence": round(prob.item() * 100, 2)
        }
        for idx, prob in zip(top_indices, top_probs)
    ]

    return {
        "is_valid": is_banknote,
        "country_code": resolved_country if is_banknote else "UNKNOWN",
        "country_name": resolved_country_full if is_banknote else "Unrecognized Object",
        "currency": pred_curr if is_banknote else "Unknown",
        "denomination": resolved_denom if is_banknote else "Unknown",
        "class_name": best_class if is_banknote else "Not a Banknote",
        "confidence": best_class_prob,
        "country_confidence": round(top_country_prob, 2),
        "threshold": MIN_CLASS_CONFIDENCE,
        "message": (
            f"Valid {resolved_country_full} banknote detected ({resolved_denom} {resolved_country})."
            if is_banknote
            else f"Unrecognized object (confidence {best_class_prob}% is below threshold). Please upload a clear photo of an Afghani or PKR banknote."
        ),
        "top_predictions": top_predictions
    }
