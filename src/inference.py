from __future__ import annotations

import os
from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import torch
from torchvision import transforms, models
from ultralytics import YOLO

try:
    from .utils import load_config  # when imported as part of package `src`
    from .geo import extract_gps_from_image
except Exception:
    from utils import load_config  # fallback for direct script execution
    from geo import extract_gps_from_image


def load_best_model(cfg: dict) -> YOLO:
    best_model_path_file = os.path.join(cfg["paths"]["models_dir"], "best_model.path")
    if os.path.isfile(best_model_path_file):
        with open(best_model_path_file, "r", encoding="utf-8") as f:
            best_model_path = f.read().strip()
    else:
        # Fallback to configured model (for zero-shot usage)
        best_model_path = cfg["training"]["model"]
    return YOLO(best_model_path)


# Global variable to cache the classifier model
_classifier_model = None
_classifier_device = None


def load_classifier_model(cfg: dict):
    """Load the trained plastic/non-plastic classifier model if available."""
    global _classifier_model, _classifier_device
    
    if _classifier_model is not None:
        return _classifier_model, _classifier_device
    
    classifier_cfg = cfg.get("classifier", {})
    classifier_path = classifier_cfg.get("model_path", "models/classifier/plastic_classifier_best.pt")
    
    if not os.path.isfile(classifier_path):
        return None, None
    
    try:
        # Setup device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _classifier_device = device
        
        # Create model architecture
        model = models.resnet18(pretrained=False)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 2)
        
        # Load weights
        checkpoint = torch.load(classifier_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model = model.to(device)
        
        _classifier_model = model
        print(f"Loaded classifier model from {classifier_path}")
        return model, device
    except Exception as e:
        print(f"Warning: Could not load classifier model: {e}")
        return None, None


def classify_debris_type_with_model(image_path: str, bbox: List[float], model, device) -> Tuple[str, float]:
    """Classify using trained model."""
    try:
        # Load image and extract region
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "non-plastic", 0.5
        
        region = img_array[y1:y2, x1:x2]
        if region.size == 0:
            return "non-plastic", 0.5
        
        # Transform for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        region_img = Image.fromarray(region)
        tensor = transform(region_img).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()
        
        label = "plastic" if predicted_class == 1 else "non-plastic"
        return label, float(confidence)
    except Exception as e:
        return "non-plastic", 0.5


def classify_debris_type(image_path: str, bbox: List[float], cfg: Optional[dict] = None) -> Tuple[str, float]:
    """
    Classify detected debris into 'plastic' or 'non-plastic' using advanced heuristics.
    
    Args:
        image_path: Path to the image
        bbox: Bounding box coordinates [x1, y1, x2, y2]
    
    Returns:
        Tuple of (classification_label, confidence)
    """
    try:
        # Load image and extract the bounding box region
        img = Image.open(image_path).convert("RGB")
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # Extract bounding box region with padding for context
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        # Add small padding to get more context
        pad = 5
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        if x2 <= x1 or y2 <= y1:
            return "non-plastic", 0.5
        
        region = img_array[y1:y2, x1:x2]
        
        if region.size == 0 or region.shape[0] < 10 or region.shape[1] < 10:
            return "non-plastic", 0.5
        
        # Convert to different color spaces for comprehensive analysis
        region_img = Image.fromarray(region)
        hsv_region = np.array(region_img.convert("HSV"))
        lab_region = np.array(region_img.convert("LAB"))
        
        # Extract RGB channels
        r_channel = region[:, :, 0].astype(np.float32)
        g_channel = region[:, :, 1].astype(np.float32)
        b_channel = region[:, :, 2].astype(np.float32)
        
        # Calculate comprehensive features
        mean_hue = np.mean(hsv_region[:, :, 0])
        mean_saturation = np.mean(hsv_region[:, :, 1])
        mean_value = np.mean(hsv_region[:, :, 2])
        
        # Color statistics
        mean_r, mean_g, mean_b = np.mean(r_channel), np.mean(g_channel), np.mean(b_channel)
        std_r, std_g, std_b = np.std(r_channel), np.std(g_channel), np.std(b_channel)
        
        # Color variance (lower = more uniform, common in plastics)
        color_variance = np.var(region.reshape(-1, 3), axis=0).mean()
        
        # Brightness/lightness
        brightness = mean_value
        lightness = np.mean(lab_region[:, :, 0])  # L channel in LAB
        
        # Edge detection for texture analysis (plastics often have smoother surfaces)
        from scipy import ndimage
        gray = np.mean(region, axis=2)
        edges = ndimage.sobel(gray)
        edge_density = np.mean(np.abs(edges))
        
        # Calculate color distribution characteristics
        # Plastics often have specific color characteristics
        color_uniformity = 1.0 / (1.0 + color_variance / 1000.0)  # Normalized
        
        # Plastic score calculation with improved heuristics
        plastic_score = 0.0
        
        # 1. Brightness/Reflectivity (plastics are often shiny/reflective)
        if brightness > 120:  # Lowered threshold
            plastic_score += 0.25
        if brightness > 160:  # Very bright
            plastic_score += 0.15
        if lightness > 60:  # High lightness in LAB space
            plastic_score += 0.1
        
        # 2. Color Uniformity (plastics often have uniform colors)
        if color_variance < 3000:  # More lenient threshold
            plastic_score += 0.2
        if color_variance < 1500:  # Very uniform
            plastic_score += 0.15
        if std_r < 30 and std_g < 30 and std_b < 30:  # Low standard deviation
            plastic_score += 0.15
        
        # 3. Hue Analysis (common plastic colors)
        # Plastics come in many colors: blue, green, red, yellow, white, clear
        hue = mean_hue
        # Blue plastics (common for bottles)
        if 100 <= hue <= 130:
            plastic_score += 0.2
        # Green plastics
        elif 50 <= hue <= 80:
            plastic_score += 0.15
        # Red/Pink plastics
        elif (hue <= 10 or hue >= 170):
            plastic_score += 0.15
        # Yellow/Orange plastics
        elif 15 <= hue <= 40:
            plastic_score += 0.1
        # White/Clear plastics (low saturation, high brightness)
        elif mean_saturation < 50 and brightness > 140:
            plastic_score += 0.25
        
        # 4. Saturation Analysis (many plastics are vibrant or clear)
        if mean_saturation < 40:  # Very low saturation (clear/white plastics)
            plastic_score += 0.15
        elif mean_saturation > 150:  # High saturation (colored plastics)
            plastic_score += 0.1
        
        # 5. Texture Analysis (plastics often have smoother surfaces)
        if edge_density < 15:  # Smooth surface
            plastic_score += 0.15
        elif edge_density < 25:  # Relatively smooth
            plastic_score += 0.1
        
        # 6. Color Balance (plastics often have balanced RGB)
        rgb_balance = np.std([mean_r, mean_g, mean_b])
        if rgb_balance < 40:  # Balanced colors (common in plastics)
            plastic_score += 0.1
        
        # 7. Specific plastic color patterns
        # Bright white/clear (common in plastic bags, bottles)
        if mean_r > 200 and mean_g > 200 and mean_b > 200 and color_variance < 2000:
            plastic_score += 0.2
        
        # Bright blue (common in water bottles)
        if 100 <= mean_hue <= 130 and mean_b > mean_r and mean_b > mean_g and brightness > 120:
            plastic_score += 0.2
        
        # Bright red/pink (common in plastic covers)
        if (mean_hue <= 10 or mean_hue >= 170) and mean_r > 150 and brightness > 100:
            plastic_score += 0.15
        
        # 8. Overall characteristics that strongly suggest plastic
        # Very uniform and bright
        if color_variance < 1000 and brightness > 150:
            plastic_score += 0.2
        
        # Normalize and classify
        # Lower threshold to be more inclusive of plastics
        plastic_score = min(1.0, plastic_score)  # Cap at 1.0
        
        if plastic_score >= 0.35:  # Lowered threshold from 0.5
            # Calculate confidence based on score strength
            if plastic_score >= 0.7:
                confidence = 0.85 + (plastic_score - 0.7) * 0.5  # 0.85-0.95
            elif plastic_score >= 0.5:
                confidence = 0.75 + (plastic_score - 0.5) * 0.5  # 0.75-0.85
            else:
                confidence = 0.65 + (plastic_score - 0.35) * 0.67  # 0.65-0.75
            return "plastic", min(0.95, confidence)
        else:
            # Non-plastic confidence
            non_plastic_score = 1.0 - plastic_score
            if non_plastic_score >= 0.7:
                confidence = 0.80 + (non_plastic_score - 0.7) * 0.5
            else:
                confidence = 0.65 + (non_plastic_score - 0.5) * 0.75
            return "non-plastic", min(0.95, confidence)
            
    except Exception as e:
        # On error, default to non-plastic with low confidence
        return "non-plastic", 0.5


def predict_on_images(image_paths: List[str], conf: Optional[float] = None):
    cfg = load_config("configs/config.yaml")
    model = load_best_model(cfg)
    conf_thr = conf if conf is not None else float(cfg["inference"]["conf"])  # type: ignore
    iou_thr = float(cfg["inference"]["iou"])  # type: ignore
    
    # Try to load classifier model
    classifier_model, classifier_device = load_classifier_model(cfg)
    use_classifier = classifier_model is not None
    
    results = model.predict(
        image_paths,
        imgsz=int(cfg["training"]["imgsz"]),
        conf=conf_thr,
        iou=iou_thr,
        verbose=False,
    )

    outputs = []
    for img_path, res in zip(image_paths, results):
        # Extract GPS if present
        gps = extract_gps_from_image(img_path)
        if gps:
            print(f"✅ GPS extracted from {os.path.basename(img_path)}: {gps}")
        else:
            print(f"⚠️ No GPS in {os.path.basename(img_path)}")
        boxes = []
        names = res.names if hasattr(res, "names") else {}
        for b in res.boxes:  # type: ignore[attr-defined]
            cls_id = int(b.cls[0].item())
            bbox = b.xyxy[0].tolist()  # [x1,y1,x2,y2]
            
            # Classify debris into plastic or non-plastic
            if use_classifier:
                # Use trained classifier model
                debris_type, type_confidence = classify_debris_type_with_model(
                    img_path, bbox, classifier_model, classifier_device
                )
            else:
                # Fall back to heuristic-based classification
                debris_type, type_confidence = classify_debris_type(img_path, bbox, cfg)
            
            boxes.append({
                "xyxy": bbox,
                "conf": float(b.conf[0].item()),  # Detection confidence
                "type_conf": type_confidence,  # Classification confidence
                "cls": cls_id,
                "label": debris_type,  # "plastic" or "non-plastic"
                "original_label": names.get(cls_id, str(cls_id)),  # Keep original for reference
            })
        outputs.append({
            "image_path": img_path,
            "gps": gps,  # (lat, lon) or None
            "boxes": boxes,
        })
    return outputs


if __name__ == "__main__":
    # Example
    test_images = [
        # Add paths here for manual testing
    ]
    preds = predict_on_images(test_images)
    print(preds)


