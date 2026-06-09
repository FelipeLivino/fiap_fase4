from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import torch
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from torch import nn
from torchvision import models, transforms


BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "notebooks" / "artifacts_cardioia_pytorch_multilabel"
MODEL_PATH = ARTIFACT_DIR / "modelos" / "vision_transformer_vit.pt"
FINAL_METRICS_PATH = ARTIFACT_DIR / "tabelas" / "final_model_selection.csv"
CLASS_METRICS_PATH = ARTIFACT_DIR / "tabelas" / "metricas_por_classe_vision_transformer_vit.csv"

SELECTED_PATHOLOGIES = ["Infiltration", "Effusion", "Atelectasis", "Pneumothorax"]
IMAGE_SIZE = 224
THRESHOLD = float(os.getenv("CARDIOIA_THRESHOLD", "0.5"))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)
CORS(app)

_model: nn.Module | None = None


def build_vit_model() -> nn.Module:
    model = models.vit_b_16(weights=None)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, len(SELECTED_PATHOLOGIES))
    return model


def load_model() -> nn.Module:
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Checkpoint nao encontrado: {MODEL_PATH}")

    model = build_vit_model()
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    _model = model
    return _model


def preprocess_image(image_path: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)


def read_csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pd.read_csv(path).fillna("").to_dict(orient="records")


@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(DEVICE),
            "model_path": str(MODEL_PATH),
            "model_exists": MODEL_PATH.exists(),
            "labels": SELECTED_PATHOLOGIES,
            "threshold": THRESHOLD,
        }
    )


@app.get("/metrics")
def metrics():
    final_metrics = read_csv_records(FINAL_METRICS_PATH)
    class_metrics = read_csv_records(CLASS_METRICS_PATH)
    return jsonify(
        {
            "final_model": final_metrics[0] if final_metrics else None,
            "class_metrics": class_metrics,
        }
    )


@app.post("/predict")
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Envie a imagem no campo multipart chamado 'image'."}), 400

    uploaded = request.files["image"]
    suffix = Path(uploaded.filename or "image.png").suffix or ".png"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = Path(tmp.name)
        uploaded.save(tmp)

    try:
        model = load_model()
        tensor = preprocess_image(temp_path).to(DEVICE)
        with torch.no_grad():
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()[0]

        predictions = []
        for label, probability in zip(SELECTED_PATHOLOGIES, probabilities):
            predictions.append(
                {
                    "label": label,
                    "probability": round(float(probability), 6),
                    "detected": bool(probability >= THRESHOLD),
                }
            )

        detected = [item["label"] for item in predictions if item["detected"]]
        return jsonify(
            {
                "model": "Vision Transformer ViT",
                "threshold": THRESHOLD,
                "device": str(DEVICE),
                "detected_labels": detected,
                "predictions": predictions,
                "disclaimer": "Prototipo academico. Nao utilizar como diagnostico medico.",
            }
        )
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=debug)
