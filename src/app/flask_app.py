"""Flask prototype for CardioIA Vision inference."""

from __future__ import annotations

import uuid
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from werkzeug.utils import secure_filename

from src import config
from src.inference import load_model_from_checkpoint, predict_image


ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def create_app() -> Flask:
    app = Flask(__name__)
    app.secret_key = "cardioia-vision-dev-secret"
    config.ensure_project_directories()

    model = None
    model_metadata = None
    model_error = None

    try:
        model, model_metadata = load_model_from_checkpoint(device=config.DEVICE)
    except Exception as exc:  # noqa: BLE001 - show setup issue in prototype UI.
        model_error = str(exc)

    app.config["MODEL"] = model
    app.config["MODEL_METADATA"] = model_metadata
    app.config["MODEL_ERROR"] = model_error

    @app.route("/", methods=["GET"])
    def index():
        return render_template(
            "index.html",
            model_metadata=app.config["MODEL_METADATA"],
            model_error=app.config["MODEL_ERROR"],
        )

    @app.route("/predict", methods=["POST"])
    def predict():
        if app.config["MODEL"] is None:
            flash("Modelo final ainda nao carregado. Execute o treinamento e a comparacao final.")
            return redirect(url_for("index"))

        uploaded_file = request.files.get("image")
        if uploaded_file is None or uploaded_file.filename == "":
            flash("Envie uma imagem para classificacao.")
            return redirect(url_for("index"))

        extension = Path(uploaded_file.filename).suffix.lower()
        if extension not in ALLOWED_EXTENSIONS:
            flash("Formato invalido. Use PNG, JPG ou JPEG.")
            return redirect(url_for("index"))

        safe_name = secure_filename(uploaded_file.filename)
        unique_name = f"{uuid.uuid4().hex}_{safe_name}"
        output_path = config.UPLOADS_DIR / unique_name
        uploaded_file.save(output_path)

        prediction = predict_image(
            image_path=output_path,
            model=app.config["MODEL"],
            model_name=app.config["MODEL_METADATA"]["model_name"],
            device=config.DEVICE,
        )

        return render_template(
            "result.html",
            prediction=prediction,
            image_url=url_for("static", filename=f"uploads/{unique_name}"),
            model_metadata=app.config["MODEL_METADATA"],
        )

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
