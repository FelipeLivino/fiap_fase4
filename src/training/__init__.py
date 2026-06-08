"""Training and evaluation utilities for CardioIA Vision."""

from src.training.evaluate import evaluate_model
from src.training.train import TrainConfig, train_model

__all__ = ["TrainConfig", "evaluate_model", "train_model"]
