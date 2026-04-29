from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

try:
    from .model import EMOTION_LABELS, EmotionCNN, ModelConfig
except ImportError:
    from model import EMOTION_LABELS, EmotionCNN, ModelConfig


Array = np.ndarray


def load_dataset(data_dir: Path, image_size: tuple[int, int]) -> tuple[Array, Array]:
    """Load preprocessed images from emotion subfolders."""
    images: list[Array] = []
    labels: list[int] = []
    label_lookup = {label.lower(): index for index, label in enumerate(EMOTION_LABELS)}
    target_height, target_width = image_size

    for emotion_dir in sorted(data_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue

        label_name = emotion_dir.name.lower()
        if label_name not in label_lookup:
            continue

        label = label_lookup[label_name]
        for image_path in sorted(emotion_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue

            if image.shape != (target_height, target_width):
                image = cv2.resize(image, (target_width, target_height))

            image = image.astype(np.float32) / 127.5 - 1.0
            images.append(image.reshape(1, target_height, target_width))
            labels.append(label)

    if not images:
        raise ValueError(f"No images were loaded from {data_dir}")

    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64)


def batch_iterator(
    images: Array,
    labels: Array,
    batch_size: int,
    rng: np.random.Generator,
) -> Iterator[tuple[Array, Array]]:
    indices = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        yield images[batch_indices], labels[batch_indices]


def compute_class_weights(labels: Array, num_classes: int) -> Array:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = labels.shape[0] / (num_classes * counts[nonzero])
    return weights


def accuracy(model: EmotionCNN, images: Array, labels: Array) -> float:
    logits = model.forward(images)
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def confusion_matrix(model: EmotionCNN, images: Array, labels: Array) -> Array:
    logits = model.forward(images)
    predictions = np.argmax(logits, axis=1)
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions, strict=False):
        matrix[true_label, predicted_label] += 1
    return matrix


def print_evaluation(model: EmotionCNN, images: Array, labels: Array, split_name: str) -> None:
    matrix = confusion_matrix(model, images, labels)
    split_accuracy = float(np.trace(matrix) / np.sum(matrix))
    print(f"{split_name}_accuracy={split_accuracy:.3f}")
    print(f"{split_name}_confusion_matrix rows=true cols=pred")
    print(matrix)
    print(f"{split_name}_per_class_accuracy")
    for class_index, label in enumerate(EMOTION_LABELS):
        total = int(np.sum(matrix[class_index]))
        correct = int(matrix[class_index, class_index])
        class_accuracy = correct / total if total else 0.0
        print(f"{label}: {correct}/{total} = {class_accuracy:.3f}")


def train(
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    l2_lambda: float,
) -> None:
    start_time = perf_counter()
    config = ModelConfig(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    model = EmotionCNN(config)
    rng = np.random.default_rng(model.config.random_seed)
    _, image_height, image_width = model.config.input_shape

    train_images, train_labels = load_dataset(train_dir, (image_height, image_width))
    test_images, test_labels = load_dataset(test_dir, (image_height, image_width))
    class_weights = compute_class_weights(train_labels, model.config.num_classes)
    print(f"class_weights={class_weights}")

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch_images, batch_labels in batch_iterator(
            train_images,
            train_labels,
            batch_size,
            rng,
        ):
            loss = model.train_step(
                batch_images,
                batch_labels,
                learning_rate,
                class_weights,
                model.config.l2_lambda,
            )
            losses.append(loss)

        train_accuracy = accuracy(model, train_images, train_labels)
        test_accuracy = accuracy(model, test_images, test_labels)
        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.4f} "
            f"train_acc={train_accuracy:.3f} "
            f"test_acc={test_accuracy:.3f}"
        )

    print_evaluation(model, train_images, train_labels, "train")
    print_evaluation(model, test_images, test_labels, "test")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_parameters(str(output_path))
    print(f"saved weights to {output_path}")
    elapsed_seconds = perf_counter() - start_time
    print(f"total_training_time={format_duration(elapsed_seconds)}")

# Added timing profiler for comparing CPU/GPU training times
def format_duration(seconds: float) -> str:
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CPU NumPy emotion CNN.")
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--train-dir", type=Path, default=repo_root / "data" / "train")
    parser.add_argument("--test-dir", type=Path, default=repo_root / "data" / "test")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path(__file__).with_name("emotion_cnn_weights.npz"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--l2-lambda", type=float, default=1e-4)
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        dropout_rate=args.dropout_rate,
        l2_lambda=args.l2_lambda,
    )


if __name__ == "__main__":
    main()