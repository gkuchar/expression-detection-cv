from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

try:
    from .model import EMOTION_LABELS, EmotionCNN
except ImportError:
    from model import EMOTION_LABELS, EmotionCNN


Array = np.ndarray


def load_dataset(data_dir: Path) -> tuple[Array, Array]:
    """Load preprocessed images from emotion subfolders."""
    images: list[Array] = []
    labels: list[int] = []
    label_lookup = {label.lower(): index for index, label in enumerate(EMOTION_LABELS)}

    for emotion_dir in sorted(data_dir.iterdir()):
        if not emotion_dir.is_dir():
            continue

        label_name = emotion_dir.name.lower()
        label = label_lookup[label_name]
        for image_path in sorted(emotion_dir.glob("*.jpg")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image.shape != (128, 128):
                image = cv2.resize(image, (128, 128))

            image = image.astype(np.float32) / 255.0
            images.append(image.reshape(1, 128, 128))
            labels.append(label)

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


def accuracy(model: EmotionCNN, images: Array, labels: Array) -> float:
    logits = model.forward(images)
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))


def train(
    train_dir: Path,
    test_dir: Path,
    output_path: Path,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    model = EmotionCNN()
    rng = np.random.default_rng(model.config.random_seed)

    train_images, train_labels = load_dataset(train_dir)
    test_images, test_labels = load_dataset(test_dir)

    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch_images, batch_labels in batch_iterator(
            train_images,
            train_labels,
            batch_size,
            rng,
        ):
            loss = model.train_step(batch_images, batch_labels, learning_rate)
            losses.append(loss)

        train_accuracy = accuracy(model, train_images, train_labels)
        test_accuracy = accuracy(model, test_images, test_labels)
        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.4f} "
            f"train_acc={train_accuracy:.3f} "
            f"test_acc={test_accuracy:.3f}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_parameters(str(output_path))
    print(f"saved weights to {output_path}")


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
    args = parser.parse_args()

    train(
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        output_path=args.output_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )


if __name__ == "__main__":
    main()
