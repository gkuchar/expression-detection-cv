from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import cv2
import cupy as cp
import numpy as np

try:
    from .model import EMOTION_LABELS, EmotionCNNGPU, ModelConfig
except ImportError:
    from model import EMOTION_LABELS, EmotionCNNGPU, ModelConfig


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
    augment: bool = False,
) -> Iterator[tuple[Array, Array]]:
    indices = rng.permutation(images.shape[0])
    for start in range(0, images.shape[0], batch_size):
        batch_indices = indices[start : start + batch_size]
        batch_images = images[batch_indices]
        if augment:
            batch_images = augment_batch(batch_images, rng)
        yield batch_images, labels[batch_indices]


def augment_batch(images: Array, rng: np.random.Generator) -> Array:
    """Apply light geometric and intensity augmentation to a batch."""
    augmented = np.empty_like(images)
    for index, image in enumerate(images):
        augmented[index, 0] = augment_image(image[0], rng)
    return augmented


def augment_image(image: Array, rng: np.random.Generator) -> Array:
    """Return a lightly perturbed 48x48 grayscale image in [-1, 1]."""
    height, width = image.shape
    center = (width / 2.0, height / 2.0)
    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.95, 1.05))
    translate_x = float(rng.uniform(-0.08 * width, 0.08 * width))
    translate_y = float(rng.uniform(-0.08 * height, 0.08 * height))

    transform = cv2.getRotationMatrix2D(center, angle, scale)
    transform[0, 2] += translate_x
    transform[1, 2] += translate_y

    augmented = cv2.warpAffine(
        image,
        transform,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    if rng.random() < 0.5:
        augmented = cv2.flip(augmented, 1)

    contrast = float(rng.uniform(0.9, 1.1))
    brightness = float(rng.uniform(-0.08, 0.08))
    augmented = augmented * contrast + brightness
    return np.clip(augmented, -1.0, 1.0).astype(np.float32)


def compute_class_weights(labels: Array, num_classes: int) -> Array:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    weights[nonzero] = labels.shape[0] / (num_classes * counts[nonzero])
    return weights


def stratified_split(
    images: Array,
    labels: Array,
    validation_fraction: float,
    rng: np.random.Generator,
) -> tuple[Array, Array, Array, Array]:
    """Split data into train and validation subsets while preserving class mix."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    train_indices: list[np.ndarray] = []
    val_indices: list[np.ndarray] = []

    for class_index in range(len(EMOTION_LABELS)):
        class_indices = np.flatnonzero(labels == class_index)
        if class_indices.size == 0:
            continue

        shuffled = rng.permutation(class_indices)
        val_count = int(round(class_indices.size * validation_fraction))
        val_count = max(1, val_count)
        if val_count >= class_indices.size:
            val_count = class_indices.size - 1

        val_indices.append(shuffled[:val_count])
        train_indices.append(shuffled[val_count:])

    if not train_indices or not val_indices:
        raise ValueError("Could not create non-empty train/validation splits.")

    train_idx = np.concatenate(train_indices)
    val_idx = np.concatenate(val_indices)
    train_idx = rng.permutation(train_idx)
    val_idx = rng.permutation(val_idx)
    return images[train_idx], labels[train_idx], images[val_idx], labels[val_idx]


def accuracy(model: EmotionCNNGPU, images: Array, labels: Array) -> float:
    predictions = model.predict_batch(images)
    return float(np.mean(predictions == labels))


def confusion_matrix(model: EmotionCNNGPU, images: Array, labels: Array) -> Array:
    predictions = model.predict_batch(images)
    matrix = np.zeros((len(EMOTION_LABELS), len(EMOTION_LABELS)), dtype=np.int64)
    for true_label, predicted_label in zip(labels, predictions):
        matrix[true_label, predicted_label] += 1
    return matrix


def print_evaluation(model: EmotionCNNGPU, images: Array, labels: Array, split_name: str) -> None:
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
    patience: int,
    min_delta: float,
    augment: bool,
    validation_fraction: float,
) -> None:
    start_time = perf_counter()
    config = ModelConfig(dropout_rate=dropout_rate, l2_lambda=l2_lambda)
    model = EmotionCNNGPU(config)
    rng = np.random.default_rng(model.config.random_seed)
    _, image_height, image_width = model.config.input_shape

    train_images, train_labels = load_dataset(train_dir, (image_height, image_width))
    test_images, test_labels = load_dataset(test_dir, (image_height, image_width))
    train_images, train_labels, val_images, val_labels = stratified_split(
        train_images,
        train_labels,
        validation_fraction,
        rng,
    )
    class_weights = compute_class_weights(train_labels, model.config.num_classes)
    print(
        f"split_sizes train={train_images.shape[0]} "
        f"val={val_images.shape[0]} "
        f"test={test_images.shape[0]}"
    )
    print(f"class_weights={class_weights}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = float("-inf")
    best_epoch = 0
    epochs_without_improvement = 0

    cp.cuda.Stream.null.synchronize()
    for epoch in range(1, epochs + 1):
        losses: list[float] = []
        for batch_images, batch_labels in batch_iterator(
            train_images,
            train_labels,
            batch_size,
            rng,
            augment=augment,
        ):
            loss = model.train_step(
                batch_images,
                batch_labels,
                learning_rate,
                class_weights,
                model.config.l2_lambda,
            )
            cp.cuda.Stream.null.synchronize()
            losses.append(loss)

        train_accuracy = accuracy(model, train_images, train_labels)
        validation_accuracy = accuracy(model, val_images, val_labels)
        cp.cuda.Stream.null.synchronize()
        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.4f} "
            f"train_acc={train_accuracy:.3f} "
            f"val_acc={validation_accuracy:.3f}"
        )

        if validation_accuracy > best_accuracy + min_delta:
            best_accuracy = validation_accuracy
            best_epoch = epoch
            epochs_without_improvement = 0
            model.save_parameters(str(output_path))
            print(
                f"best_checkpoint epoch={epoch} "
                f"val_acc={validation_accuracy:.3f} "
                f"saved_to={output_path}"
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"early_stopping epoch={epoch} "
                    f"best_epoch={best_epoch} "
                    f"best_val_acc={best_accuracy:.3f} "
                    f"patience={patience}"
                )
                break

    if best_epoch:
        model.load_parameters(str(output_path))
        cp.cuda.Stream.null.synchronize()
        print(f"loaded_best_checkpoint epoch={best_epoch} val_acc={best_accuracy:.3f}")

    print_evaluation(model, train_images, train_labels, "train")
    print_evaluation(model, val_images, val_labels, "val")
    print_evaluation(model, test_images, test_labels, "test")
    cp.cuda.Stream.null.synchronize()

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
    parser = argparse.ArgumentParser(description="Train the GPU CuPy emotion CNN.")
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
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument(
        "--disable-augmentation",
        action="store_true",
        help="Disable random training-time image augmentation.",
    )
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
        patience=args.patience,
        min_delta=args.min_delta,
        augment=not args.disable_augmentation,
        validation_fraction=args.validation_fraction,
    )


if __name__ == "__main__":
    main()