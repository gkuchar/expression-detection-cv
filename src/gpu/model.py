from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Run this on an NVIDIA CUDA machine, such as Delta. Apple Silicon Macs cannot
# execute CuPy CUDA kernels locally.
import cupy as cp
import numpy as np

try:
    from .kernels import (
        argmax_forward,
        conv2d_same_backward,
        conv2d_same_forward,
        cross_entropy_loss,
        dense_backward,
        dense_forward,
        dropout_backward,
        dropout_forward,
        max_pool2d_backward,
        max_pool2d_forward,
        relu_backward,
        relu_forward,
        sgd_update,
        softmax_cross_entropy_gradient,
        sum_squares,
    )
except ImportError:
    from kernels import (
        argmax_forward,
        conv2d_same_backward,
        conv2d_same_forward,
        cross_entropy_loss,
        dense_backward,
        dense_forward,
        dropout_backward,
        dropout_forward,
        max_pool2d_backward,
        max_pool2d_forward,
        relu_backward,
        relu_forward,
        sgd_update,
        softmax_cross_entropy_gradient,
        sum_squares,
    )


Array = cp.ndarray

EMOTION_LABELS: tuple[str, ...] = (
    "happy",
    "neutral",
    "sad",
)


@dataclass
class ModelConfig:
    """GPU CNN shape/config values matching the CPU model."""

    input_shape: tuple[int, int, int] = (1, 48, 48)
    num_classes: int = len(EMOTION_LABELS)
    conv1_filters: int = 16
    conv1_kernel: int = 3
    conv2_filters: int = 32
    conv2_kernel: int = 3
    hidden_units: int = 64
    random_seed: int = 42
    pool_size: int = 2
    dropout_rate: float = 0.5
    l2_lambda: float = 5e-4


class EmotionCNNGPU:
    """CuPy/CUDA CNN for emotion classification of a 48x48 grayscale image."""

    def __init__(
        self,
        config: ModelConfig | None = None,
        weights_path: str | Path | None = None,
    ) -> None:
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.gpu_rng = cp.random.default_rng(self.config.random_seed)
        self.params: dict[str, Array] = {}
        self.grads: dict[str, Array] = {}
        self.cache: dict[str, Array] = {}

        self._initialize_parameters()
        if weights_path is not None:
            self.load_parameters(weights_path)

    def _initialize_parameters(self) -> None:
        """Create convolution and dense layer weights on the GPU."""
        channels, height, width = self.config.input_shape
        conv1_filters = self.config.conv1_filters
        conv1_kernel = self.config.conv1_kernel
        conv2_filters = self.config.conv2_filters
        conv2_kernel = self.config.conv2_kernel
        hidden_units = self.config.hidden_units
        num_classes = self.config.num_classes

        self.params["W_conv1"] = self._he_initialize(
            size=(conv1_filters, channels, conv1_kernel, conv1_kernel),
            fan_in=channels * conv1_kernel * conv1_kernel,
        )
        self.params["b_conv1"] = cp.zeros(conv1_filters, dtype=cp.float32)

        self.params["W_conv2"] = self._he_initialize(
            size=(conv2_filters, conv1_filters, conv2_kernel, conv2_kernel),
            fan_in=conv1_filters * conv2_kernel * conv2_kernel,
        )
        self.params["b_conv2"] = cp.zeros(conv2_filters, dtype=cp.float32)

        pooled_height = height // (self.config.pool_size * self.config.pool_size)
        pooled_width = width // (self.config.pool_size * self.config.pool_size)
        flattened_units = conv2_filters * pooled_height * pooled_width

        self.params["W_fc1"] = self._he_initialize(
            size=(flattened_units, hidden_units),
            fan_in=flattened_units,
        )
        self.params["b_fc1"] = cp.zeros(hidden_units, dtype=cp.float32)

        self.params["W_fc2"] = self._he_initialize(
            size=(hidden_units, num_classes),
            fan_in=hidden_units,
        )
        self.params["b_fc2"] = cp.zeros(num_classes, dtype=cp.float32)

    def forward(self, x: Array | np.ndarray, training: bool = False) -> Array:
        """Run the forward pass and return raw logits."""
        x = self._prepare_batch(x)

        conv1 = conv2d_same_forward(x, self.params["W_conv1"], self.params["b_conv1"])
        relu1 = relu_forward(conv1)
        pool1 = max_pool2d_forward(relu1, self.config.pool_size)

        conv2 = conv2d_same_forward(pool1, self.params["W_conv2"], self.params["b_conv2"])
        relu2 = relu_forward(conv2)
        pool2 = max_pool2d_forward(relu2, self.config.pool_size)

        flattened = pool2.reshape(pool2.shape[0], -1)
        fc1_linear = dense_forward(flattened, self.params["W_fc1"], self.params["b_fc1"])
        hidden = relu_forward(fc1_linear)
        hidden_dropout, dropout_mask = self._dropout(hidden, self.config.dropout_rate, training)
        logits = dense_forward(hidden_dropout, self.params["W_fc2"], self.params["b_fc2"])

        self.cache = {
            "x": x,
            "conv1": conv1,
            "relu1": relu1,
            "pool1": pool1,
            "conv2": conv2,
            "relu2": relu2,
            "pool2": pool2,
            "flattened": flattened,
            "fc1_linear": fc1_linear,
            "hidden": hidden,
            "hidden_dropout": hidden_dropout,
            "dropout_mask": dropout_mask,
            "logits": logits,
        }
        return logits

    def cross_entropy_loss(
        self,
        logits: Array | np.ndarray,
        labels: Array | np.ndarray,
        class_weights: Array | np.ndarray | None = None,
        l2_lambda: float = 0.0,
    ) -> float:
        """Return weighted average cross-entropy plus optional L2 loss."""
        labels = self._prepare_labels(labels, logits.shape[0])
        prepared_class_weights = self._prepare_class_weights(class_weights)
        data_loss = cross_entropy_loss(logits, labels, prepared_class_weights)
        total_loss = float(cp.asnumpy(data_loss)[0])

        if l2_lambda:
            l2_sum = 0.0
            for name in ("W_conv1", "W_conv2", "W_fc1", "W_fc2"):
                l2_sum += float(cp.asnumpy(sum_squares(self.params[name]))[0])
            total_loss += 0.5 * l2_lambda * l2_sum

        return total_loss

    def backward(
        self,
        labels: Array | np.ndarray,
        class_weights: Array | np.ndarray | None = None,
    ) -> Array:
        """Compute gradients for all trainable parameters after forward()."""
        logits = self.cache["logits"]
        hidden_dropout = self.cache["hidden_dropout"]
        dropout_mask = self.cache["dropout_mask"]
        fc1_linear = self.cache["fc1_linear"]
        flattened = self.cache["flattened"]
        pool2 = self.cache["pool2"]
        relu2 = self.cache["relu2"]
        conv2 = self.cache["conv2"]
        pool1 = self.cache["pool1"]
        relu1 = self.cache["relu1"]
        conv1 = self.cache["conv1"]
        x = self.cache["x"]

        labels = self._prepare_labels(labels, logits.shape[0])
        prepared_class_weights = self._prepare_class_weights(class_weights)
        d_logits = softmax_cross_entropy_gradient(logits, labels, prepared_class_weights)

        d_hidden_dropout, self.grads["W_fc2"], self.grads["b_fc2"] = dense_backward(
            d_logits,
            hidden_dropout,
            self.params["W_fc2"],
        )
        d_hidden = dropout_backward(d_hidden_dropout, dropout_mask)
        d_fc1_linear = relu_backward(d_hidden, fc1_linear)

        d_flattened, self.grads["W_fc1"], self.grads["b_fc1"] = dense_backward(
            d_fc1_linear,
            flattened,
            self.params["W_fc1"],
        )

        d_pool2 = d_flattened.reshape(pool2.shape)
        d_relu2 = max_pool2d_backward(d_pool2, relu2, self.config.pool_size)
        d_conv2 = relu_backward(d_relu2, conv2)

        d_pool1, self.grads["W_conv2"], self.grads["b_conv2"] = conv2d_same_backward(
            d_conv2,
            pool1,
            self.params["W_conv2"],
        )
        d_relu1 = max_pool2d_backward(d_pool1, relu1, self.config.pool_size)
        d_conv1 = relu_backward(d_relu1, conv1)

        d_x, self.grads["W_conv1"], self.grads["b_conv1"] = conv2d_same_backward(
            d_conv1,
            x,
            self.params["W_conv1"],
        )

        self.cache["d_input"] = d_x
        return d_x

    def update_parameters(self, learning_rate: float, l2_lambda: float = 0.0) -> None:
        """Apply SGD updates, with L2 regularization on weight tensors only."""
        for name, parameter in self.params.items():
            parameter_l2 = l2_lambda if name.startswith("W_") else 0.0
            sgd_update(parameter, self.grads[name], learning_rate, parameter_l2)

    def train_step(
        self,
        x: Array | np.ndarray,
        labels: Array | np.ndarray,
        learning_rate: float,
        class_weights: Array | np.ndarray | None = None,
        l2_lambda: float = 0.0,
    ) -> float:
        """Run one forward/backward/update step and return the batch loss."""
        logits = self.forward(x, training=True)
        loss = self.cross_entropy_loss(logits, labels, class_weights, l2_lambda)
        self.backward(labels, class_weights)
        self.update_parameters(learning_rate, l2_lambda)
        return loss

    def predict(self, img: Array | np.ndarray) -> int:
        """Return the emotion class index for one preprocessed image."""
        batch = self._prepare_single_image(img)
        logits = self.forward(batch, training=False)
        prediction = argmax_forward(logits)
        return int(cp.asnumpy(prediction)[0])

    def predict_batch(self, x: Array | np.ndarray) -> np.ndarray:
        """Return emotion class indexes for a batch of preprocessed images."""
        logits = self.forward(x, training=False)
        predictions = argmax_forward(logits)
        return cp.asnumpy(predictions)

    def save_parameters(self, path: str | Path) -> None:
        """Save trained GPU parameters as a NumPy .npz file."""
        cpu_params = {name: cp.asnumpy(value) for name, value in self.params.items()}
        np.savez(path, **cpu_params)

    def load_parameters(self, path: str | Path) -> None:
        """Load trained parameters saved as a NumPy .npz file."""
        weights = np.load(path)
        for name in self.params:
            self.params[name] = cp.asarray(weights[name], dtype=cp.float32)

    def _he_initialize(self, size: tuple[int, ...], fan_in: int) -> Array:
        scale = np.sqrt(2.0 / fan_in)
        values = self.rng.normal(loc=0.0, scale=scale, size=size).astype(np.float32)
        return cp.asarray(values)

    def _dropout(
        self,
        x: Array,
        dropout_rate: float,
        training: bool,
    ) -> tuple[Array, Array]:
        if not training or dropout_rate <= 0.0:
            mask = cp.ones_like(x, dtype=cp.float32)
            return x, mask

        random_values = self.gpu_rng.random(x.shape, dtype=cp.float32)
        return dropout_forward(x, random_values, dropout_rate)

    def _prepare_single_image(self, img: Array | np.ndarray) -> Array:
        img = cp.asarray(img, dtype=cp.float32)
        return img.reshape(1, *self.config.input_shape)

    def _prepare_batch(self, x: Array | np.ndarray) -> Array:
        x = cp.asarray(x, dtype=cp.float32)
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        return x

    def _prepare_labels(self, labels: Array | np.ndarray, batch_size: int) -> Array:
        return cp.asarray(labels, dtype=cp.int32).reshape(-1)

    def _prepare_class_weights(self, class_weights: Array | np.ndarray | None) -> Array | None:
        if class_weights is None:
            return None
        return cp.asarray(class_weights, dtype=cp.float32)


_MODEL = EmotionCNNGPU()


def predict_emotion_gpu(img: Array | np.ndarray) -> int:
    """Frontend entry point: predict an emotion from a 48x48 float image."""
    return _MODEL.predict(img)