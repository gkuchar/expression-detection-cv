from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from pathlib import Path

_WEIGHTS_PATH = Path(__file__).with_name("emotion_cnn_weights.npz")


Array = np.ndarray

EMOTION_LABELS: tuple[str, ...] = (
    "happy",
    "neutral",
    "sad",
)


@dataclass
class ModelConfig:
    """"""
    # 1 channel means grayscale, 48x48 pixel image
    input_shape: tuple[int, int, int] = (1, 48, 48)

    # 3 emotions
    num_classes: int = len(EMOTION_LABELS)

    # 2 Layer CNN
    conv1_filters: int = 16
    conv1_kernel: int = 3
    conv2_filters: int = 32
    conv2_kernel: int = 3

    hidden_units: int = 128
    random_seed: int = 42
    pool_size: int = 2
    dropout_rate: float = 0.3
    l2_lambda: float = 1e-4


class EmotionCNN:
    """NumPy CNN for emotion classification of a 48x48 grayscaled image."""
    def __init__(self, config: ModelConfig | None = None) -> None:
        self.config = config or ModelConfig()
        self.rng = np.random.default_rng(self.config.random_seed)
        self.params: dict[str, Array] = {}
        self.grads: dict[str, Array] = {}
        self.cache: dict[str, Array] = {}

        self._initialize_parameters()

    def _initialize_parameters(self) -> None:
        """Create convolution and dense layer weights."""
        # Read config values into local variables
        channels, height, width = self.config.input_shape
        conv1_filters = self.config.conv1_filters
        conv1_kernel = self.config.conv1_kernel
        conv2_filters = self.config.conv2_filters
        conv2_kernel = self.config.conv2_kernel
        hidden_units = self.config.hidden_units
        num_classes = self.config.num_classes
        # First convolution weight tensor and bias vector
        self.params["W_conv1"] = self._he_initialize(
            size=(conv1_filters, channels, conv1_kernel, conv1_kernel),
            fan_in=channels * conv1_kernel * conv1_kernel,
        )

        self.params["b_conv1"] = np.zeros(conv1_filters, dtype=np.float32)

        # Second convolution weight tensor and bias vector
        # This tensor uses the output of conv1_filters as the input channel
        self.params["W_conv2"] = self._he_initialize(
            size=(conv2_filters, conv1_filters, conv2_kernel, conv2_kernel),
            fan_in=conv1_filters * conv2_kernel * conv2_kernel,
        )

        self.params["b_conv2"] = np.zeros(conv2_filters, dtype=np.float32)

        pooled_height = height // (self.config.pool_size * self.config.pool_size)
        pooled_width = width // (self.config.pool_size * self.config.pool_size)
        flattened_units = conv2_filters * pooled_height * pooled_width

        self.params["W_fc1"] = self._he_initialize(
            size=(flattened_units, hidden_units),
            fan_in=flattened_units,
        )
        self.params["b_fc1"] = np.zeros(hidden_units, dtype=np.float32)

        self.params["W_fc2"] = self._he_initialize(
            size=(hidden_units, num_classes),
            fan_in=hidden_units,
        )
        self.params["b_fc2"] = np.zeros(num_classes, dtype=np.float32)

    def forward(self, x: Array, training: bool = False) -> Array:
        """
        Uses current parameters to compute class logits for a batch.
        The logits are raw class scores, which are floats (can be negative)
        """
        x = self._prepare_batch(x)

        conv1 = self._conv2d_same(x, self.params["W_conv1"], self.params["b_conv1"])
        relu1 = self._relu(conv1)
        pool1 = self._max_pool2d(relu1, self.config.pool_size)

        conv2 = self._conv2d_same(
            pool1,
            self.params["W_conv2"],
            self.params["b_conv2"],
        )
        relu2 = self._relu(conv2)
        pool2 = self._max_pool2d(relu2, self.config.pool_size)

        flattened = pool2.reshape(pool2.shape[0], -1)
        fc1_linear = flattened @ self.params["W_fc1"] + self.params["b_fc1"]
        hidden = self._relu(fc1_linear)
        hidden_dropout, dropout_mask = self._dropout(hidden, self.config.dropout_rate, training)
        logits = hidden_dropout @ self.params["W_fc2"] + self.params["b_fc2"]

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

    def predict_proba(self, img: Array) -> Array:
        """
        Return a probability vector for one preprocessed image.
        The class probabilities must all sum to 1.0.
        """
        batch = self._prepare_single_image(img)
        logits = self.forward(batch)
        return self._softmax(logits)[0]

    def predict(self, img: Array) -> int:
        """Return the emotion class index for one preprocessed image."""
        probabilities = self.predict_proba(img)
        return int(np.argmax(probabilities))

    def cross_entropy_loss(
        self,
        logits: Array,
        labels: Array,
        class_weights: Array | None = None,
        l2_lambda: float = 0.0,
    ) -> float:
        """
        Return average cross-entropy loss for a batch of class logits.
        Labels should be integer class ids in {0, ..., num_classes - 1}.
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = self._prepare_labels(labels, logits.shape[0])
        probabilities = self._softmax(logits)

        batch_indices = np.arange(logits.shape[0])
        correct_class_probs = probabilities[batch_indices, labels]
        losses = -np.log(correct_class_probs + 1e-12)

        if class_weights is None:
            data_loss = float(np.mean(losses))
        else:
            sample_weights = np.asarray(class_weights, dtype=np.float32)[labels]
            data_loss = float(np.sum(losses * sample_weights) / np.sum(sample_weights))

        l2_loss = 0.5 * l2_lambda * sum(
            float(np.sum(self.params[name] ** 2))
            for name in ("W_conv1", "W_conv2", "W_fc1", "W_fc2")
        )
        return data_loss + l2_loss

    def softmax_gradient(
        self,
        logits: Array,
        labels: Array,
        class_weights: Array | None = None,
    ) -> Array:
        """
        Return dLoss/dLogits for softmax followed by cross-entropy loss.
        This is the first gradient needed when implementing backward().
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = self._prepare_labels(labels, logits.shape[0])
        probabilities = self._softmax(logits)

        batch_indices = np.arange(logits.shape[0])
        probabilities[batch_indices, labels] -= 1.0

        if class_weights is None:
            return probabilities / logits.shape[0]

        sample_weights = np.asarray(class_weights, dtype=np.float32)[labels]
        normalization = np.sum(sample_weights)
        return probabilities * sample_weights[:, None] / normalization

    def backward(
        self,
        labels: Array,
        class_weights: Array | None = None,
        l2_lambda: float = 0.0,
    ) -> Array:
        """
        Compute gradients for all trainable parameters after forward().
        Returns dLoss/dInput for optional debugging or gradient checks.
        """
        logits = self.cache["logits"]
        hidden = self.cache["hidden"]
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

        d_logits = self.softmax_gradient(logits, labels, class_weights)

        self.grads["W_fc2"] = hidden_dropout.T @ d_logits
        self.grads["b_fc2"] = np.sum(d_logits, axis=0)

        d_hidden = d_logits @ self.params["W_fc2"].T
        d_hidden *= dropout_mask
        d_fc1_linear = d_hidden * (fc1_linear > 0.0)

        self.grads["W_fc1"] = flattened.T @ d_fc1_linear
        self.grads["b_fc1"] = np.sum(d_fc1_linear, axis=0)

        d_flattened = d_fc1_linear @ self.params["W_fc1"].T
        d_pool2 = d_flattened.reshape(pool2.shape)
        d_relu2 = self._max_pool2d_backward(d_pool2, relu2, self.config.pool_size)
        d_conv2 = d_relu2 * self._relu_gradient(conv2)

        d_pool1, self.grads["W_conv2"], self.grads["b_conv2"] = self._conv2d_same_backward(
            d_conv2,
            pool1,
            self.params["W_conv2"],
        )
        d_relu1 = self._max_pool2d_backward(d_pool1, relu1, self.config.pool_size)
        d_conv1 = d_relu1 * self._relu_gradient(conv1)

        d_x, self.grads["W_conv1"], self.grads["b_conv1"] = self._conv2d_same_backward(
            d_conv1,
            x,
            self.params["W_conv1"],
        )

        self.grads["W_fc2"] += l2_lambda * self.params["W_fc2"]
        self.grads["W_fc1"] += l2_lambda * self.params["W_fc1"]
        self.grads["W_conv2"] += l2_lambda * self.params["W_conv2"]
        self.grads["W_conv1"] += l2_lambda * self.params["W_conv1"]

        self.cache["d_input"] = d_x
        return d_x

    def update_parameters(self, learning_rate: float) -> None:
        """Apply a simple stochastic gradient descent update."""
        for name, parameter in self.params.items():
            parameter -= learning_rate * self.grads[name]

    def train_step(
        self,
        x: Array,
        labels: Array,
        learning_rate: float,
        class_weights: Array | None = None,
        l2_lambda: float = 0.0,
    ) -> float:
        """Run one forward/backward/update step and return the batch loss."""
        logits = self.forward(x, training=True)
        loss = self.cross_entropy_loss(logits, labels, class_weights, l2_lambda)
        self.backward(labels, class_weights, l2_lambda)
        self.update_parameters(learning_rate)
        return loss

    def _he_initialize(self, size: tuple[int, ...], fan_in: int) -> Array:
        scale = np.sqrt(2.0 / fan_in)
        return self.rng.normal(loc=0.0, scale=scale, size=size).astype(np.float32)

    def _dropout(self, x: Array, dropout_rate: float, training: bool) -> tuple[Array, Array]:
        if not training or dropout_rate <= 0.0:
            mask = np.ones_like(x, dtype=np.float32)
            return x, mask

        keep_prob = 1.0 - dropout_rate
        mask = (self.rng.random(x.shape) < keep_prob).astype(np.float32) / keep_prob
        return x * mask, mask

    def save_parameters(self, path: str) -> None:
        """Save trained parameters as a NumPy .npz file."""
        np.savez(path, **self.params)

    def load_parameters(self, path: str) -> None:
        """Load trained parameters saved as a NumPy .npz file."""
        weights = np.load(path)
        for name in self.params:
            self.params[name] = weights[name].astype(np.float32)

    def _prepare_single_image(self, img: Array) -> Array:
        img = np.asarray(img, dtype=np.float32)
        return img.reshape(1, *self.config.input_shape)

    def _prepare_batch(self, x: Array) -> Array:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 3:
            x = x.reshape(1, *x.shape)
        return x

    def _prepare_labels(self, labels: Array, batch_size: int) -> Array:
        return np.asarray(labels, dtype=np.int64).reshape(-1)

    @staticmethod
    def _conv2d_same(x: Array, weights: Array, bias: Array) -> Array:
        """
        Performs a 2D convolution and returns an array of same height and width
        Each output pixel is a weighted sum of the respective 3x3 pixel region
        """
        _, _, _, _ = x.shape
        _, _, kernel_height, kernel_width = weights.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        # Pad the border with zeroes
        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
            mode="constant",
        )
        windows = sliding_window_view(
            padded,
            (kernel_height, kernel_width),
            axis=(2, 3),
        )
        output = np.tensordot(windows, weights, axes=([1, 4, 5], [1, 2, 3]))
        output = np.moveaxis(output, -1, 1)
        return output.astype(np.float32) + bias.reshape(1, -1, 1, 1)

    @staticmethod
    def _max_pool2d(x: Array, pool_size: int) -> Array:
        """
        Given a set of outputs, pool them in 2x2 regions and take the maximum
        This will reduce computation time and keep the strongest features
        """
        batch_size, channels, height, width = x.shape
        output_height = height // pool_size
        output_width = width // pool_size
        pooled = x.reshape(
            batch_size,
            channels,
            output_height,
            pool_size,
            output_width,
            pool_size,
        )
        return pooled.max(axis=(3, 5))

    @staticmethod
    def _max_pool2d_backward(d_pooled: Array, x: Array, pool_size: int) -> Array:
        """Move each pooled gradient back to the max location from the forward pass."""
        batch_size, channels, height, width = x.shape
        output_height = height // pool_size
        output_width = width // pool_size

        x_blocks = x.reshape(
            batch_size,
            channels,
            output_height,
            pool_size,
            output_width,
            pool_size,
        )
        max_values = x_blocks.max(axis=(3, 5), keepdims=True)
        max_mask = x_blocks == max_values
        ties = max_mask.sum(axis=(3, 5), keepdims=True)

        d_blocks = max_mask * d_pooled[:, :, :, None, :, None] / ties
        return d_blocks.reshape(x.shape).astype(np.float32)

    @staticmethod
    def _conv2d_same_backward(d_output: Array, x: Array, weights: Array) -> tuple[Array, Array, Array]:
        """Return gradients for input, weights, and bias of a same-padded convolution."""
        _, _, height, width = x.shape
        _, _, kernel_height, kernel_width = weights.shape
        pad_height = kernel_height // 2
        pad_width = kernel_width // 2

        padded = np.pad(
            x,
            ((0, 0), (0, 0), (pad_height, pad_height), (pad_width, pad_width)),
            mode="constant",
        )
        windows = sliding_window_view(
            padded,
            (kernel_height, kernel_width),
            axis=(2, 3),
        )

        d_weights = np.tensordot(d_output, windows, axes=([0, 2, 3], [0, 2, 3]))
        d_bias = np.sum(d_output, axis=(0, 2, 3))

        d_padded = np.zeros_like(padded, dtype=np.float32)
        for row in range(kernel_height):
            for col in range(kernel_width):
                d_patch = np.tensordot(d_output, weights[:, :, row, col], axes=([1], [0]))
                d_padded[:, :, row : row + height, col : col + width] += np.moveaxis(
                    d_patch,
                    -1,
                    1,
                )

        if pad_height == 0 and pad_width == 0:
            d_x = d_padded
        else:
            d_x = d_padded[:, :, pad_height:-pad_height, pad_width:-pad_width]

        return d_x.astype(np.float32), d_weights.astype(np.float32), d_bias.astype(np.float32)

    @staticmethod
    def _relu(x: Array) -> Array:
        """ReLU adds more nonlinearity which will increase performance at the cost of training time"""
        return np.maximum(x, 0.0)

    @staticmethod
    def _relu_gradient(x: Array) -> Array:
        return (x > 0.0).astype(np.float32)

    @staticmethod
    def _softmax(logits: Array) -> Array:
        """Converts raw class scores into probabilites between [0,1]"""
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


""" Make a new model that loads the training weights"""
_WEIGHTS_PATH = Path(__file__).with_name("emotion_cnn_weights.npz")

_MODEL = EmotionCNN()
if _WEIGHTS_PATH.exists():
    _MODEL.load_parameters(str(_WEIGHTS_PATH))



def predict_emotion(img: Array) -> int:
    """Frontend entry point: predict an emotion from a 48x48 float image."""
    return _MODEL.predict(img)