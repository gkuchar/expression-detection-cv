from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


Array = np.ndarray

EMOTION_LABELS: tuple[str, ...] = (
    "angry",
    "contempt",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprised",
)


@dataclass
class ModelConfig:
    """"""
    # 1 channel means grayscale, 128x128 pixel image
    input_shape: tuple[int, int, int] = (1, 128, 128)

    # 8 emotions
    num_classes: int = 8

    # 2 Layer CNN
    conv1_filters: int = 8
    conv1_kernel: int = 3
    conv2_filters: int = 16
    conv2_kernel: int = 3

    hidden_units: int = 64
    weight_scale: float = 0.01
    random_seed: int = 42
    pool_size: int = 2


class EmotionCNN:
    """NumPy CNN for emotion classification of a 128x128 grayscaled image."""
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
        weight_scale = self.config.weight_scale

        # First convolution weight tensor and bias vector
        self.params["W_conv1"] = self.rng.normal(
            loc=0.0,
            scale=weight_scale,
            size=(conv1_filters, channels, conv1_kernel, conv1_kernel),
        ).astype(np.float32)

        self.params["b_conv1"] = np.zeros(conv1_filters, dtype=np.float32)

        # Second convolution weight tensor and bias vector
        # This tensor uses the output of conv1_filters as the input channel
        self.params["W_conv2"] = self.rng.normal(
            loc=0.0,
            scale=weight_scale,
            size=(conv2_filters, conv1_filters, conv2_kernel, conv2_kernel),
        ).astype(np.float32)

        self.params["b_conv2"] = np.zeros(conv2_filters, dtype=np.float32)

        pooled_height = height // (self.config.pool_size * self.config.pool_size)
        pooled_width = width // (self.config.pool_size * self.config.pool_size)
        flattened_units = conv2_filters * pooled_height * pooled_width

        self.params["W_fc1"] = self.rng.normal(
            loc=0.0,
            scale=weight_scale,
            size=(flattened_units, hidden_units),
        ).astype(np.float32)
        self.params["b_fc1"] = np.zeros(hidden_units, dtype=np.float32)

        self.params["W_fc2"] = self.rng.normal(
            loc=0.0,
            scale=weight_scale,
            size=(hidden_units, num_classes),
        ).astype(np.float32)
        self.params["b_fc2"] = np.zeros(num_classes, dtype=np.float32)

    def forward(self, x: Array) -> Array:
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
        logits = hidden @ self.params["W_fc2"] + self.params["b_fc2"]

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
            "logits": logits,
        }
        return logits

    def predict_proba(self, img: Array) -> Array:
        """
        Return an 8-element probability vector for one preprocessed image.
        The 8 probabilities must all sum to 1.0
        """
        batch = self._prepare_single_image(img)
        logits = self.forward(batch)
        return self._softmax(logits)[0]

    def predict(self, img: Array) -> int:
        """Return the emotion class index for one preprocessed image."""
        probabilities = self.predict_proba(img)
        return int(np.argmax(probabilities))

    def cross_entropy_loss(self, logits: Array, labels: Array) -> float:
        """
        Return average cross-entropy loss for a batch of class logits.
        Labels should be integer class ids in {0, ..., num_classes - 1}.
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = self._prepare_labels(labels, logits.shape[0])
        probabilities = self._softmax(logits)

        batch_indices = np.arange(logits.shape[0])
        correct_class_probs = probabilities[batch_indices, labels]
        return float(-np.mean(np.log(correct_class_probs + 1e-12)))

    def softmax_gradient(self, logits: Array, labels: Array) -> Array:
        """
        Return dLoss/dLogits for softmax followed by cross-entropy loss.
        This is the first gradient needed when implementing backward().
        """
        logits = np.asarray(logits, dtype=np.float32)
        labels = self._prepare_labels(labels, logits.shape[0])
        probabilities = self._softmax(logits)

        batch_indices = np.arange(logits.shape[0])
        probabilities[batch_indices, labels] -= 1.0
        return probabilities / logits.shape[0]

    def backward(self, labels: Array) -> Array:
        """
        Compute gradients for all trainable parameters after forward().
        Returns dLoss/dInput for optional debugging or gradient checks.
        """
        logits = self.cache["logits"]
        hidden = self.cache["hidden"]
        fc1_linear = self.cache["fc1_linear"]
        flattened = self.cache["flattened"]
        pool2 = self.cache["pool2"]
        relu2 = self.cache["relu2"]
        conv2 = self.cache["conv2"]
        pool1 = self.cache["pool1"]
        relu1 = self.cache["relu1"]
        conv1 = self.cache["conv1"]
        x = self.cache["x"]

        d_logits = self.softmax_gradient(logits, labels)

        self.grads["W_fc2"] = hidden.T @ d_logits
        self.grads["b_fc2"] = np.sum(d_logits, axis=0)

        d_hidden = d_logits @ self.params["W_fc2"].T
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

        self.cache["d_input"] = d_x
        return d_x

    def update_parameters(self, learning_rate: float) -> None:
        """Apply a simple stochastic gradient descent update."""
        for name, parameter in self.params.items():
            parameter -= learning_rate * self.grads[name]

    def train_step(self, x: Array, labels: Array, learning_rate: float) -> float:
        """Run one forward/backward/update step and return the batch loss."""
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, labels)
        self.backward(labels)
        self.update_parameters(learning_rate)
        return loss

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


_MODEL = EmotionCNN()


def predict_emotion(img: Array) -> int:
    """Frontend entry point: predict an emotion from a 128x128 float image."""
    return _MODEL.predict(img)
