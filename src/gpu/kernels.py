from __future__ import annotations

from dataclasses import dataclass

import cupy as cp
import numpy as np


Array = cp.ndarray


@dataclass(frozen=True)
class KernelLaunchConfig:
    """Default launch sizes for simple one-element-per-thread kernels."""

    threads_per_block: int = 256

    def blocks_for(self, element_count: int) -> int:
        return (element_count + self.threads_per_block - 1) // self.threads_per_block


DEFAULT_LAUNCH = KernelLaunchConfig()
SIZE = 48


# Keep CUDA source strings close to the wrappers that launch them. For now these
# are intentionally placeholders so the GPU model can be wired up incrementally.
_RELU_SOURCE = r"""
extern "C" __global__
void relu_forward(const float* x, float* out, const int size) {
    // One element per thread
    // O(n) -> O(1) time complexity
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        float value = x[i];
        out[i] = value > 0.0f ? value : 0.0f;
    }
}
"""


_CONV2D_SAME_SOURCE = r"""
extern "C" __global__
void conv2d_same_forward(
    const float* x,
    const float* weights,
    const float* bias,
    float* out,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width
) {
    // out shape is (N, C_out, H, W)
    // So each thread will take one flattened index and decompose it into:
    //  n = batch index
    //  oc = output channel
    //  oy = output row
    //  ox = output col
    //
    // Then loop over input channels, kernel rows, and kernel cols (skip padded pos's)

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = batch_size * out_channels * in_height * in_width;
    if (i >= output_size) {
        return;
    }

    int ox = i % in_width;
    int temp = i / in_width;

    int oy = temp % in_height;
    temp = temp / in_height;

    int oc = temp % out_channels;
    int n = temp / out_channels;

    int pad_y = kernel_height / 2;
    int pad_x = kernel_width / 2;

    float sum = bias[oc];

    for (int ic = 0; ic < in_channels; ic++) {
        for (int ky = 0; ky < kernel_height; ky++) {
            int iy = oy + ky - pad_y;

            if (iy < 0 || iy >= in_height) {
                continue;
            }

            for (int kx = 0; kx < kernel_width; kx++) {
                int ix = ox + kx - pad_x;

                if (ix < 0 || ix >= in_width) {
                    continue;
                }

                // Compute flattened index from row-major layout
                int x_index = ((n * in_channels + ic) * in_height + iy) * in_width + ix;

                int weight_index = ((oc * in_channels + ic) * kernel_height + ky) * kernel_width + kx;

                sum += x[x_index] * weights[weight_index];
            }
        }
    }

    // Each thread's i represents out[n, oc, oy, ox]
    out[i] = sum;
}
"""


_MAX_POOL2D_SOURCE = r"""
extern "C" __global__
void max_pool2d_forward(
    const float* x,
    float* out,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int pool_size
) {
    // x shape:   (N, C, H, W)
    // out shape: (N, C, H / pool_size, W / pool_size)

    // pool_size is 2, so each thread checks a 2x2 region
    // x[n, c, 2*oy,     2*ox  ]
    // x[n, c, 2*oy,     2*ox+1]
    // x[n, c, 2*oy + 1, 2*ox  ]
    // x[n, c, 2*oy + 1, 2*ox+1]
    // Then the largest is written to out[n, c, oy, ox]
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    int output_size = batch_size * channels * out_height * out_width;

    if (i >= output_size) {
        return;
    }

    int ox = i % out_width;
    int temp = i / out_width;

    int oy = temp % out_height;
    temp = temp / out_height;

    int c = temp % channels;
    int n = temp / channels;

    int start_y = oy * pool_size;
    int start_x = ox * pool_size;

    float max_value = x[((n * channels + c) * in_height + start_y) * in_width + start_x];

    for (int py = 0; py < pool_size; py++) {
        int iy = start_y + py;

        for (int px = 0; px < pool_size; px++) {
            int ix = start_x + px;

            int x_index = ((n * channels + c) * in_height + iy) * in_width + ix;
            float value = x[x_index];

            if (value > max_value) {
                max_value = value;
            }
        }
    }

    out[i] = max_value;
}
"""


_DENSE_SOURCE = r"""
extern "C" __global__
void dense_forward(
    const float* x,
    const float* weights,
    const float* bias,
    float* out,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    // W_fc1 has shape (in_features, out_features)
    // So weights[in_feature, out_feature] flattens to 
    //  weight_index := in_feature * out_features + out_feature

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int output_size = batch_size * out_features;

    if (i >= output_size) {
        return;
    }

    int out_feature = i % out_features;
    int n = i / out_features;

    float sum = bias[out_feature];

    for (int in_feature = 0; in_feature < in_features; in_feature++) {
        int x_index = n * in_features + in_feature;
        int weight_index = in_feature * out_features + out_feature;

        sum += x[x_index] * weights[weight_index];
    }

    out[i] = sum;
}
"""


_ARGMAX_SOURCE = r"""
extern "C" __global__
void argmax_forward(
    const float* x,
    int* out,
    const int batch_size,
    const int classes
) {
    // x shape:   (N, classes)
    // out shape: (N,)

    // Only argmax is needed since the final dense layer still returns raw logits
    // So argmax(logits) = argmax(softmax(logits))

    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size) {
        return;
    }

    int best_class = 0;
    float best_value = x[n * classes];

    for (int c = 1; c < classes; c++) {
        float value = x[n * classes + c];

        if (value > best_value) {
            best_value = value;
            best_class = c;
        }
    }

    out[n] = best_class;
}
"""


_RELU_BACKWARD_SOURCE = r"""
extern "C" __global__
void relu_backward(
    const float* d_out,
    const float* x,
    float* d_x,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        d_x[i] = x[i] > 0.0f ? d_out[i] : 0.0f;
    }
}
"""


_MAX_POOL2D_BACKWARD_SOURCE = r"""
extern "C" __global__
void max_pool2d_backward(
    const float* d_pooled,
    const float* x,
    float* d_x,
    const int batch_size,
    const int channels,
    const int in_height,
    const int in_width,
    const int pool_size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = in_height / pool_size;
    int out_width = in_width / pool_size;
    int output_size = batch_size * channels * out_height * out_width;

    if (i >= output_size) {
        return;
    }

    int ox = i % out_width;
    int temp = i / out_width;

    int oy = temp % out_height;
    temp = temp / out_height;

    int c = temp % channels;
    int n = temp / channels;

    int start_y = oy * pool_size;
    int start_x = ox * pool_size;

    float max_value = x[((n * channels + c) * in_height + start_y) * in_width + start_x];

    for (int py = 0; py < pool_size; py++) {
        int iy = start_y + py;

        for (int px = 0; px < pool_size; px++) {
            int ix = start_x + px;
            int x_index = ((n * channels + c) * in_height + iy) * in_width + ix;
            float value = x[x_index];

            if (value > max_value) {
                max_value = value;
            }
        }
    }

    int ties = 0;
    for (int py = 0; py < pool_size; py++) {
        int iy = start_y + py;

        for (int px = 0; px < pool_size; px++) {
            int ix = start_x + px;
            int x_index = ((n * channels + c) * in_height + iy) * in_width + ix;

            if (x[x_index] == max_value) {
                ties += 1;
            }
        }
    }

    float grad = d_pooled[i] / (float)ties;
    for (int py = 0; py < pool_size; py++) {
        int iy = start_y + py;

        for (int px = 0; px < pool_size; px++) {
            int ix = start_x + px;
            int x_index = ((n * channels + c) * in_height + iy) * in_width + ix;

            d_x[x_index] = x[x_index] == max_value ? grad : 0.0f;
        }
    }
}
"""


_DENSE_BACKWARD_INPUT_SOURCE = r"""
extern "C" __global__
void dense_backward_input(
    const float* d_out,
    const float* weights,
    float* d_x,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int input_size = batch_size * in_features;
    if (i >= input_size) {
        return;
    }

    int in_feature = i % in_features;
    int n = i / in_features;

    float sum = 0.0f;
    for (int out_feature = 0; out_feature < out_features; out_feature++) {
        int d_out_index = n * out_features + out_feature;
        int weight_index = in_feature * out_features + out_feature;
        sum += d_out[d_out_index] * weights[weight_index];
    }

    d_x[i] = sum;
}
"""


_DENSE_BACKWARD_WEIGHTS_SOURCE = r"""
extern "C" __global__
void dense_backward_weights(
    const float* x,
    const float* d_out,
    float* d_weights,
    const int batch_size,
    const int in_features,
    const int out_features
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int weight_size = in_features * out_features;
    if (i >= weight_size) {
        return;
    }

    int out_feature = i % out_features;
    int in_feature = i / out_features;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        int x_index = n * in_features + in_feature;
        int d_out_index = n * out_features + out_feature;
        sum += x[x_index] * d_out[d_out_index];
    }

    d_weights[i] = sum;
}
"""


_DENSE_BACKWARD_BIAS_SOURCE = r"""
extern "C" __global__
void dense_backward_bias(
    const float* d_out,
    float* d_bias,
    const int batch_size,
    const int out_features
) {
    int out_feature = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_feature >= out_features) {
        return;
    }

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        sum += d_out[n * out_features + out_feature];
    }

    d_bias[out_feature] = sum;
}
"""


_CONV2D_SAME_BACKWARD_INPUT_SOURCE = r"""
extern "C" __global__
void conv2d_same_backward_input(
    const float* d_out,
    const float* weights,
    float* d_x,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int input_size = batch_size * in_channels * in_height * in_width;
    if (i >= input_size) {
        return;
    }

    int ix = i % in_width;
    int temp = i / in_width;

    int iy = temp % in_height;
    temp = temp / in_height;

    int ic = temp % in_channels;
    int n = temp / in_channels;

    int pad_y = kernel_height / 2;
    int pad_x = kernel_width / 2;

    float sum = 0.0f;
    for (int oc = 0; oc < out_channels; oc++) {
        for (int ky = 0; ky < kernel_height; ky++) {
            int oy = iy - ky + pad_y;
            if (oy < 0 || oy >= in_height) {
                continue;
            }

            for (int kx = 0; kx < kernel_width; kx++) {
                int ox = ix - kx + pad_x;
                if (ox < 0 || ox >= in_width) {
                    continue;
                }

                int d_out_index = ((n * out_channels + oc) * in_height + oy) * in_width + ox;
                int weight_index = ((oc * in_channels + ic) * kernel_height + ky) * kernel_width + kx;
                sum += d_out[d_out_index] * weights[weight_index];
            }
        }
    }

    d_x[i] = sum;
}
"""


_CONV2D_SAME_BACKWARD_WEIGHTS_SOURCE = r"""
extern "C" __global__
void conv2d_same_backward_weights(
    const float* d_out,
    const float* x,
    float* d_weights,
    const int batch_size,
    const int in_channels,
    const int in_height,
    const int in_width,
    const int out_channels,
    const int kernel_height,
    const int kernel_width
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int weight_size = out_channels * in_channels * kernel_height * kernel_width;
    if (i >= weight_size) {
        return;
    }

    int kx = i % kernel_width;
    int temp = i / kernel_width;

    int ky = temp % kernel_height;
    temp = temp / kernel_height;

    int ic = temp % in_channels;
    int oc = temp / in_channels;

    int pad_y = kernel_height / 2;
    int pad_x = kernel_width / 2;

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int oy = 0; oy < in_height; oy++) {
            int iy = oy + ky - pad_y;
            if (iy < 0 || iy >= in_height) {
                continue;
            }

            for (int ox = 0; ox < in_width; ox++) {
                int ix = ox + kx - pad_x;
                if (ix < 0 || ix >= in_width) {
                    continue;
                }

                int d_out_index = ((n * out_channels + oc) * in_height + oy) * in_width + ox;
                int x_index = ((n * in_channels + ic) * in_height + iy) * in_width + ix;
                sum += d_out[d_out_index] * x[x_index];
            }
        }
    }

    d_weights[i] = sum;
}
"""


_CONV2D_SAME_BACKWARD_BIAS_SOURCE = r"""
extern "C" __global__
void conv2d_same_backward_bias(
    const float* d_out,
    float* d_bias,
    const int batch_size,
    const int out_channels,
    const int out_height,
    const int out_width
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;

    if (oc >= out_channels) {
        return;
    }

    float sum = 0.0f;
    for (int n = 0; n < batch_size; n++) {
        for (int oy = 0; oy < out_height; oy++) {
            for (int ox = 0; ox < out_width; ox++) {
                int d_out_index = ((n * out_channels + oc) * out_height + oy) * out_width + ox;
                sum += d_out[d_out_index];
            }
        }
    }

    d_bias[oc] = sum;
}
"""


_SOFTMAX_CROSS_ENTROPY_GRADIENT_SOURCE = r"""
extern "C" __global__
void softmax_cross_entropy_gradient(
    const float* logits,
    const int* labels,
    const float* class_weights,
    float* d_logits,
    const int batch_size,
    const int classes,
    const int use_class_weights
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;

    if (n >= batch_size) {
        return;
    }

    int row_start = n * classes;
    float max_logit = logits[row_start];

    for (int c = 1; c < classes; c++) {
        float value = logits[row_start + c];
        if (value > max_logit) {
            max_logit = value;
        }
    }

    float sum_exp = 0.0f;
    for (int c = 0; c < classes; c++) {
        sum_exp += expf(logits[row_start + c] - max_logit);
    }

    float normalization = (float)batch_size;
    if (use_class_weights) {
        normalization = 0.0f;
        for (int row = 0; row < batch_size; row++) {
            normalization += class_weights[labels[row]];
        }
    }

    int label = labels[n];
    float sample_weight = use_class_weights ? class_weights[label] : 1.0f;
    for (int c = 0; c < classes; c++) {
        float probability = expf(logits[row_start + c] - max_logit) / sum_exp;
        float target = c == label ? 1.0f : 0.0f;
        d_logits[row_start + c] = (probability - target) * sample_weight / normalization;
    }
}
"""


_CROSS_ENTROPY_LOSS_SOURCE = r"""
extern "C" __global__
void cross_entropy_loss(
    const float* logits,
    const int* labels,
    const float* class_weights,
    float* out,
    const int batch_size,
    const int classes,
    const int use_class_weights
) {
    float weighted_loss_sum = 0.0f;
    float normalization = use_class_weights ? 0.0f : (float)batch_size;

    for (int n = 0; n < batch_size; n++) {
        int row_start = n * classes;
        float max_logit = logits[row_start];

        for (int c = 1; c < classes; c++) {
            float value = logits[row_start + c];
            if (value > max_logit) {
                max_logit = value;
            }
        }

        float sum_exp = 0.0f;
        for (int c = 0; c < classes; c++) {
            sum_exp += expf(logits[row_start + c] - max_logit);
        }

        int label = labels[n];
        float probability = expf(logits[row_start + label] - max_logit) / sum_exp;
        float sample_weight = use_class_weights ? class_weights[label] : 1.0f;

        weighted_loss_sum += -logf(probability + 1e-12f) * sample_weight;
        if (use_class_weights) {
            normalization += sample_weight;
        }
    }

    out[0] = weighted_loss_sum / normalization;
}
"""


_DROPOUT_FORWARD_SOURCE = r"""
extern "C" __global__
void dropout_forward(
    const float* x,
    const float* random_values,
    float* out,
    float* mask,
    const float keep_prob,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        if (random_values[i] < keep_prob) {
            mask[i] = 1.0f / keep_prob;
            out[i] = x[i] * mask[i];
        } else {
            mask[i] = 0.0f;
            out[i] = 0.0f;
        }
    }
}
"""


_DROPOUT_BACKWARD_SOURCE = r"""
extern "C" __global__
void dropout_backward(
    const float* d_out,
    const float* mask,
    float* d_x,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        d_x[i] = d_out[i] * mask[i];
    }
}
"""


_SUM_SQUARES_SOURCE = r"""
extern "C" __global__
void sum_squares(
    const float* x,
    float* out,
    const int size
) {
    float sum = 0.0f;

    for (int i = 0; i < size; i++) {
        sum += x[i] * x[i];
    }

    out[0] = sum;
}
"""


_SGD_UPDATE_SOURCE = r"""
extern "C" __global__
void sgd_update(
    float* parameter,
    const float* gradient,
    const float learning_rate,
    const float l2_lambda,
    const int size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size) {
        parameter[i] -= learning_rate * (gradient[i] + l2_lambda * parameter[i]);
    }
}
"""


_relu_kernel = cp.RawKernel(_RELU_SOURCE, "relu_forward")
_conv2d_same_kernel = cp.RawKernel(_CONV2D_SAME_SOURCE, "conv2d_same_forward")
_max_pool2d_kernel = cp.RawKernel(_MAX_POOL2D_SOURCE, "max_pool2d_forward")
_dense_kernel = cp.RawKernel(_DENSE_SOURCE, "dense_forward")
_argmax_kernel = cp.RawKernel(_ARGMAX_SOURCE, "argmax_forward")
_relu_backward_kernel = cp.RawKernel(_RELU_BACKWARD_SOURCE, "relu_backward")
_max_pool2d_backward_kernel = cp.RawKernel(_MAX_POOL2D_BACKWARD_SOURCE, "max_pool2d_backward")
_dense_backward_input_kernel = cp.RawKernel(_DENSE_BACKWARD_INPUT_SOURCE, "dense_backward_input")
_dense_backward_weights_kernel = cp.RawKernel(_DENSE_BACKWARD_WEIGHTS_SOURCE, "dense_backward_weights")
_dense_backward_bias_kernel = cp.RawKernel(_DENSE_BACKWARD_BIAS_SOURCE, "dense_backward_bias")
_conv2d_same_backward_input_kernel = cp.RawKernel(
    _CONV2D_SAME_BACKWARD_INPUT_SOURCE,
    "conv2d_same_backward_input",
)
_conv2d_same_backward_weights_kernel = cp.RawKernel(
    _CONV2D_SAME_BACKWARD_WEIGHTS_SOURCE,
    "conv2d_same_backward_weights",
)
_conv2d_same_backward_bias_kernel = cp.RawKernel(
    _CONV2D_SAME_BACKWARD_BIAS_SOURCE,
    "conv2d_same_backward_bias",
)
_softmax_cross_entropy_gradient_kernel = cp.RawKernel(
    _SOFTMAX_CROSS_ENTROPY_GRADIENT_SOURCE,
    "softmax_cross_entropy_gradient",
)
_cross_entropy_loss_kernel = cp.RawKernel(_CROSS_ENTROPY_LOSS_SOURCE, "cross_entropy_loss")
_dropout_forward_kernel = cp.RawKernel(_DROPOUT_FORWARD_SOURCE, "dropout_forward")
_dropout_backward_kernel = cp.RawKernel(_DROPOUT_BACKWARD_SOURCE, "dropout_backward")
_sum_squares_kernel = cp.RawKernel(_SUM_SQUARES_SOURCE, "sum_squares")
_sgd_update_kernel = cp.RawKernel(_SGD_UPDATE_SOURCE, "sgd_update")


def relu_forward(x: Array) -> Array:
    """Return ReLU(x) using a custom CUDA kernel."""
    x = cp.asarray(x, dtype=cp.float32)
    out = cp.empty_like(x)

    # Use a 1D launch since input array is 1D
    size = x.size
    grid = (DEFAULT_LAUNCH.blocks_for(size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _relu_kernel(grid, block, (x, out, size))
    return out


def conv2d_same_forward(x: Array, weights: Array, bias: Array) -> Array:
    """Run same-padded 2D convolution using a custom CUDA kernel."""
    x = cp.asarray(x, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)
    bias = cp.asarray(bias, dtype=cp.float32)

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = weights.shape
    out = cp.empty((batch_size, out_channels, in_height, in_width), dtype=cp.float32)

    element_count = out.size
    grid = (DEFAULT_LAUNCH.blocks_for(element_count),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _conv2d_same_kernel(
        grid,
        block,
        (
            x,
            weights,
            bias,
            out,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
        ),
    )
    return out


def max_pool2d_forward(x: Array, pool_size: int) -> Array:
    """Run 2D max pooling using a custom CUDA kernel."""
    x = cp.asarray(x, dtype=cp.float32)
    batch_size, channels, in_height, in_width = x.shape
    out_height = in_height // pool_size
    out_width = in_width // pool_size
    out = cp.empty((batch_size, channels, out_height, out_width), dtype=cp.float32)

    element_count = out.size
    grid = (DEFAULT_LAUNCH.blocks_for(element_count),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _max_pool2d_kernel(
        grid,
        block,
        (
            x,
            out,
            batch_size,
            channels,
            in_height,
            in_width,
            pool_size,
        ),
    )
    return out


def dense_forward(x: Array, weights: Array, bias: Array) -> Array:
    """Run a fully connected layer using a custom CUDA kernel."""
    x = cp.asarray(x, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)
    bias = cp.asarray(bias, dtype=cp.float32)

    batch_size, in_features = x.shape
    _, out_features = weights.shape
    out = cp.empty((batch_size, out_features), dtype=cp.float32)

    element_count = out.size
    grid = (DEFAULT_LAUNCH.blocks_for(element_count),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _dense_kernel(
        grid,
        block,
        (
            x,
            weights,
            bias,
            out,
            batch_size,
            in_features,
            out_features,
        ),
    )
    return out


def argmax_forward(x: Array) -> Array:
    """Return argmax for each batch row using a custom CUDA kernel."""
    x = cp.asarray(x, dtype=cp.float32)
    batch_size, classes = x.shape
    out = cp.empty((batch_size,), dtype=cp.int32)

    grid = (DEFAULT_LAUNCH.blocks_for(batch_size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _argmax_kernel(grid, block, (x, out, batch_size, classes))
    return out


def relu_backward(d_out: Array, x: Array) -> Array:
    """Backpropagate through ReLU."""
    d_out = cp.asarray(d_out, dtype=cp.float32)
    x = cp.asarray(x, dtype=cp.float32)
    d_x = cp.empty_like(x)

    size = x.size
    grid = (DEFAULT_LAUNCH.blocks_for(size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _relu_backward_kernel(grid, block, (d_out, x, d_x, size))
    return d_x


def max_pool2d_backward(d_pooled: Array, x: Array, pool_size: int) -> Array:
    """Backpropagate through non-overlapping 2D max pooling."""
    d_pooled = cp.asarray(d_pooled, dtype=cp.float32)
    x = cp.asarray(x, dtype=cp.float32)
    batch_size, channels, in_height, in_width = x.shape
    d_x = cp.empty_like(x)

    element_count = d_pooled.size
    grid = (DEFAULT_LAUNCH.blocks_for(element_count),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _max_pool2d_backward_kernel(
        grid,
        block,
        (
            d_pooled,
            x,
            d_x,
            batch_size,
            channels,
            in_height,
            in_width,
            pool_size,
        ),
    )
    return d_x


def dense_backward(d_out: Array, x: Array, weights: Array) -> tuple[Array, Array, Array]:
    """Return d_x, d_weights, and d_bias for a dense layer."""
    d_out = cp.asarray(d_out, dtype=cp.float32)
    x = cp.asarray(x, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)

    batch_size, in_features = x.shape
    _, out_features = weights.shape

    d_x = cp.empty_like(x)
    d_weights = cp.empty_like(weights)
    d_bias = cp.empty((out_features,), dtype=cp.float32)

    block = (DEFAULT_LAUNCH.threads_per_block,)

    input_grid = (DEFAULT_LAUNCH.blocks_for(d_x.size),)
    _dense_backward_input_kernel(
        input_grid,
        block,
        (d_out, weights, d_x, batch_size, in_features, out_features),
    )

    weight_grid = (DEFAULT_LAUNCH.blocks_for(d_weights.size),)
    _dense_backward_weights_kernel(
        weight_grid,
        block,
        (x, d_out, d_weights, batch_size, in_features, out_features),
    )

    bias_grid = (DEFAULT_LAUNCH.blocks_for(out_features),)
    _dense_backward_bias_kernel(
        bias_grid,
        block,
        (d_out, d_bias, batch_size, out_features),
    )

    return d_x, d_weights, d_bias


def conv2d_same_backward(
    d_out: Array,
    x: Array,
    weights: Array,
) -> tuple[Array, Array, Array]:
    """Return d_x, d_weights, and d_bias for same-padded 2D convolution."""
    d_out = cp.asarray(d_out, dtype=cp.float32)
    x = cp.asarray(x, dtype=cp.float32)
    weights = cp.asarray(weights, dtype=cp.float32)

    batch_size, in_channels, in_height, in_width = x.shape
    out_channels, _, kernel_height, kernel_width = weights.shape

    d_x = cp.empty_like(x)
    d_weights = cp.empty_like(weights)
    d_bias = cp.empty((out_channels,), dtype=cp.float32)

    block = (DEFAULT_LAUNCH.threads_per_block,)

    input_grid = (DEFAULT_LAUNCH.blocks_for(d_x.size),)
    _conv2d_same_backward_input_kernel(
        input_grid,
        block,
        (
            d_out,
            weights,
            d_x,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
        ),
    )

    weight_grid = (DEFAULT_LAUNCH.blocks_for(d_weights.size),)
    _conv2d_same_backward_weights_kernel(
        weight_grid,
        block,
        (
            d_out,
            x,
            d_weights,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            kernel_height,
            kernel_width,
        ),
    )

    bias_grid = (DEFAULT_LAUNCH.blocks_for(out_channels),)
    _conv2d_same_backward_bias_kernel(
        bias_grid,
        block,
        (d_out, d_bias, batch_size, out_channels, in_height, in_width),
    )

    return d_x, d_weights, d_bias


def softmax_cross_entropy_gradient(
    logits: Array,
    labels: Array,
    class_weights: Array | None = None,
) -> Array:
    """Return dLoss/dLogits for softmax followed by cross-entropy."""
    logits = cp.asarray(logits, dtype=cp.float32)
    labels = cp.asarray(labels, dtype=cp.int32).reshape(-1)
    use_class_weights = class_weights is not None
    if class_weights is None:
        class_weights = cp.empty((1,), dtype=cp.float32)
    else:
        class_weights = cp.asarray(class_weights, dtype=cp.float32)
    batch_size, classes = logits.shape
    d_logits = cp.empty_like(logits)

    grid = (DEFAULT_LAUNCH.blocks_for(batch_size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _softmax_cross_entropy_gradient_kernel(
        grid,
        block,
        (
            logits,
            labels,
            class_weights,
            d_logits,
            batch_size,
            classes,
            np.int32(use_class_weights),
        ),
    )
    return d_logits


def cross_entropy_loss(
    logits: Array,
    labels: Array,
    class_weights: Array | None = None,
) -> Array:
    """Return scalar cross-entropy loss as a one-element GPU array."""
    logits = cp.asarray(logits, dtype=cp.float32)
    labels = cp.asarray(labels, dtype=cp.int32).reshape(-1)
    use_class_weights = class_weights is not None
    if class_weights is None:
        class_weights = cp.empty((1,), dtype=cp.float32)
    else:
        class_weights = cp.asarray(class_weights, dtype=cp.float32)

    batch_size, classes = logits.shape
    out = cp.empty((1,), dtype=cp.float32)
    _cross_entropy_loss_kernel(
        (1,),
        (1,),
        (
            logits,
            labels,
            class_weights,
            out,
            batch_size,
            classes,
            np.int32(use_class_weights),
        ),
    )
    return out


def dropout_forward(x: Array, random_values: Array, dropout_rate: float) -> tuple[Array, Array]:
    """Apply inverted dropout using random values."""
    x = cp.asarray(x, dtype=cp.float32)
    random_values = cp.asarray(random_values, dtype=cp.float32)
    out = cp.empty_like(x)
    mask = cp.empty_like(x)
    keep_prob = np.float32(1.0 - dropout_rate)

    size = x.size
    grid = (DEFAULT_LAUNCH.blocks_for(size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _dropout_forward_kernel(grid, block, (x, random_values, out, mask, keep_prob, size))
    return out, mask


def dropout_backward(d_out: Array, mask: Array) -> Array:
    """Backpropagate through inverted dropout."""
    d_out = cp.asarray(d_out, dtype=cp.float32)
    mask = cp.asarray(mask, dtype=cp.float32)
    d_x = cp.empty_like(d_out)

    size = d_out.size
    grid = (DEFAULT_LAUNCH.blocks_for(size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _dropout_backward_kernel(grid, block, (d_out, mask, d_x, size))
    return d_x


def sum_squares(x: Array) -> Array:
    """Return sum(x ** 2) as a one-element GPU array."""
    x = cp.asarray(x, dtype=cp.float32)
    out = cp.empty((1,), dtype=cp.float32)
    _sum_squares_kernel((1,), (1,), (x, out, x.size))
    return out


def sgd_update(
    parameter: Array,
    gradient: Array,
    learning_rate: float,
    l2_lambda: float = 0.0,
) -> None:
    """Apply parameter -= learning_rate * (gradient + l2_lambda * parameter)."""
    parameter = cp.asarray(parameter, dtype=cp.float32)
    gradient = cp.asarray(gradient, dtype=cp.float32)

    size = parameter.size
    grid = (DEFAULT_LAUNCH.blocks_for(size),)
    block = (DEFAULT_LAUNCH.threads_per_block,)
    _sgd_update_kernel(
        grid,
        block,
        (parameter, gradient, np.float32(learning_rate), np.float32(l2_lambda), size),
    )

_preprocess_kernel = cp.RawKernel(r'''
extern "C" __global__
void preprocess_face_kernel(
    const unsigned char* __restrict__ frame,
    float*               __restrict__ out,
    int frame_width,
    int x1, int y1,
    int src_w, int src_h,
    int dst_size
) {
    int dx = threadIdx.x + blockIdx.x * blockDim.x;
    int dy = threadIdx.y + blockIdx.y * blockDim.y;
    if (dx >= dst_size || dy >= dst_size) return;

    int wx0 = x1 + (dx       * src_w) / dst_size;
    int wy0 = y1 + (dy       * src_h) / dst_size;
    int wx1 = x1 + ((dx + 1) * src_w) / dst_size;
    int wy1 = y1 + ((dy + 1) * src_h) / dst_size;

    if (wx1 <= wx0) wx1 = wx0 + 1;
    if (wy1 <= wy0) wy1 = wy0 + 1;

    float sum = 0.0f;
    int count = 0;

    for (int sy = wy0; sy < wy1; sy++) {
        for (int sx = wx0; sx < wx1; sx++) {
            sum += frame[sy * frame_width + sx];
            count++;
        }
    }

    out[dy * dst_size + dx] = (sum / count) / 127.5f - 1.0f;
}
''', 'preprocess_face_kernel')