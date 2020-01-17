# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of multiheaded attention and self-attention layers."""


import tensorflow as tf


class Dense3D(tf.keras.layers.Layer):
    """A Dense Layer using 3D kernel with tf.einsum implementation.
  Attributes:
    num_attention_heads: An integer, number of attention heads for each
      multihead attention layer.
    size_per_head: An integer, hidden size per attention head.
    hidden_size: An integer, dimension of the hidden layer.
    kernel_initializer: An initializer for the kernel weight.
    bias_initializer: An initializer for the bias.
    activation: An activation function to use. If nothing is specified, no
      activation is applied.
    use_bias: A bool, whether the layer uses a bias.
    output_projection: A bool, whether the Dense3D layer is used for output
      linear projection.
    backward_compatible: A bool, whether the variables shape are compatible
      with checkpoints converted from TF 1.x.
  """

    def __init__(
        self,
        num_attention_heads=12,
        size_per_head=72,
        kernel_initializer=None,
        bias_initializer="zeros",
        activation=None,
        use_bias=True,
        output_projection=False,
        backward_compatible=False,
        **kwargs
    ):
        """Inits Dense3D."""
        super(Dense3D, self).__init__(**kwargs)
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.hidden_size = num_attention_heads * size_per_head
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.activation = activation
        self.use_bias = use_bias
        self.output_projection = output_projection
        self.backward_compatible = backward_compatible

    @property
    def compatible_kernel_shape(self):
        if self.output_projection:
            return [self.hidden_size, self.hidden_size]
        return [self.last_dim, self.hidden_size]

    @property
    def compatible_bias_shape(self):
        return [self.hidden_size]

    @property
    def kernel_shape(self):
        if self.output_projection:
            return [self.num_attention_heads, self.size_per_head, self.size_per_head]
        return [self.last_dim, self.num_attention_heads, self.size_per_head]

    @property
    def bias_shape(self):
        if self.output_projection:
            return [self.size_per_head]
        return [self.num_attention_heads, self.size_per_head]

    def build(self, input_shape):
        """Implements build() for the layer."""
        dtype = tf.as_dtype(self.dtype or tf.keras.backend.floatx())
        if not (dtype.is_floating or dtype.is_complex):
            raise TypeError(
                "Unable to build `Dense3D` layer with non-floating "
                "point (and non-complex) dtype %s" % (dtype,)
            )
        input_shape = tf.TensorShape(input_shape)
        if tf.compat.dimension_value(input_shape[-1]) is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense3D` "
                "should be defined. Found `None`."
            )
        self.last_dim = tf.compat.dimension_value(input_shape[-1])
        self.input_spec = tf.keras.layers.InputSpec(
            min_ndim=3, axes={-1: self.last_dim}
        )
        # Determines variable shapes.
        if self.backward_compatible:
            kernel_shape = self.compatible_kernel_shape
            bias_shape = self.compatible_bias_shape
        else:
            kernel_shape = self.kernel_shape
            bias_shape = self.bias_shape

        print("kernel", kernel_shape)

        self.kernel = self.add_weight(
            "kernel",
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            dtype=self.dtype,
            trainable=True,
        )
        if self.use_bias:
            self.bias = self.add_weight(
                "bias",
                shape=bias_shape,
                initializer=self.bias_initializer,
                dtype=self.dtype,
                trainable=True,
            )
        else:
            self.bias = None
        super(Dense3D, self).build(input_shape)

    def call(self, inputs):
        """Implements ``call()`` for Dense3D.
    Args:
      inputs: A float tensor of shape [batch_size, sequence_length, hidden_size]
        when output_projection is False, otherwise a float tensor of shape
        [batch_size, sequence_length, num_heads, dim_per_head].
    Returns:
      The projected tensor with shape [batch_size, sequence_length, num_heads,
        dim_per_head] when output_projection is False, otherwise [batch_size,
        sequence_length, hidden_size].
    """
        if self.backward_compatible:
            kernel = tf.keras.backend.reshape(self.kernel, self.kernel_shape)
            bias = (
                tf.keras.backend.reshape(self.bias, self.bias_shape)
                if self.use_bias
                else None
            )
        else:
            kernel = self.kernel
            bias = self.bias

        if self.output_projection:
            ret = tf.einsum("abcd,cde->abe", inputs, kernel)
        else:
            ret = tf.einsum("abc,cde->abde", inputs, kernel)
        if self.use_bias:
            ret += bias
        if self.activation is not None:
            return self.activation(ret)
        return ret


class MultiHeadAttention(tf.keras.layers.Layer):
    """Multi-headed attention layer."""

    def __init__(self, size_per_head, num_heads, dropout_rate=0.0):
        """Initialize Attention.

    Args:
      size_per_head: int, output dim of hidden layer.
      num_heads: int, number of heads to repeat the same attention structure.
      dropout_rate: float, dropout rate inside attention for training.
    """

        super().__init__()
        self.size_per_head = size_per_head
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        """Builds the layer."""
        # Layers for linearly projecting the queries, keys, and values.

        if self.dropout_rate:
            self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

        self.query_dense_layer = Dense3D(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="query",
        )
        self.key_dense_layer = Dense3D(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="key",
        )
        self.value_dense_layer = Dense3D(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            name="value",
        )
        self.output_dense_layer = Dense3D(
            self.num_heads,
            self.size_per_head,
            kernel_initializer="glorot_uniform",
            use_bias=False,
            output_projection=True,
            name="output_transform",
        )
        super().build(input_shape)

    def get_config(self):
        return {
            "size_per_head": self.size_per_head,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate,
        }

    def call(
        self, inputs, training=None,
    ):
        """Apply attention mechanism to query_input and source_input.

    Args:
      query_input: A tensor with shape [batch_size, length_query, hidden_size].
      source_input: A tensor with shape [batch_size, length_source,
        hidden_size]

    Returns:
      Attention layer output with shape [batch_size, length_query, hidden_size]
    """
        # Linearly project the query, key and value using different learned
        # projections. Splitting heads is automatically done during the linear
        # projections --> [batch_size, length, num_heads, dim_per_head].
        query_input, source_input = inputs

        query = self.query_dense_layer(query_input)
        key = self.key_dense_layer(source_input)
        value = self.value_dense_layer(source_input)

        # Scale query to prevent the dot product between query and key from growing
        # too large.
        depth = float(self.size_per_head)

        query /= tf.sqrt(depth)

        # Calculate dot product attention
        logits = tf.einsum("BTNH,BFNH->BNFT", key, query)

        # Note that softmax internally performs math operations using float32
        # for numeric stability. When training with float16, we keep the input
        # and output in float16 for better performance.
        weights = tf.nn.softmax(logits, name="attention_weights")

        if self.dropout_rate:
            weights = self.dropout(weights, training=training)

        attention_output = tf.einsum("BNFT,BTNH->BFNH", weights, value)

        # Run the outputs through another linear projection layer. Recombining heads
        # is automatically done --> [batch_size, length, hidden_size]
        attention_output = self.output_dense_layer(attention_output)

        return attention_output


class MultiHeadSelfAttention(MultiHeadAttention):
    def call(self, inputs, **kwargs):
        return super().call([inputs, inputs], **kwargs)
