class TFViTPatchEmbeddings(keras.layers.Layer):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """
    def __init__(self, config: ConfigDict, **kwargs):
        super(TFViTPatchEmbeddings, self).__init__(**kwargs)
        image_size = config.image_size
        patch_size = config.patch_size
        projection_dim = config.projection_dim
        n_channels = config.n_channels

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = ((image_size[0] // patch_size[0]) * (image_size[1] // patch_size[1]))

        # calculation of num of patches
        self.num_patches = num_patches
        self.config = config
        self.image_size = image_size
        self.n_channels = n_channels
        self.projection_dim = projection_dim
        self.patch_size = patch_size

        # patch generator
        self.projection = Conv2D(
            kernel_size=patch_size,
            strides=patch_size,
            data_format="channels_last",
            filters=projection_dim,
            padding="valid",
            use_bias=True,
            kernel_initializer=get_initializer(self.config.initializer_range),
            bias_initializer="zeros",
            name="projection"
        )

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        shape = tf.shape(x)
        batch_size, height, width, n_channel = shape[0], shape[1], shape[2], shape[3]

        projection = self.projection(x)
        embeddings = tf.reshape(tensor=projection, shape=(batch_size, self.num_patches, -1))

        return embeddings


# position embed
class TFViTEmbeddings(keras.layers.Layer):
    """
    Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, config: ConfigDict, **kwargs):
        super(TFViTEmbeddings, self).__init__(**kwargs)

        self.patch_embeddings = TFViTPatchEmbeddings(config, name="patch_embedding")
        self.dropout = Dropout(rate=config.hidden_dropout_prob)
        self.config = config

    def build(self, input_shape: tf.TensorShape):
        num_patches = self.patch_embeddings.num_patches
        self.cls_token = self.add_weight(
            shape=(1, 1, self.config.projection_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="cls_token",
        )
        self.position_embeddings = self.add_weight(
            shape=(1, num_patches + 1, self.config.projection_dim),
            initializer=get_initializer(self.config.initializer_range),
            trainable=True,
            name="position_embeddings",
        )

        super().build(input_shape)

    def call(self, x, training=False):
        shape = tf.shape(x)
        batch_size, height, width, n_channels = shape[0], shape[1], shape[2], shape[3]

        patch_embeddings = self.patch_embeddings(x, training)

        # repeating the class token for n batch size
        cls_tokens = tf.repeat(self.cls_token,
                               repeats=batch_size,
                               axis=0)
        # adding the [CLS] token to patch_embeeding
        patch_embeddings = tf.concat([cls_tokens, patch_embeddings], axis=1)

        # adding positional embedding to patch_embeddings
        patch_embeddings = patch_embeddings + self.position_embeddings
        patch_embeddings = self.dropout(patch_embeddings, training=training)

        return patch_embeddings
