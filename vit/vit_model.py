from .main import TFViTEmbeddings, TFVITTransformerBlock
from .utils import * 
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
import numpy as np 
import os, sys, shutil
from tensorflow.keras.layers import Conv2D, Dropout, LayerNormalization, Dense, Input, Add
from ml_collections import ConfigDict 


class ViTClassifier(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, config: ConfigDict, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Patch embed layer
        self.patch_embed = TFViTEmbeddings(config, name="patch_embedding")
        dpr = [x for x in tf.linspace(0.0, self.config.drop_path_rate, self.config.num_layers)]

        # transformer blocks
        transformer_blocks = [
            TFVITTransformerBlock(config, name=f"transformer_block_{i}", drop_prob=dpr[i])
            for i in range(config.num_layers)
        ]

        self.transformer_blocks = transformer_blocks

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        # Other layers.
        self.dropout = Dropout(config.dropout_rate)
        self.layer_norm = LayerNormalization(
            epsilon=config.layer_norm_eps
        )
        if self.config.include_top:
            self.head = layers.Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="classification_head",
            )

    def call(self, inputs, training=None):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.patch_embed(inputs)
        print(projected_patches.shape)

        encoded_patches = self.dropout(projected_patches)

        # Initialize a dictionary to store attention scores from each transformer
        # block.
        attention_scores = dict()

        # Iterate over the number of layers and stack up blocks of
        # Transformer.
        for transformer_module in self.transformer_blocks:
            # Add a Transformer block.
            encoded_patches, attention_score = transformer_module(
                encoded_patches,
                output_attentions = True
            )
            attention_scores[f"{transformer_module.name}_att"] = attention_score

        # Final layer normalization.
        representation = self.layer_norm(encoded_patches)

        # Pool representation.
        if self.config.classifier == "token":
            encoded_patches = representation[:, 0]
        elif self.config.classifier == "gap":
            encoded_patches = self.gap_layer(representation)

        if not self.config.include_top:
            return encoded_patches, attention_scores

        # Classification head.
        else:
            output = self.head(encoded_patches)
            return output, attention_scores