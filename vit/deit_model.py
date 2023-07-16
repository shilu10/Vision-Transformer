from main import TFViTEmbeddings, TFVITTransformerBlock
from utils import * 
import tensorflow as tf 
import tensorflow.keras as keras 
from tensorflow.keras import * 
from tensorflow.keras.layers import * 
import numpy as np 
import os, sys, shutil
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from ml_collections import ConfigDict 


class ViTDistilled(keras.Model):
    """Vision Transformer base class."""

    def __init__(self, config: ConfigDict, **kwargs):
        super(ViTDistilled, self).__init__(**kwargs)
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

        # Other layers.
        self.dropout = Dropout(config.dropout_rate)
        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps)

        if self.config.include_top:
            self.head = Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="classification_head",
            )

            self.dist_head = Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="distillation_head",
            )

    def call(self, inputs, training=False):
        n = tf.shape(inputs)[0]

        # Create patches and project the patches.
        projected_patches = self.patch_embed(inputs)

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

        # when inference time, needing only the pre-norm value.
        if not self.config.include_top and not training:
          assert not training, "It only returns the pre-norm value, during inference time"
          return (representation[:, 0] + representation[:, 1]) / 2, attention_scores

        # Classification head.
        else:
            cls_output = self.head(representation[:, 0])
            dist_output = self.dist_head(representation[:, 1])

            # when training, we need both outputs(head and dist head output for training) 
            if training:
              return cls_output, dist_output, attention_scores 
            
            else:
              return (cls_output + dist_output) / 2, attention_scores