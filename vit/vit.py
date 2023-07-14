import sys 
sys.path.append("..")
from tensorflow import keras 
import tensorflow as tf 
import numpy as np 
from ml_collections import ConfigDict 
import os, sys 
from tensorflow.keras import layers
from layers.mh_self_attention import * 
from layers.layerscale import LayerScale 
from layers.drop_path import DropPath 
from layers.mlp import MLP


class TFVITTransformerBlock(keras.Model):
    def __init__(self, config: ConfigDict, drop_prob, **kwargs):
        super(TFVITTransformerBlock, self).__init__(**kwargs)

        self.attention = TFViTAttention(config, name="attention")
        self.mlp = MLP(config, name="mlp_output")
        self.config = config

        self.layernorm_before = LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_before"
        )
        self.layernorm_after = LayerNormalization(
            epsilon=config.layer_norm_eps,
            name="layernorm_after"
        )

        self.drop_prob = drop_prob

    def call(
        self,
        hidden_states: tf.Tensor,
        head_mask: tf.Tensor = False,
        output_attentions: bool = False,
      #  drop_prob: float = 0.0,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:

        # first layernormalization
        x1 = self.layernorm_before(hidden_states)
        attention_outputs = self.attention(
                                input_tensor = x1,
                                head_mask = head_mask,
                                output_attentions = output_attentions,
                                training = training
                            )

        # attn outputs, attn scores
        if output_attentions:
            attention_output, attention_scores = attention_outputs[0], attention_outputs[1]

        else:
            attention_output = attention_outputs[0]

        attention_output = (
                        LayerScale(self.config)(attention_output)
                        if self.config.init_values
                        else attention_output)

        attention_output = (
                DropPath(self.drop_prob)(attention_output)
                if self.drop_prob
                else attention_output)

        # first residual connection
        x2 = Add()([attention_output + hidden_states])

        x3 = self.layernorm_after(x2)

        x4 = self.mlp(x3, training=training)
        x4 = LayerScale(self.config)(x4) if self.config.init_values else x4
        x4 = DropPath(self.drop_prob)(x4) if self.drop_prob else x4

        # second residual connection
        outputs = Add()([x2, x4])

        if output_attentions:
            return outputs, attention_scores

        return outputs


class ViTClassifier(keras.models.Model):
    def __init__(self, config: ConfigDict, **kwargs):
        super(ViTClassifier, self).__init__(**kwargs)

        self.patch_embeddings = TFViTEmbeddings(config, name="patch_embedding")

        # drop path / stochastic depth
        dpr = [x for x in tf.linspace(0.0, config.drop_path_rate, config.num_layers)]

        blocks = [
            TFVITTransformerBlock(config, drop_prob=dpr[i], name=f"transformer_block{i}")
            for i in range(config.num_layers)
        ]

        self.blocks = blocks
        self.layer_norm = LayerNormalization(epsilon=config.layer_norm_eps)

        if config.classifier == "gap":
            self.gap_layer = layers.GlobalAvgPool1D()

        if config.include_top:
            self.classification_head = Dense(
                config.num_classes,
                kernel_initializer="zeros",
                dtype="float32",
                name="classification_head",
            )

        self.config = config

    def call(self, x, training=False):

        # generate patch embeddings (N, num_patches, projection_dims)
        embeddings = self.patch_embeddings(x, training=training)

        # for storing the attention score at each transformer layer
        attention_scores = dict()
        for block in self.blocks:
            res = block(
                hidden_states=embeddings,
                output_attentions=self.config.return_attention_scores,
                training=training
            )
            if self.config.return_attention_scores:
                embeddings, attention_score = res[0], res[1]
            else:
                embeddings = res[0]

            attention_scores[f"{block.name}_attn_scores"] = attention_score

        # norm and head layer
        representation = self.layer_norm(embeddings)

        # Pool representation.
        if self.config.classifier == "token":
            embeddings = representation[:, 0]
        elif self.config.classifier == "gap":
            embeddings = self.gap_layer(representation)

        if not self.config.include_top:
            return embeddings, attention_scores

        # Classification head.
        else:
            output = self.classification_head(embeddings)
            return output, attention_scores
