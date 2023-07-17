import numpy as np 
import tensorflow as tf 
from tensorflow.keras import * 
from tensorflow import keras 
from tensorflow.keras.layers import * 
import os, sys, shutil, glob 
import timm 
from .utils import modify_tf_block, get_tf_qkv
from .deit.deit_model import ViTDistilled
from .deit.vit_model import ViTClassifier
from .deit import main
from .base_config import get_baseconfig
import yaml

def port(model_type, model_savepath, include_top):

    print("Instantiating PyTorch model...")
    pt_model = timm.create_model(
        model_name=model_type, 
        num_classes=1000, 
        pretrained=True
    )

    if "distilled" in model_type:
        assert (
            "dist_token" in pt_model.state_dict()
        ), "Distillation token must be present for models trained with distillation."
    pt_model.eval()

    print("Instantiating TF model...")
    model_cls = ViTDistilled if "distilled" in model_type else ViTClassifier

    config_file_path = f"configs/{model_type}.yaml"
    with open(config_file_path, "r") as f:
        data = yaml.safe_load(f)

    config = get_baseconfig(
        model_type = model_type,
        image_size = data.get("image_size"),
        patch_size = data.get("patch_size"),
        num_heads = data.get("num_heads"),
        num_layers = data.get("num_layers"),
        projection_dim = data.get("projection_dim"),
        include_top = include_top
    )

    tf_model = model_cls(config)

    #print(tf_model.positional_embedding, tf_model.cls_token)

    dummy_inputs = tf.ones((2, 224, 224, 3))
    _ = tf_model(dummy_inputs)[0]

    if include_top:
        assert tf_model.count_params() == sum(
            p.numel() for p in pt_model.parameters()
        )

    # Load the PT params.
    pt_model_dict = pt_model.state_dict()
    pt_model_dict = {k: pt_model_dict[k].numpy() for k in pt_model_dict}

    print("Beginning parameter porting process...")

    # Projection layers.
    tf_model.layers[0].patch_embeddings.projection = modify_tf_block(
        tf_model.layers[0].patch_embeddings.projection,
        pt_model_dict["patch_embed.proj.weight"],
        pt_model_dict["patch_embed.proj.bias"],
    )

    # Positional embedding.
    tf_model.layers[0].position_embeddings.assign(
        tf.Variable(pt_model_dict["pos_embed"])
    )

    # CLS and (optional) Distillation tokens.
    # Distillation token won't be present in the models trained without distillation.
    tf_model.layers[0].cls_token.assign(tf.Variable(pt_model_dict["cls_token"]))
    if "distilled" in model_type:
        tf_model.layers[0].dist_token.assign(tf.Variable(pt_model_dict["dist_token"]))

    # Layer norm layers.
    ln_idx = -3 if "distilled" in model_type else -2
    tf_model.layers[ln_idx] = modify_tf_block(
        tf_model.layers[ln_idx],
        pt_model_dict["norm.weight"],
        pt_model_dict["norm.bias"],
    )

    # Head layers.
    if include_top:
        head_layer = tf_model.get_layer("classification_head")
        head_layer_idx = -2 if "distilled" in  model_type else -1
        tf_model.layers[head_layer_idx] = modify_tf_block(
            head_layer,
            pt_model_dict["head.weight"],
            pt_model_dict["head.bias"],
        )
        if "distilled" in  model_type:
            head_dist_layer = tf_model.get_layer("distillation_head")
            tf_model.layers[-1] = modify_tf_block(
                head_dist_layer,
                pt_model_dict["head_dist.weight"],
                pt_model_dict["head_dist.bias"],
            )

    # Transformer blocks.
    idx = 0

    for outer_layer in tf_model.layers:
        if (
            isinstance(outer_layer, tf.keras.Model)
            and outer_layer.name != "projection"
        ):
            tf_block = tf_model.get_layer(outer_layer.name)
            pt_block_name = f"blocks.{idx}"

            # LayerNorm layers.
            layer_norm_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.LayerNormalization):
                    layer_norm_pt_prefix = (
                        f"{pt_block_name}.norm{layer_norm_idx}"
                    )
                    layer.gamma.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.weight"]
                        )
                    )
                    layer.beta.assign(
                        tf.Variable(
                            pt_model_dict[f"{layer_norm_pt_prefix}.bias"]
                        )
                    )
                    layer_norm_idx += 1

            # FFN layers.
            ffn_layer_idx = 1
            for layer in tf_block.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    dense_layer_pt_prefix = (
                        f"{pt_block_name}.mlp.fc{ffn_layer_idx}"
                    )
                    layer = modify_tf_block(
                        layer,
                        pt_model_dict[f"{dense_layer_pt_prefix}.weight"],
                        pt_model_dict[f"{dense_layer_pt_prefix}.bias"],
                    )
                    ffn_layer_idx += 1


            # Attention layer.
            for layer in tf_block.layers:
                (q_w, k_w, v_w), (q_b, k_b, v_b) = get_tf_qkv(
                    f"{pt_block_name}.attn",
                    pt_model_dict,
                    config,
                )

                if isinstance(layer, main.TFViTAttention):
                    # Key
                    layer.self_attention.key = modify_tf_block(
                        layer.self_attention.key,
                        k_w,
                        k_b,
                        is_attn=True,
                    )
                    # Query
                    layer.self_attention.query = modify_tf_block(
                        layer.self_attention.query,
                        q_w,
                        q_b,
                        is_attn=True,
                    )
                    # Value
                    layer.self_attention.value = modify_tf_block(
                        layer.self_attention.value,
                        v_w,
                        v_b,
                        is_attn=True,
                    )
                    # Final dense projection
                    layer.dense_output.dense = modify_tf_block(
                        layer.dense_output.dense,
                        pt_model_dict[f"{pt_block_name}.attn.proj.weight"],
                        pt_model_dict[f"{pt_block_name}.attn.proj.bias"],
                    )

            for layer in tf_block.layers:

              if isinstance(layer, tf.keras.Sequential):
                d_indx = 1
                for indx, inner_layer in enumerate(layer.layers):
                  if len(layer.layers) >= 2 and isinstance(inner_layer, Dense):
                    inner_layer = modify_tf_block(
                            inner_layer,
                            pt_model_dict[f"{pt_block_name}.mlp.fc{d_indx}.weight"],
                            pt_model_dict[f"{pt_block_name}.mlp.fc{d_indx}.bias"],
                        )
                    d_indx += 1

            idx += 1

    print("Porting successful, serializing TensorFlow model...")

    save_path = os.path.join(model_savepath, model_type)
    save_path = f"{save_path}_fe" if not include_top else save_path
    tf_model.save(save_path)
    print(f"TensorFlow model serialized at: {save_path}...")
