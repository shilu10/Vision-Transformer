import numpy as np 
import yaml 
import os, sys, shutil, glob
from imutils import paths 
import numpy as np 
import pandas as pd 
from .convert import port 

all_model_types = [
    "deit_tiny_patch16_224",
    "deit_small_patch16_224",
    "deit_base_patch16_384",
    "deit_base_patch16_224",
    "deit_tiny_distilled_patch16_224",
    "deit_small_distilled_patch16_224",
    "deit_base_distilled_patch16_224",
    "deit_base_distilled_patch16_384"
]

def port_all(mtype="classifier"):
    if mtype == "classifier":
        for model_type in all_model_types:
            print("Processing model type: ", model_type)
            port(
                model_type = model_type,
                model_savepath = ".",
                include_top = True
            )
    
    elif mtype == "feature_extractor":
        for model_type in all_model_types:
            print("Processing model type: ", model_type)
            port(
                model_type = model_type,
                model_savepath = ".",
                include_top = False
            )
