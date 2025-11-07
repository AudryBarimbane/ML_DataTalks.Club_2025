#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# load_pipeline.py

import pickle

def load_pipeline(model_version: str = "v1"):
    """
    Charge le pipeline Pickle depuis le fichier pipeline_v1.bin
    et retourne l'objet modèle.
    """
    filename = f"pipeline_{model_version}.bin"
    with open(filename, "rb") as f_in:
        pipeline = pickle.load(f_in)
    return pipeline

