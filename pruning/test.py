import os
import torch
import argparse
import warnings
import numpy as np
warnings.filterwarnings('ignore')
with warnings.catch_warnings():
    warnings.simplefilter('ignore')

def numParams(model):
        return sum([len(w.view(-1)) for w in model.parameters()])


