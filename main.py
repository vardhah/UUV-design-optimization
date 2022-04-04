import argparse
import os
import numpy as np
from utils import *
import pandas as pd



input_size=2                             # input size may change if integer/ordinal type variable and represented by one-hot encoding

num_iteration=20                        # Number of iteration of sampling
init_samples=50 
budget_samples=50                        # Number of samples-our budget
ranges=[-10,0,-6.5,0]                    # ranges in form of [low1,high1,low2,high2,...]

