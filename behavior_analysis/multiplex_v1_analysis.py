import numpy as np
import pandas as pd
import os


file_path = "G:/My Drive/Work/MSc Neuroscience/Moshe Parnas/Experiments/Serotonergic system/5ht_behavior/operant_conditioning/raw_data/behavior/13.8.24/5ht_rnai/32471/20240813_083500_Log.txt"

df = pd.read_csv(file_path, sep=r'\s+')

df.to_csv('filename.csv')

