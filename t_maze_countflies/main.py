
# Imports
from functions import *
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


# If we want to work with a saved image, use this approuch
currentDirectory = os.getcwd()
imgPath = os.path.join(currentDirectory, 't_maze_countflies/source_img.png')

imagePrepare(imgPath)
w