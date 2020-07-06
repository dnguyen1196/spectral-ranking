import os

# Data sets
# http://people.stern.nyu.edu/wgreene/Microeconometrics-GCEP.htm
# TODO: read the exercise description to see which data is required
from . import utils
from spectral.data.utils import from_csv

sfshop = from_csv(os.path.dirname(utils.__file__)+ "/dataset/SF/SFshop.csv", True)

sfwork = from_csv(os.path.dirname(utils.__file__) + "/dataset/SF/SFwork.csv", True)

youtube = from_csv(os.path.dirname(utils.__file__) + "/dataset/Youtube/Youtube.csv", False)

gif_anger = from_csv(os.path.dirname(utils.__file__) + "/dataset/GIF-anger/GIF-anger.csv", False)