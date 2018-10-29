# Add this line to the beginning of relative.py file
import sys
sys.path.append('..')
from fastai.courses.dl1.fastai import *
from fastai.courses.dl1.fastai.transforms import *
from fastai.courses.dl1.fastai.conv_learner import *
from fastai.courses.dl1.fastai.model import *
from fastai.courses.dl1.fastai.dataset import *
from fastai.courses.dl1.fastai.sgdr import *
from fastai.courses.dl1.fastai.plots import *

PATH = "data/PokemonDigimon" # Path to dataset
sz=224 # Size of images
arch=resnet34 # Neural net
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms_from_model(arch, sz))
learn = ConvLearner.pretrained(arch, data, precompute=True)
learn.fit(0.01, 2)
tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=1.1)
data = ImageClassifierData.from_paths(PATH, val_name='test', tfms=tfms)
learn = ConvLearner.pretrained(arch, data, precompute=True) # Learning Model used to train our ResNet
learn.unfreeze() # Usually we only train the later layers of our neural net but by unfreezing the we train every layer
# This is good because it can increase the accurary of our model by allowing the earlier layers to learn
learn.fit(0.01, 2)
