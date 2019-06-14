import numpy as np
import os
import scipy as sc
from scipy import special
from ray import Ray
import itertools as itool
from numpy import linalg as LAi
from calc_intensity import intensity
from intersection import inter
from intersection import intersection
import generation_rays as gr
import reflection_refraction as rf
import time
import pandas as pd
import cmath as cmath
import math as math
import calc_intensity as clc_int
import copy as copy
import time
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from joblib import Parallel, delayed
import multiprocessing
from ray import Ray as r
from numpy import unravel_index
import creating_STL
import create_topology as tp