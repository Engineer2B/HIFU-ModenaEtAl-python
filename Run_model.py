import numpy as np
import os
import scipy as sc
from scipy import special
from ray import Ray
import create_topology as tp
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
import calc_intensity as clc_int
import copy as copy
import time
import scipy.integrate as integrate
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
# from joblib import Parallel, delayed
import multiprocessing
from ray import Ray as r
from numpy import unravel_index


# ---------------------------------------------------
# maskable list
# ---------------------------------------------------
class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(itool.compress(self, index))

# ---------------------------------------------------
# definition of the topology:
# materials, geometry and trd elements
# ---------------------------------------------------
top = [] # the list containing all the objects topology
omega = 2*np.pi*1.2*1e6
material_1 = tp.Material('lossless',omega)
geometry_1 = tp.Geometry('plane', -16, -1000, -1000, 10, 2000, 2000)
lossles = tp.Topology(geometry_1,material_1)
top.append(lossles)
material_2 = tp.Material('muscle',omega)
geometry_2 = tp.Geometry('plane',  -6, -1000, -1000, 10, 2000, 2000)
muscl = tp.Topology(geometry_2,material_2)
top.append(muscl)
#material_3 = tp.Material('bone', omega)
#geometry_3 = tp.Geometry('plane',  -1, -1000, -1000, 3, 2000, 2000)
#bon = tp.Topology(geometry_3,material_3)
#top.append(bon)
top = MaskableList(top)
# TRd elements
trd = tp.Transducer('transducer_position.txt', 0.35)
radius = trd.r
n = 0 # n is the actual object of the ray
last_x = 4 # last_x end of the geometry
first_x = -16 # first_x starting geometry
# ---------------------------------------------------
# calculation of the power correction
# ---------------------------------------------------
RequiredTotalPower = 250 # Watt
ptrd_tot = 0
for o in range(len(top)):
    if top[o].material.name == 'lossless':
        index = o
k_r = top[index].material.k * radius
r_1 = 3.8317 # first zero Bessels
sin_theta_max = r_1 / (k_r*4)
theta_max = np.arcsin(sin_theta_max)
cos_theta_max = np.cos(theta_max)
AA = 1 - cos_theta_max

IntTot = integrate.quad(lambda x: np.sin(x) * (2* special.j1(k_r * x) / (k_r * x) )**2, 0.00000001, np.pi / 2)
IntRestr = integrate.quad(lambda x: np.sin(x) * (2* special.j1(k_r * x) / (k_r * x) )**2, 0.00000001, theta_max)
powerfr = IntRestr[0] / IntTot[0]

# ---------------------------------------------------
# creation cubes of the grid and saving the values in
# 3D matrices -> TO CHANGE
# ---------------------------------------------------
# index of the lossless
for o in range(len(top)):
    if top[o].material.name == 'lossless':
        index = o
Nrays = 20000# nrays per trd element
# creation of the lists of cubes
cubes = tp.creation_cubes(top, -1-0.5, 0, -0.3, 0.3, -0.3, 0.3, 0.02, 0.02, 0.02)
cubes_1 = pd.DataFrame([vars(s) for s in cubes])
cubes_1 = cubes_1.astype('object')

xmin = cubes_1.at[0, 'xmin']
xmax = cubes_1.at[0, 'xmax']
ymin = cubes_1.at[0, 'ymin']
ymax = cubes_1.at[0, 'ymax']
zmin = cubes_1.at[0, 'zmin']
zmax = cubes_1.at[0, 'zmax']
xxb = cubes_1.at[0, 'xxb']
yyb = cubes_1.at[0, 'yyb']
zzb = cubes_1.at[0, 'zzb']
dx = cubes_1.at[0, 'dx']
dy = cubes_1.at[0, 'dy']
dz = cubes_1.at[0, 'dz']
Nx = cubes_1.at[0, 'Nx']
Ny = cubes_1.at[0, 'Ny']
Nz = cubes_1.at[0, 'Nz']

alpha = (np.array(cubes_1['material'].apply(lambda v: v.alpha).values.reshape([Nx,Ny,Nz])))
alphaL = (np.array(cubes_1['material'].apply(lambda v: v.alphaL).values.reshape([Nx,Ny,Nz])))
alphaS = (np.array(cubes_1['material'].apply(lambda v: v.alphaS).values.reshape([Nx,Ny,Nz])))
kS = (np.array(cubes_1['material'].apply(lambda v: v.kS).values.reshape([Nx,Ny,Nz])))
kL = (np.array(cubes_1['material'].apply(lambda v: v.kL).values.reshape([Nx,Ny,Nz])))
k = (np.array(cubes_1['material'].apply(lambda v: v.k).values.reshape([Nx,Ny,Nz])))
type_mat = (np.array(cubes_1['material'].apply(lambda v: v.type).values.reshape([Nx,Ny,Nz])))
Const7 = (np.array(cubes_1['material'].apply(lambda v: v.Const7).values.reshape([Nx,Ny,Nz])))
Const4 = (np.array(cubes_1['material'].apply(lambda v: v.Const4).values.reshape([Nx,Ny,Nz])))
Const6 = (np.array(cubes_1['material'].apply(lambda v: v.Const6).values.reshape([Nx,Ny,Nz])))
Const3 = (np.array(cubes_1['material'].apply(lambda v: v.Const3).values.reshape([Nx,Ny,Nz])))
z = (np.array(cubes_1['material'].apply(lambda v: v.z).values.reshape([Nx,Ny,Nz])))

# ---------------------------------------------------
# initialization variables
# ---------------------------------------------------
vl_1 = 0
vl_2 = 0
vl_3 = 0
eps_1_2 = 0
eps_1_3 = 0
eps_2_3 = 0
pres = 0


start_time = time.time()
Ntrd = len(trd.coord)
#Ntrd = 130
for i in range(0, Ntrd):
    ray_prova, ptrd = gr.generation_rays_lossless(trd, top, Nrays, last_x, first_x, i, AA, k_r,index)
    ray_tot = gr.Generation_ray(top, ray_prova, last_x, first_x)
    ptrd_tot = ptrd_tot + ptrd
    if len(ray_tot) > 0 :
        vl1, vl2, vl3, eps12, eps13, eps23, pressure_one = clc_int.Splitting_the_groups(ray_tot, alpha, alphaS, alphaL, type_mat, Const7,
                                                                                              Const4, Const6,Const3, k, kL, kS,
                                                                                              xmin, xmax, ymin, ymax, zmin, zmax, xxb,yyb,zzb, dx, dy,dz, Nx, Ny, Nz, z)
        vl_1 = vl_1 + vl1
        vl_2 = vl_2 + vl2
        vl_3 = vl_3 + vl3
        eps_1_2 = eps_1_2 + eps12
        eps_1_3 = eps_1_3 + eps13
        eps_2_3 = eps_2_3 + eps23
        pres = pres + pressure_one




    print(i)
# ==========================================================================================
# Calculation of the powerloss
# ==========================================================================================

phiv1 = np.angle(vl_1)
phiv2 = np.angle(vl_2)
phiv3 = np.angle(vl_3)


xi = (np.array(cubes_1['material'].apply(lambda v: v.xi).values.reshape([Nx,Ny,Nz])))
eta = (np.array(cubes_1['material'].apply(lambda v: v.eta).values.reshape([Nx,Ny,Nz])))

Q_bone = (xi+4*eta/3)*(np.abs(vl_1)**2+np.abs(vl_2)**2+np.abs(vl_3)**2)/2 + \
  eta* (np.abs(eps_1_2)**2 + np.abs(eps_1_3)**2 +np.abs(eps_2_3)**2)/2 + \
(xi-2*eta/3) *(np.abs(vl_1)*np.abs(vl_2)*np.cos(phiv1-phiv2)+ \
 np.abs(vl_1)*np.abs(vl_3)*np.cos(phiv1-phiv3)+np.abs(vl_2)*np.abs(vl_3)*np.cos(phiv2-phiv3))

z = (np.array(cubes_1['material'].apply(lambda v: v.z).values.reshape([Nx,Ny,Nz])))
Q_soft = 2* alpha*((np.abs(pres**2))/(2*z))

# Corrections
PowerCorrectionFactor = powerfr*RequiredTotalPower/ptrd_tot

Q_bone = Q_bone * PowerCorrectionFactor * 1e-6
Q_soft = Q_soft * PowerCorrectionFactor
Qtot = Q_bone + Q_soft
print(PowerCorrectionFactor)
print(Qtot[11,15,29])

# ========= Print On a txt file ========== #


print("--- %s seconds ---" % (time.time() - start_time))

# ==========================================================================================
# Plot the powerloss
# ==========================================================================================

xx = np.array(cubes_1.iloc[0]['xx'])  # cubes center
yy = np.array(cubes_1.iloc[0]['yy'])
zz = np.array(cubes_1.iloc[0]['zz'])
zmin = cubes_1.iloc[0]['zmin']

# Q3D = np.array(Qtot.values.reshape([Nx,Ny,Nz]))

A = Qtot.max()
print(A)
z0 = 0
f = np.array(np.abs(zz-z0)) # convert in an array (Phyton Bug)
iz = f.argmin()
# iz = 4
B = (Qtot[:,:,iz]).max()
Powerz = np.array(np.squeeze(Qtot[:,:,iz]))
Powerz = Powerz.transpose()


X, Y = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, Powerz, cmap=cm.coolwarm,linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Powerz)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

