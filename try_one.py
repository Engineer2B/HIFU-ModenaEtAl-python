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
import math as math
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
import creating_STL


# ---------------------------------------------------
# maskable list
# ---------------------------------------------------
class MaskableList(list):
    def __getitem__(self, index):
        try:
            return super(MaskableList, self).__getitem__(index)
        except TypeError:
            return MaskableList(itool.compress(self, index))


# Aligning STL with transducers
focal_point = [-1, 0, 0]  # coordinates of focal point in centimeters
tumor_centre = [-3.3, 0, 0]  # coordinates of tumor centre in centimeters
# cortical_stl = 'solid_bone.stl'
# marrow_stl = 'marrow_bone.stl'
os.chdir("./Code_python")
stl_list = ['solid_bone.stl', 'marrow_bone.stl', 'tumor.stl']
# stl_list = ['CT-data_0.stl', 'CT-data_1.stl']
um = 'cm'  # unit of measure of STL file 'mm', 'cm' or 'm'
# 'creating_STL' moves the STL files to align the tumor centre with
# the focal point and generates new STL files saved
# as 'CT-data_<# of stl>.stl'
# new STLs are saved as 'CT-data_<# of stl>.stl'
creating_STL.move_stl(focal_point, tumor_centre, stl_list, um)

# ---------------------------------------------------
# definition of the topology:
# materials, geometry and trd elements
# ---------------------------------------------------

# '''


top = []  # the list containing all the objects topology
omega = 2*np.pi*1.2*1e6
# lossless:
material_1 = tp.Material('lossless', omega)
geometry_1 = tp.Geometry('plane', -16, -1000, -1000, 8, 2000, 2000, "")
lossles = tp.Topology(geometry_1, material_1)
top.append(lossles)
# muscle:
material_2 = tp.Material('muscle', omega)
geometry_2 = tp.Geometry('plane',  -8, -1000, -1000, 12, 2000, 2000, "")
muscl = tp.Topology(geometry_2, material_2)
# muscl.contains = 2
top.append(muscl)
# bone:
material_3 = tp.Material('bone', omega)
# geometry_3 = tp.Geometry('plane',  -1, -1000, -1000, 3, 2000, 2000)
# test_solid_py.stl cylinder_bone6.stl
geometry_3 = tp.Geometry('ct', -1, -1000, -1000, 4,
                         2000, 2000, "CT-data_0.stl")
bon = tp.Topology(geometry_3, material_3)
bon.is_contained_in = 1
# bon.contains = 2
top.append(bon)
# marrow:
material_4 = tp.Material('muscle', omega)
# test_marrow_py.stl cylinder_marrow6.stl
geometry_4 = tp.Geometry('ct', 0, -1000, -1000, 2, 2000, 2000, "CT-data_1.stl")
marr = tp.Topology(geometry_4, material_4)
marr.is_contained_in = 2
top.append(marr)
# tumor:
# material_5 = tp.Material('muscle', omega)
# geometry_5 = tp.Geometry('ct', -0.9, -1000, -1000, -0.6, 2000, 2000,
#  "CT-data_2.stl")
# tum = tp.Topology(geometry_5, material_5)
# tum.is_contained_in = 2
# top.append(tum)

top = MaskableList(top)
# TRd elements
trd = tp.Transducer('transducer_position.txt', 0.35)
radius = trd.r
n = 0  # n is the actual object of the ray
last_x = 2  # last_x end of the geometry
first_x = -8  # first_x starting geometry
# ---------------------------------------------------
# calculation of the power correction
# ---------------------------------------------------
RequiredTotalPower = 25  # Watt
ptrd_tot = 0
for o in range(len(top)):
    if top[o].material.name == 'lossless':
        index = o
k_r = top[index].material.k * radius
r_1 = 3.8317  # first zero Bessels
sin_theta_max = r_1 / (k_r*4)
theta_max = np.arcsin(sin_theta_max)
cos_theta_max = np.cos(theta_max)
AA = 1 - cos_theta_max

IntTot = integrate.quad(lambda x: np.sin(
    x) * (2 * special.j1(k_r * x) / (k_r * x))**2, 0.0000001, np.pi / 2)
IntRestr = integrate.quad(lambda x: np.sin(
    x) * (2 * special.j1(k_r * x) / (k_r * x))**2, 0.0000001, theta_max)
powerfr = IntRestr[0] / IntTot[0]

# ---------------------------------------------------
# creation cubes of the grid and saving the values in
# 3D matrices -> TO CHANGE
# ---------------------------------------------------
# index of the lossless
for o in range(len(top)):
    if top[o].material.name == 'lossless':
        index = o
Nrays = 1  # nrays per trd element
# creation of the lists of cubes
cubes = tp.creation_cubes(top, -2, 0, -0.3, 0.3, -0.3, 0.3, 0.02, 0.02, 0.02)
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

alpha = (np.array(cubes_1['material'].apply(
    lambda v: v.alpha).values.reshape([Nx, Ny, Nz])))
alphaL = (np.array(cubes_1['material'].apply(
    lambda v: v.alphaL).values.reshape([Nx, Ny, Nz])))
alphaS = (np.array(cubes_1['material'].apply(
    lambda v: v.alphaS).values.reshape([Nx, Ny, Nz])))
kS = (np.array(cubes_1['material'].apply(
    lambda v: v.kS).values.reshape([Nx, Ny, Nz])))
kL = (np.array(cubes_1['material'].apply(
    lambda v: v.kL).values.reshape([Nx, Ny, Nz])))
k = (np.array(cubes_1['material'].apply(
    lambda v: v.k).values.reshape([Nx, Ny, Nz])))
type_mat = (np.array(cubes_1['material'].apply(
    lambda v: v.type).values.reshape([Nx, Ny, Nz])))
Const7 = (np.array(cubes_1['material'].apply(
    lambda v: v.Const7).values.reshape([Nx, Ny, Nz])))
Const4 = (np.array(cubes_1['material'].apply(
    lambda v: v.Const4).values.reshape([Nx, Ny, Nz])))
Const6 = (np.array(cubes_1['material'].apply(
    lambda v: v.Const6).values.reshape([Nx, Ny, Nz])))
Const3 = (np.array(cubes_1['material'].apply(
    lambda v: v.Const3).values.reshape([Nx, Ny, Nz])))
z = (np.array(cubes_1['material'].apply(
    lambda v: v.z).values.reshape([Nx, Ny, Nz])))
#
#
# # ---------------------------------------------------
# # initialization variables
# # ---------------------------------------------------
vl_1 = 0
vl_2 = 0
vl_3 = 0
eps_1_2 = 0
eps_1_3 = 0
eps_2_3 = 0
pres = 0


# start_time = time.time()

# file_start = open('starting_points_STL.txt', 'w')
# file_end = open('end_points_STL.txt', 'w')
# file_index = open('index_rays_STL.txt', 'w')
# file_shear = open('shear_STL.txt', 'w')
# Ntrd = len(trd.coord)
Ntrd = 256

initial_time = time.time()
for i in range(0, Ntrd):
    print("# trd: ", i)
    # start_time = time.time()
    ray_prova, ptrd = gr.generation_rays_lossless(
        trd, top, Nrays, last_x, first_x, i, AA, k_r, index)
    # print("Generation_rays_lossless: " +
    # str(time.time()-start_time) + " seconds")
    # start_time = time.time()
    ray_tot = gr.Generation_ray(top, ray_prova, last_x, first_x)
    # print("Generation_ray: " + str(time.time() - start_time) + " seconds")
    if len(ray_tot) > 0:
        # start_time = time.time()
        vl1, vl2, vl3, eps12, eps13, eps23, pressure_one =
        clc_int.Splitting_the_groups(
            ray_tot, alpha, alphaS, alphaL, type_mat, Const7, Const4, Const6,
            Const3, k, kL, kS, xmin, xmax, ymin, ymax, zmin, zmax, xxb, yyb,
            zzb, dx, dy, dz, Nx, Ny, Nz, z, top)
        # print("Splitting_the_groups: " + str(time.time()
        # - start_time) + " seconds")
        vl_1 = vl_1 + vl1
        vl_2 = vl_2 + vl2
        vl_3 = vl_3 + vl3
        eps_1_2 = eps_1_2 + eps12
        eps_1_3 = eps_1_3 + eps13
        eps_2_3 = eps_2_3 + eps23
        pres = pres + pressure_one
        ptrd_tot = ptrd_tot + ptrd

#     for j in range(0, len(ray_prova)):
#         start_trd = np.real(ray_prova[j].start)
#         end_trd = np.real(ray_prova[j].end)
#         # vray_trd = np.real(ray_prova[j].vray)
#         start_trd = start_trd.tolist()
#         end_trd = end_trd.tolist()
#         # vray_trd = vray_trd.tolist()
#         start_trd = ' '.join(str(e) for e in start_trd)
#         end_trd = ' '.join(str(e) for e in end_trd)
#         # vray_trd = ' '.join(str(e) for e in vray_trd)
#         index_ray = ray_prova[j].obj_index
#         file_start.write(start_trd + '\n')
#         file_end.write(end_trd + '\n')
#         file_index.write(str(index_ray) + '\n')
#         # file_vray.write(vray_trd + '\n')
#         if ray_prova[j].shear is True:
#             file_shear.write(str(1) + '\n')
#         else:
#             file_shear.write(str(0) + '\n')
#     for j in range(0, len(ray_tot)):
#         startstart = np.real(ray_tot[j].start)
#         startstart = startstart.tolist()
#         endend = np.real(ray_tot[j].end)
#         endend = endend.tolist()
#         # vrayvray = np.real(ray_tot[j].vray)
#         # vrayvray = vrayvray.tolist()
#         startstart = ' '.join(str(e) for e in startstart)
#         endend = ' '.join(str(e) for e in endend)
#         # vrayvray = ' '.join(str(e) for e in vrayvray)
#         index_ray = ray_tot[j].obj_index
#         file_start.write(startstart + '\n')
#         file_end.write(endend + '\n')
#         file_index.write(str(index_ray) + '\n')
#         # file_vray.write(vrayvray + '\n')
#         if ray_tot[j].shear is True:
#             file_shear.write(str(1) + '\n')
#         else:
#             file_shear.write(str(0) + '\n')
# file_start.close()
# file_end.close()
# file_index.close()
# file_shear.close()

print("Total time: %s seconds " % (time.time() - initial_time))
# ==========================================================================================
# Calculation of the powerloss
# ==========================================================================================


phiv1 = np.angle(vl_1)
phiv2 = np.angle(vl_2)
phiv3 = np.angle(vl_3)


xi = (np.array(cubes_1['material'].apply(
    lambda v: v.xi).values.reshape([Nx, Ny, Nz])))
eta = (np.array(cubes_1['material'].apply(
    lambda v: v.eta).values.reshape([Nx, Ny, Nz])))

Q_bone = (xi+4*eta/3)*(np.abs(vl_1)**2+np.abs(vl_2)**2+np.abs(vl_3)**2)/2 + \
    eta * (np.abs(eps_1_2)**2 + np.abs(eps_1_3)**2 + np.abs(eps_2_3)**2)/2 + \
    (xi-2*eta/3) * (np.abs(vl_1)*np.abs(vl_2)*np.cos(phiv1-phiv2) +
                    np.abs(vl_1)*np.abs(vl_3)*np.cos(phiv1-phiv3) +
                    np.abs(vl_2)*np.abs(vl_3)*np.cos(phiv2-phiv3))


z = (np.array(cubes_1['material'].apply(
    lambda v: v.z).values.reshape([Nx, Ny, Nz])))
Q_soft = 2 * alpha*((np.abs(pres**2))/(2*z))

# Corrections
PowerCorrectionFactor = powerfr*RequiredTotalPower/ptrd_tot

Q_bone = Q_bone * PowerCorrectionFactor * 1e-6
Q_soft = Q_soft * PowerCorrectionFactor
Qtot = Q_bone + Q_soft

# ========= Print On a txt file ========== #


# print("--- %s seconds 1 ---" % (time.time() - start_time))
# print("ray_prova.start: ", ray_prova[0].start)
# print("ray_prova.end: ", ray_prova[0].end)
# print("ray_prova.vray: ", ray_prova[0].vray)
# print("ray_tot.start: ", ray_tot[2].start)
# print("ray_tot.vray: ", ray_tot[2].vray)
# print("Qtot: ", Qtot)
# ==========================================================================================
# Plot the powerloss
# ==========================================================================================

xx = np.array(cubes_1.iloc[0]['xx'])  # cubes center
yy = np.array(cubes_1.iloc[0]['yy'])
zz = np.array(cubes_1.iloc[0]['zz'])
zmin = cubes_1.iloc[0]['zmin']

# Q3D = np.array(Qtot.values.reshape([Nx,Ny,Nz]))

A = Qtot.max()
z0 = 0
f = np.array(np.abs(zz-z0))  # convert in an array (Phyton Bug)
iz = f.argmin()
# iz = 4
B = (Qtot[:, :, iz]).max()
Powerz = np.array(np.squeeze(Qtot[:, :, iz]))
Powerz = Powerz.transpose()


X, Y = np.meshgrid(xx, yy)
fig = plt.figure()
ax = fig.gca(projection='3d')
# Plot the surface.
surf = ax.plot_surface(X, Y, Powerz, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# surf = ax.plot_surface(X, Y, Powerz)

# Customize the z axis.
# ax.set_zlim(-1.01, 1.01)
# ax.zaxis.set_major_locator(LinearLocator(10))
# ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

print("--- End of Plot ---")

xx_old = np.copy(xx)
yy_old = np.copy(yy)
zz_old = np.copy(zz)

x_start = xx[0]
y_start = yy[0]
z_start = zz[0]

x_end = xx[-1]
y_end = yy[-1]
z_end = zz[-1]


x_add_minus = np.arange(x_start-0.5, x_start-0.03, 0.03)
y_add_minus = np.arange(y_start-0.5, y_start-0.03, 0.03)
z_add_minus = np.arange(z_start-0.5, z_start-0.03, 0.03)
x_add_plus = np.arange(x_end+0.03, x_end+0.5, 0.03)
y_add_plus = np.arange(y_end+0.03, y_end+0.5, 0.03)
z_add_plus = np.arange(z_end+0.03, z_end+0.5, 0.03)

xx = np.concatenate((x_add_minus, xx), axis=0)
xx = np.concatenate((xx, x_add_plus), axis=0)
yy = np.concatenate((y_add_minus, yy), axis=0)
yy = np.concatenate((yy, y_add_plus), axis=0)
zz = np.concatenate((z_add_minus, zz), axis=0)
zz = np.concatenate((zz, z_add_plus), axis=0)
# Save in a file
f = open('Q_to_write.txt', 'w')
g = open('coord_to_write.txt', 'w')
mu = open('mu.txt', 'w')
rho = open('rho.txt', 'w')
cp = open('cp.txt', 'w')

tlt = len(xx)*len(yy)*len(zz)

g.write('%d' % tlt + '\n')
sz = -1
start_time = time.time()
print('Generating the txt')
for i in range(0, len(xx)):
    for j in range(0, len(yy)):
        for k in range(0, len(zz)):
            if (xx[i] < x_start) or (yy[j] < y_start) or
            (zz[k] < z_start) or (xx[i] > x_end) or
            (yy[j] > y_end) or (zz[k] > z_end):
                    f.write('%f' % 0 + '\n')
                    g.write('%f' % xx[i] + ' ' + '%f' %
                            yy[j] + ' ' + '%f' % zz[k] + '\n')
                    material = tp.check_material(xx[i], yy[j], zz[k], top)
                    mu.write('%f' % material.kt + '\n')
                    rho.write('%f' % material.rho + '\n')
                    cp.write('%f' % material.cp + '\n')
            else:
                ind_x_old = np.where(xx_old == xx[i])
                ind_y_old = np.where(yy_old == yy[j])
                ind_z_old = np.where(zz_old == zz[k])
                f.write('%f' % Qtot[ind_x_old, ind_y_old, ind_z_old] + '\n')
                g.write('%f' % xx[i] + ' ' + '%f' %
                        yy[j] + ' ' + '%f' % zz[k] + '\n')
                material = tp.check_material(xx[i], yy[j], zz[k], top)
                mu.write('%f' % material.kt + '\n')
                rho.write('%f' % material.rho + '\n')
                cp.write('%f' % material.cp + '\n')

f.close()
g.close()
mu.close()
rho.close()
cp.close()

# os.system('cd C:/Users/20180781/PycharmProjects/Try_Phy_Revolution')
# os.system('FreeFem++-cli TEST_Daniela_faster.edp')
# print("--- %s seconds 2 ---" % (time.time() - start_time))
#
# # '''
#
#
# Atot = 0
# Btot = 0
# Ctot = 0
# def f(x,y,z):
#     if x==1:
#         A = np.zeros((10, 10, 10))
#         B = np.zeros((10, 10, 10))
#         C = np.zeros((10, 10, 10))
#     else:
#         A = np.ones((10, 10, 10))
#         B = np.ones((10, 10, 10))
#         C = np.ones((10, 10, 10))
#     return A,B,C
#
# for i in range(0,200):
#     A,B,C = f(i,i,i)
#     Atot = Atot + A
#     Btot = Btot + B
#     Ctot = Ctot + C
#
# #print(ray_prova, "ray_prova")
# #print(ray_tot, "ray_tot")
# #print(ptrd, "ptrd")
# #print(eps_1_2, "eps_1_2")
# #print(eps_1_3, "eps_1_3")
# #print(eps_2_3, "eps_2_3")
# #print(vl_1, "v1_1")
# #print(vl_2, "v1_2")
# #print(vl_3, "v1_3")
# #print(xi, "xi")
# #print(Q_bone, "Q_bone")
# #print(Q_soft, "Q_soft")
# #print(Qtot, "Qtot")
