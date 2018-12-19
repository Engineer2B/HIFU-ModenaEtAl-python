import numpy as np
from ray import Ray
import pandas as pd
import copy as copy
import cmath as cmath
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import cm

from numpy.linalg import norm


def cal_norm(v1, v2, v3, r, one_matrix):
    v1 = v1 / (np.maximum(r, one_matrix))
    v2 = v2 / (np.maximum(r, one_matrix))
    v3 = v3 / (np.maximum(r, one_matrix))

    vlx2 = v1 ** 2
    vly2 = v2 ** 2
    vlz2 = v3 ** 2

    vnorm = vlx2 + vly2 + vlz2
    vnorm = (np.maximum(vnorm, one_matrix))
    vnorm = np.sqrt(vnorm)

    v1 = v1 / vnorm
    v2 = v2 / vnorm
    v3 = v3 / vnorm

    return v1, v2, v3

# Splitting the groups is called 256 times (Ntrd times)
#
def Splitting_the_groups(ray_tot, alpha, alphaS, alphaL, type_mat, Const7, Const4, Const6, Const3, k,
                         kL, kS, xmin, xmax, ymin, ymax, zmin, zmax, xxb,yyb,zzb, dx, dy,dz, Nx,Ny,Nz,z):
    # convert in a data_frame
    ray_tot_data_frame = pd.DataFrame([vars(s) for s in ray_tot])
    # convert in tuple ->iteration! It's a vector
    # print(ray_tot_data_frame.shape)
    ray_tot_data_frame['path'] = ray_tot_data_frame['path'].apply(lambda x: tuple(x))
    # cut at the first decimal to avoid too many groups!
    #ray_tot_data_frame['phase_shift'] = round(ray_tot_data_frame['phase_shift'], 1)
    # convert in tuple -> nb float not iterable
    #ray_tot_data_frame['phase_shift'] = tuple(ray_tot_data_frame['phase_shift'])
    # use groupby to group basing on path and phase_shif
    Ray_group = ray_tot_data_frame.groupby(['path']).count()
    Ray_group = Ray_group.reset_index()
    # cubes vuoto
    eps12 = 0
    eps13 = 0
    eps23 = 0
    pressure = 0

    vl2 = 0
    vl3 = 0

    vl1 = np.zeros((Nx, Ny, Nz))
    one_matrix = np.ones((Nx, Ny, Nz))
    DD = len(Ray_group)
    for i in range(0, len(Ray_group)):
        # groups : based on on the path

        grp = ray_tot_data_frame[(ray_tot_data_frame.ix[:, 'path'] ==
                                  Ray_group['path'].iat[i])]

        int_long, int_shear, nrays_s, nrays_l, phase_shear, phase_long, pol1, pol2, pol3, ks1, ks2, ks3, kl1, kl2, kl3, phase, intensit, nrays = \
            intensity(grp, alpha, alphaS, alphaL, type_mat, k, kL, kS, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, xxb,yyb,zzb, dx, dy,dz)  # here adding eh

        # soft tissue
        phase = phase / (np.maximum(nrays, one_matrix))
        pressure_one = np.sqrt(intensit * 2 * z) * np.exp(1j * phase)
        # including interference
        pressure = pressure + pressure_one
        # bone
        phase_long = phase_long / (np.maximum(nrays_l, one_matrix))
        phase_shear = phase_shear / (np.maximum(nrays_s, one_matrix))
        kl1, kl2, kl3 = cal_norm(kl1, kl2, kl3, nrays_l, one_matrix)
        ks1, ks2, ks3 = cal_norm(ks1, ks2, ks3, nrays_s, one_matrix)
        pol1, pol2, pol3 = cal_norm(pol1, pol2, pol3, nrays_s, one_matrix)

        B1 = np.sqrt(Const6 * (int_shear * 1e4))
        A1 = np.sqrt(Const3 * (int_long * 1e4))

        # including interference


        vl1 = vl1 + A1 * Const4 * np.exp(1j * phase_long) * (kl1 ** 2) + B1 * Const7 * np.exp(1j * phase_shear) * (
                ks1 * pol1)
        vl2 = vl2 + A1 * Const4 * np.exp(1j * phase_long) * (kl2 ** 2) + B1 * Const7 * np.exp(1j * phase_shear) * (
                ks2 * pol2)
        vl3 = vl3 + A1 * Const4 * np.exp(1j * phase_long) * (kl3 ** 2) + B1 * Const7 * np.exp(1j * phase_shear) * (
                ks3 * pol3)
        eps12 = eps12 + A1 * Const4 * np.exp(1j * phase_long) * 2 * kl1 * kl2 + B1 * Const7 * np.exp(
            1j * phase_shear) * (ks1 * pol2 + ks2 * pol1)
        eps13 = eps13 + A1 * Const4 * np.exp(1j * phase_long) * 2 * kl1 * kl3 + B1 * Const7 * np.exp(
            1j * phase_shear) * (ks1 * pol3 + ks3 * pol1)
        eps23 = eps23 + A1 * Const4 * np.exp(1j * phase_long) * 2 * kl2 * kl3 + B1 * Const7 * np.exp(
            1j * phase_shear) * (ks2 * pol3 + ks3 * pol2)

    return vl1, vl2, vl3, eps12, eps13, eps23, pressure


def intensity(rays, alpha, alphaS, alphaL,type_mat, k, kL, kS, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, xxb,yyb,zzb, dx, dy,dz):
    # cb_1 = copy.copy(cb)
    # here only the rays with the same phase

    # 3D matrices and other values

    kl1 = np.zeros((Nx, Ny, Nz))
    kl2 = np.zeros((Nx, Ny, Nz))
    kl3 = np.zeros((Nx, Ny, Nz))
    ks1 = np.zeros((Nx, Ny, Nz))
    ks2 = np.zeros((Nx, Ny, Nz))
    ks3 = np.zeros((Nx, Ny, Nz))
    pol1 = np.zeros((Nx, Ny, Nz))
    pol2 = np.zeros((Nx, Ny, Nz))
    pol3 = np.zeros((Nx, Ny, Nz))
    nrays_l = np.zeros((Nx, Ny, Nz))
    phase_long = np.zeros((Nx, Ny, Nz))
    int_long = np.zeros((Nx, Ny, Nz))
    nrays_s = np.zeros((Nx, Ny, Nz))
    phase_shear = np.zeros((Nx, Ny, Nz))
    int_shear = np.zeros((Nx, Ny, Nz))
    phase = np.zeros((Nx, Ny, Nz))
    intensity = np.zeros((Nx, Ny, Nz))
    nrays = np.zeros((Nx, Ny, Nz))

    array = np.array

    for ri in range(0, len(rays)):
        start = rays['start'].iat[ri]
        vray = rays['vray'].iat[ri]
        end = rays['end'].iat[ri]
        if start[0] < xmin:
            if vray[0] > 0:
                lambda_x = (xxb - start[0]) / vray[0]
            else:
                lambda_x = array([float('Inf')])

        elif start[0] > xmax:
            if vray[0] < 0:
                lambda_x = (xxb - start[0]) / vray[0]
            else:
                lambda_x = array([float('Inf')])

            if end[0] > xmax:
                lambda_x = array([float('Inf')])
        else:
            if vray[0] > 0:
                lambda_x = (xxb - start[0]) / vray[0]
                lambda_x = lambda_x[lambda_x >= 0]
                lambda_x = np.hstack([0, lambda_x])
            elif vray[0] < 0:
                lambda_x = (xxb - start[0]) / vray[0]
                lambda_x = lambda_x[lambda_x >= 0]
                lambda_x = np.flip(lambda_x, 0)
                lambda_x = np.hstack([0, lambda_x])
            else:
                lambda_x = array([0])

        if lambda_x[0] != float('Inf'):
            if start[1] < ymin:

                if vray[1] > 0:
                    lambda_y = (yyb - start[1]) / vray[1]
                else:
                    lambda_y = array([float('Inf')])

            elif start[1] > ymax:
                if vray[1] < 0:
                    lambda_y = (yyb - start[1]) / vray[1]
                else:
                    lambda_y = np.array([float('Inf')])

                if end[1] > ymax:
                    lambda_y = array([float('Inf')])
            else:
                if vray[1] > 0:
                    lambda_y = (yyb - start[1]) / vray[1]
                    lambda_y = lambda_y[lambda_y >= 0]
                    lambda_y = np.hstack([0, lambda_y])
                elif vray[1] < 0:
                    lambda_y = (yyb - start[1]) / vray[1]
                    lambda_y = lambda_y[lambda_y >= 0]
                    lambda_y = np.flip(lambda_y, 0)
                    lambda_y = np.hstack([0, lambda_y])
                else:
                    lambda_y = array([0])
        else:
            lambda_y = array([float('Inf')])
        if lambda_x[0] != float('Inf') and lambda_y[0] != float('Inf'):
            if start[2] < zmin:
                if vray[2] > 0:
                    lambda_z = (zzb - start[2]) / vray[2]
                else:
                    lambda_z = array([float('Inf')])

            elif start[2] > zmax:
                if vray[2] < 0:
                    lambda_z = (zzb - start[2]) / vray[2]
                    lambda_z = np.flip(lambda_z, 0)
                else:
                    lambda_z = array([float('Inf')])

                if end[2] > zmax:
                    lambda_z = array([float('Inf')])
            else:
                if vray[2] > 0:
                    lambda_z = (zzb - start[2]) / vray[2]
                    lambda_z = lambda_z[lambda_z >= 0]
                    lambda_z = np.hstack([0, lambda_z])
                elif vray[2] < 0:
                    lambda_z = (zzb - start[2]) / vray[2]
                    lambda_z = lambda_z[lambda_z >= 0]
                    lambda_z = np.flip(lambda_z, 0)
                    lambda_z = np.hstack([0, lambda_z])
                else:
                    lambda_z = array([0])
        else:
            lambda_z = array([float('Inf')])

        min_lambda = array([lambda_x[0], lambda_y[0], lambda_z[0]]).max(0)
        max_lambda = array([lambda_x[-1], lambda_y[-1], lambda_z[-1]]).min(0)

        if min_lambda < max_lambda:
            lambda_x = lambda_x[min_lambda <= lambda_x]
            lambda_x_restr = lambda_x[lambda_x <= max_lambda]
            lambda_y = lambda_y[min_lambda <= lambda_y]
            lambda_y_restr = lambda_y[lambda_y <= max_lambda]
            lambda_z = lambda_z[min_lambda <= lambda_z]
            lambda_z_restr = lambda_z[lambda_z <= max_lambda]
            lambda_int = np.unique(np.sort(array(np.hstack((lambda_x_restr, lambda_y_restr, lambda_z_restr)))))

            for n in range(0, len(lambda_int) - 1):
                lambda_1 = lambda_int[n]
                lambda_2 = lambda_int[n + 1]
                lambda_12 = (lambda_1 + lambda_2) * (0.5)
                P_in = start + lambda_1 * vray
                P_out = start + lambda_2 * vray
                ind = np.floor((start + lambda_12 * vray - array([xmin, ymin, zmin])) / (array([dx, dy, dz])))

                if vray[0] > 0:  # ray towards positive x
                    if P_out[0] > end[0]:  # ray ends in the cube
                        lambda_2 = (end - start).dot(vray)
                        lambda_12 = (lambda_1 + lambda_2) / 2

                else:  # ray towards negative x
                    if P_out[0] > end[0]:  # ray ends in the cube
                        lambda_2 = (end - start).dot(vray)
                        lambda_12 = (lambda_1 + lambda_2) / 2

                if ((vray[0] > 0 and P_in[0] <= end[0]) or
                        (vray[0] < 0 and P_in[0] >= end[0])):

                    # case soft tissue
                    index_cube = [[int(float(ind[0]))], [int(float(ind[1]))], [int(float(ind[2]))]]
                    '''
                    # to pass from a 3D matrix position to a 2D array! 
                    index_cube_1 = np.ravel_multi_index(
                        [[int(float(ind[0]))], [int(float(ind[1]))], [int(float(ind[2]))]],
                        (int(float(Nx)), int(float(Ny)), int(float(Nz))))

                    index_cube_1 = int(float(index_cube_1))
                    '''
                    if type_mat[index_cube[0], index_cube[1], index_cube[2]] == 'liquid':

                        intensity[index_cube[0], index_cube[1], index_cube[2]] = intensity[index_cube[0], index_cube[1],
                                                                                           index_cube[2]] \
                                                                                 + (rays['I0'].iat[ri]) * (
                            np.exp(-2 * alpha[index_cube[0], index_cube[1], index_cube[2]] * lambda_12)) * (
                                                                                         (np.abs(
                                                                                             lambda_1 - lambda_2)) / (
                                                                                                     dx * dy * dz))
                        phase[index_cube[0], index_cube[1], index_cube[2]] = phase[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + \
                                                                             k[index_cube[0], index_cube[1], index_cube[
                                                                                 2]] * lambda_12 + \
                                                                             rays['phase_initial'].iat[ri]
                        nrays[index_cube[0], index_cube[1], index_cube[2]] = nrays[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + 1

                    else:  # shear or longitudinal ray

                        if rays['shear'].iat[ri] == 0:

                            kl1[index_cube[0], index_cube[1], index_cube[2]] = kl1[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[0]
                            kl2[index_cube[0], index_cube[1], index_cube[2]] = kl2[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[1]
                            kl3[index_cube[0], index_cube[1], index_cube[2]] = kl3[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[2]
                            phase_long[index_cube[0], index_cube[1], index_cube[2]] = phase_long[
                                                                                          index_cube[0], index_cube[1],
                                                                                          index_cube[2]] + \
                                                                                      kL[index_cube[0], index_cube[1],
                                                                                         index_cube[2]] * lambda_12 + \
                                                                                      rays['phase_initial'].iat[ri]
                            nrays_l[index_cube[0], index_cube[1], index_cube[2]] = nrays_l[index_cube[0], index_cube[1],
                                                                                           index_cube[2]] + 1
                            int_long[index_cube[0], index_cube[1], index_cube[2]] = \
                                int_long[index_cube[0], index_cube[1], index_cube[2]] + \
                                (rays['I0'].iat[ri] * (
                                    np.exp(-2 * alphaL[index_cube[0], index_cube[1], index_cube[2]] * lambda_12))) * \
                                ((np.abs(lambda_1 - lambda_2)) / (dx * dy * dz))


                        else:

                            pol = rays['poldir'].iat[ri]
                            ks1[index_cube[0], index_cube[1], index_cube[2]] = ks1[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[0]
                            ks2[index_cube[0], index_cube[1], index_cube[2]] = ks2[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[1]
                            ks3[index_cube[0], index_cube[1], index_cube[2]] = ks3[index_cube[0], index_cube[1],
                                                                                   index_cube[2]] + vray[2]
                            pol1[index_cube[0], index_cube[1], index_cube[2]] = pol1[index_cube[0], index_cube[1],
                                                                                     index_cube[2]] + pol[0]
                            pol2[index_cube[0], index_cube[1], index_cube[2]] = pol2[index_cube[0], index_cube[1],
                                                                                     index_cube[2]] + pol[1]
                            pol3[index_cube[0], index_cube[1], index_cube[2]] = pol3[index_cube[0], index_cube[1],
                                                                                     index_cube[2]] + pol[2]
                            phase_shear[index_cube[0], index_cube[1], index_cube[2]] = phase_shear[
                                                                                           index_cube[0], index_cube[1],
                                                                                           index_cube[2]] + \
                                                                                       kS[index_cube[0], index_cube[1],
                                                                                          index_cube[2]] * lambda_12 + \
                                                                                       rays['phase_initial'].iat[ri]
                            nrays_s[index_cube[0], index_cube[1], index_cube[2]] = nrays_s[index_cube[0], index_cube[1],
                                                                                           index_cube[2]] + 1
                            int_shear[index_cube[0], index_cube[1], index_cube[2]] = \
                                int_shear[index_cube[0], index_cube[1], index_cube[2]] + \
                                (rays['I0'].iat[ri] * (
                                    np.exp(-2 * alphaS[index_cube[0], index_cube[1], index_cube[2]] * lambda_12))) * \
                                ((np.abs(lambda_1 - lambda_2)) / (dx * dy * dz))

    return int_long, int_shear, nrays_s, nrays_l, phase_shear, phase_long, pol1, pol2, pol3, ks1, ks2, ks3, kl1, kl2, kl3, phase, intensity, nrays
