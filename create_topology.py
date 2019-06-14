from __future__ import print_function
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import sys, vtk
sys.path.append('../')
import pyoctree
from pyoctree import pyoctree as ot
import time
import math
import itertools as itool
unique = np.unique
array = np.array


class Topology:
    def __init__(self, geometry=0, material=0):  # constructor
        self.geometry = geometry
        self.material = material
        self.contains = False
        self.is_contained_in = []


class Geometry:
    def __init__(self, type='', xmin=0, ymin=0, zmin=0, lx=0, ly=0, lz=0, filename=""):  # constructor
        self.type = type
        if self.type == 'plane':  # first type -> it's a plane
            self.x = array([xmin, xmin+lx])  # the coordinate the plane is in cm
            self.y = array([ymin, ymin+ly])
            self.z = array([zmin, zmin + lz])
            self.nn = array([-1, 0, 0])  # normal to the plane
            self.point = array([np.random.uniform(xmin+lx/100, xmin+lx-lx/100), np.random.uniform(ymin+ly/100, ymin+ly-ly/100), np.random.uniform(zmin +lz/100, zmin + lz -lz/100)])
            self.volume = lx * ly * lz
        if self.type == 'ct':  # if the type is ct, we have to import the STL file of CT data
            self.x = array([xmin, xmin + lx])  # the coordinate the plane is in cm
            self.y = array([ymin, ymin + ly])
            self.z = array([zmin, zmin + lz])
            self.nn = array([-1, 0, 0])  # normal to the plane
            self.point = array([np.random.uniform(xmin + lx / 100, xmin + lx - lx / 100),
                                np.random.uniform(ymin + ly / 100, ymin + ly - ly / 100),
                                np.random.uniform(zmin + lz / 100, zmin + lz - lz / 100)])
            self.volume = lx * ly * lz
            # CREATION PF THE OCTREE FROM THE STL FILE:
            # Read in stl file using vtk
            reader = vtk.vtkSTLReader()
            reader.SetFileName(filename)
            reader.MergingOn()
            reader.Update()
            stl = reader.GetOutput()
            # Extract polygon info from stl
            # 1. Get array of point coordinates
            numPoints = stl.GetNumberOfPoints()
            pointCoords = np.zeros((numPoints, 3), dtype=float)
            for i in range(numPoints):
                pointCoords[i, :] = stl.GetPoint(i)
            # 2. Get polygon connectivity
            numPolys = stl.GetNumberOfCells()
            connectivity = np.zeros((numPolys, 3), dtype=np.int32)
            for i in range(numPolys):
                atri = stl.GetCell(i)
                ids = atri.GetPointIds()
                for j in range(3):
                    connectivity[i, j] = ids.GetId(j)
            # Show format of pointCoords
            pointCoords
            # Show format of connectivity
            connectivity
            # Create octree structure containing stl poly mesh
            self.tree = ot.PyOctree(pointCoords, connectivity) # tree structure from the STL file


class Material:
    def __init__(self, name='', omega=0, type='', type_geom='', cL=0.0001, alphaL=0, rho=0.00001, kt=0, cp=0, cS=0.0001, alphaS=0):  # constructor
        self.name = name
        self.omega = omega
        self.type_geom=type_geom
        if self.name == 'lossless':
            self.c = 138000
            self.rho = 1030e-6
            self.z = self.c*self.rho
            self.k = self.omega/ self.c
            self.alpha = 0
            self.type = 'liquid'
        elif self.name == 'muscle':
            self.c = 153700
            self.rho = 1010e-6
            self.z = self.c * self.rho
            self.k = self.omega / self.c
            self.alpha = 0.0576
            self.type = 'liquid'
            self.alphaL = 0
            self.alphaS = 0
            self.Const2 = 0
            self.Const3 = 0
            self.Const4 = 0
            self.Const5 = 0
            self.Const6 = 0
            self.Const7 = 0
            self.kt=0.537
            self.cp=3720
            self.kS = 0
            self.kL = 0
            self.xi = 0
            self.eta = 0
        elif self.name == 'bone':
            self.cL = 373600
            self.rho = 2025e-6
            self.zL = self.cL * self.rho
            self.kL = self.omega / self.cL
            self.alphaL = 1.9
            self.cS = 199500
            self.zS = self.cS * self.rho
            self.kS = self.omega / self.cS
            self.alphaS = 2.8
            self.type = 'solid'
            alphaL = self.alphaL * 100
            alphaS = self.alphaS * 100
            rho = self.rho * 1e6
            cS = self.cS / 100
            cL = self.cL / 100
            kS = self.omega/cS
            kL = self.omega/cL
            mu, eta = FSolvePars(rho, self.omega, cS, alphaS)
            p1, p2 = FSolvePars(rho, self.omega, cL, alphaL)
            lambda_t =p1-2 * mu
            xi = p2-4 * eta / 3
            self.Const2 = omega * ((lambda_t +2 * mu) * kL + (xi+4 * eta / 3) * omega * alphaL) / 2
            self.Const3 = 1 / self.Const2;
            self.Const4 = (-1j * omega) * (1j * kL - alphaL)
            self.Const5 = (mu * omega * kS + eta * omega ** 2 * alphaS) / 2
            self.Const6 = 1 / self.Const5
            self.Const7 = (-1j * omega) * (1j * kS - alphaS)
            self.alpha = 0
            self.z = 0.00001
            self.xi = xi
            self.eta = eta
            self.k = 0
            self.cp = 3720
            self.kt = 0.487
        else:
            if self.type == 'liquid':
               self.c = cL
               self.rho = rho
               self.z = self.c * self.rho
               self.k = self.omega / self.c
               self.alpha = alphaL
               self.alphaL =0
               self.alphaS =0
               self.Const2 = 0
               self.Const3 = 0
               self.Const4 = 0
               self.Const5 = 0
               self.Const6 =0
               self.Const7 = 0
               self.xi = 0
               self.eta = 0
               self.kt = kt
               self.cp = cp
               self.kS = 0
               self.kL = 0
            else:
                self.cL = cL
                self.rho = rho
                self.zL = self.cL * self.rho
                self.kL = self.omega / self.cL
                self.alphaL = alphaL
                self.cS = cS
                self.zS = self.cS * self.rho
                self.kS = self.omega / self.cS
                self.alphaS = alphaS
                self.type = 'solid'
                self.alpha = 0
                mu, eta = FSolvePars(self.rho, self.omega, self.cS, self.alphaS)
                p1, p2 = FSolvePars(self.rho, self.omega, self.cL, self.alphaL)
                lambda_t = p1 - 2 * mu
                xi = p2 - 4 * eta / 3
                self.Const2 = omega * ((lambda_t + 2 * mu) * self.kL + (xi + 4 * eta / 3) * omega * alphaL) / 2
                self.Const3 = 1 / self.Const2;
                self.Const4 = (-1j * omega) * (1j * self.kL - alphaL)
                self.Const5 = (mu * omega * self.kS + eta * omega ** 2 * alphaS) / 2
                self.Const6 = 1 / self.Const5
                self.Const7 = (-1j * omega) * (1j * self.kS - alphaS)
                self.alpha = 0
                self.xi = xi
                self.eta = eta
                self.z = 0.001
                self.k = 0
                self.kt = kt
                self.cp = cp


class Transducer:
    def __init__(self, namefile=0, radius=0):  #constructor
        self.r = radius
        input = np.loadtxt(namefile)
        self.coord = input[:, 0:4]  # in m, x,y,z position e.g. trd.coord[3,0] trd 4 x coord or trd.coord[3,1] trd 4 y coord
        self.coord = np.column_stack((self.coord[:, 0], self.coord[:, 1]*100-14, self.coord[:, 2:4]*100))


def contains(top):

    for o in top:
        for h in top:
            if h != o:
                if o.geometry.type == 'plane' and h.geometry.type == 'plane':
                    if (h.geometry.x[0] < o.geometry.x[1] and h.geometry.x[0] > o.geometry.x[0]) and (
                            h.geometry.y[0] < o.geometry.y[1] and h.geometry.y[0] > o.geometry.y[0]) and (
                            h.geometry.z[0] < o.geometry.z[1] and h.geometry.z[0] > o.geometry.z[0]):
                        o.contains = True
                        A = h.is_contained_in
                        A.append(top.index(o))
                        h.is_contained_in = A
                    else:
                        o.contains = False
    for o in top:
        A = o.is_contained_in
        volumes=[]
        index=[]
        np.asanyarray(A)
        if A != []:
            for j in range(0,len(A)):
                volumes.append(top[A[j]].geometry.volume)
                index.append(A[j])
                o.is_contained_in = index[np.argmin(np.asarray(volumes))]


def creation_cubes(top, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz):
    Nx = int(np.floor((xmax - xmin) / dx) + 1)
    Ny = int(np.floor((ymax - ymin) / dy) + 1)
    Nz = int(np.floor((zmax - zmin) / dz) + 1)

    dx = (xmax - xmin) / Nx
    dy = (ymax - ymin) / Ny
    dz = (zmax - zmin) / Nz
    xxb = np.arange(xmin, xmax, dx)  # boundaries
    yyb = np.arange(ymin, ymax, dy)
    zzb = np.arange(zmin, zmax, dz)
    xx = np.arange(xmin + dx / 2, xmax, dx)  # centers
    yy = np.arange(ymin + dy / 2, ymax, dy)
    zz = np.arange(zmin + dz / 2, zmax, dz)

    size_cubes = len(xx)
    cubes = []
    for i in range(0, Nx):
        for l in range(0, Ny):
            for k in range(0, Nz):
                all_dist = []
                ct_top = []
                for j in range(0, len(top)):
                    if top[j].geometry.type == 'plane':
                        if xx[i]> top[j].geometry.x[0]\
                                and xx[i] < top[j].geometry.x[1]\
                                and  yy[l]> top[j].geometry.y[0]\
                                and yy[l] < top[j].geometry.y[1]\
                                and  zz[k]> top[j].geometry.z[0]\
                                and zz[k] < top[j].geometry.z[1]:
                            material_tmp = top[j].material
                            # material = top[j].material
                            # cb = cube(xx[i], yy[l], zz[k], material, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz, xxb, yyb, zzb,xx,yy,zz)
                            # cubes.append(cb)
                    else:
                        tree = top[j].geometry.tree
                        dist, bool = inside_outside_ct(tree, xx[i], yy[l], zz[k])
                        if bool is True:
                            all_dist.append(dist)
                            ct_top.append(j)
                if all_dist != []:
                    ind2 = all_dist.index(min(all_dist))
                    true_top = ct_top[ind2]
                    true_top = int(true_top)
                    material = top[true_top].material
                    cb = cube(xx[i], yy[l], zz[k], material, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy,
                              dz, xxb, yyb, zzb, xx, yy, zz)
                    cubes.append(cb)
                else:
                    material = material_tmp
                    cb = cube(xx[i], yy[l], zz[k], material, Nx, Ny, Nz, xmin, xmax, ymin, ymax, zmin, zmax, dx, dy, dz, xxb, yyb, zzb,xx,yy,zz)
                    cubes.append(cb)
    return cubes

class cube:
    def __init__(self, cx=0, cy=0, cz=0, material=0, Nx=0, Ny=0, Nz=0, xmin=0, xmax=0, ymin=0, ymax=0, zmin=0,
                 zmax=0, dx=0, dy=0, dz=0, xxb=0, yyb=0, zzb=0, xx=0,yy=0,zz=0, intensity=0, phase=0, nrays=0, pol=np.array([0,0,0]),
                 kl= np.array([0,0,0]), ks= np.array([0,0,0]), phase_long= 0, phase_shear= 0,
                 int_long=0, int_shear=0, nrays_l=0,nrays_s=0, pressure = 0):
        self.cx = cx  # centre
        self.cy = cy
        self.cz = cz
        self.material = material
        self.Nx = Nx
        self.Ny = Ny
        self.Nz = Nz
        self.xx = xx
        self.yy = yy
        self.zz = zz
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.xxb = xxb
        self.yyb = yyb
        self.zzb = zzb
        self.intensity = intensity
        self.phase = phase
        self.nrays = nrays
        self.pol = pol
        self.kl = kl
        self.ks = ks
        self.phase_long = phase_long
        self.phase_shear = phase_shear
        self.int_long = int_long
        self.int_shear = int_shear
        self.nrays_l = nrays_l
        self.nrays_s = nrays_s
        self.pressure = pressure

def FSolvePars(rho, omega, speed ,alpha):
    k = omega/speed
    C = omega**2*rho/(alpha**2+ k**2)
    D = np.sqrt(2)*C/(speed * np.sqrt(rho))
    p_1 = D**2-C
    p_2 = np.sqrt(C**2-p_1**2)/omega
    return p_1, p_2

def check_material(x,y,z, top):
    for j in range(0, len(top)):
        if top[j].geometry.type == 'plane':
            if x > top[j].geometry.x[0] \
                    and x < top[j].geometry.x[1] \
                    and y > top[j].geometry.y[0] \
                    and y < top[j].geometry.y[1] \
                    and z > top[j].geometry.z[0] \
                    and z < top[j].geometry.z[1]:
                material = top[j].material
    return material

def inside_outside_ct(tree, xx, yy, zz):
    m = 50  # magnitude of vector to calculate destination point
    start = [xx, yy, zz]
    # start = [0.99, 0.29, -0.29]
    vray = [0.5, 0.5, 0.5]
    vray = [1, 0, 0]
    destination = [start[0] + vray[0] * m, start[1] + vray[1] * m, start[2] + vray[2] * m]  # destination point
    destination = np.real(destination)
    rayPointList = np.array([[start, destination]], dtype=np.float32)
    ray = rayPointList[0]
    bool = False
    inside_dist = []
    triLabelList = [i.triLabel for i in tree.rayIntersection(ray)]
    triLabelList
    triList = [tree.polyList[i] for i in triLabelList]
    triList
    face_int = []
    dist = []
    sgn = []  # true=+ false=-
    for i in tree.rayIntersection(ray):
        face_int.append(i.triLabel)
        dist.append(np.abs(i.s))
        if i.s < 0:
            sgn.append(False)
        else:
            sgn.append(True)
    rayVect = ray[1] - ray[0]
    rayVect /= np.linalg.norm(rayVect)
    rayVect
    if dist != []:
        ind = dist.index(min(dist))
        nearest_is = sgn[ind]  # true=+ false=-
        for tri in triList:
            if tri.label == face_int[ind]:
                normal = tri.N  # normal of nearest triangle
        lambda_bone = min(dist)  # smallest lambda (distance between origin and intersection point)
        if nearest_is is True:
            if np.dot(normal, rayVect) > 0:
                # print(str(k) + ': The point is INSIDE')
                inside_dist = lambda_bone
                bool = True
                # num_top = j
        else:
            if np.dot(normal, rayVect) < 0:  # if dot product between calculated normal and ray is positive means that
                # the ray is coming from inside the bone, so the normal has to be inverted
                # (because normal are always pointing outside
                # print(str(k) + ': The point is INSIDE')
                inside_dist = lambda_bone
                bool = True
                # num_top = j
            # else:
            #     print(str(k) + ': The point is OUTSIDE')
            # print(lambda_bone)
    # else:
        # print(str(k) + ': No intersection, so the point is OUTSIDE')
    return inside_dist, bool
