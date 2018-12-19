import itertools as itool
import numpy as np


class MaskableList(list):
    def __getitem__(self, index):
        try: return super(MaskableList, self).__getitem__(index)
        except TypeError: return MaskableList(itool.compress(self, index))


# intersection_plane only used in the first plane lossless-muscle
# x is always > ray.start[0]
# vray[0] is always >= 0
def intersection_plane(start,vray,x):
    if vray[0] != 0:
        divx = 1 / vray[0]
        tmin = (x - start[0])*divx
        nn = ([-1, 0, 0])
        point = start + tmin * vray
    else:
        nn = ([-1, 0, 0])
        tmin= x - start[0]
        point = ([start[0]+tmin, start[1], start[2]])

    return nn, point



# smiths !!! to use in the big intersection for planes

def inter(start, vray, boundx, boundy, boundz):
# for i in range 0

    divx = 1/vray[0]
    if divx > 0:
        tmin = (boundx[0] - start[0])*divx
        tmax = (boundx[1] - start[0])*divx

    else:
        tmin = (boundx[1] - start[0])*divx
        tmax = (boundx[0] - start[0])*divx
# y direction
    if vray[1]!=0:
        divy= 1/vray[1]
        if divy > 0:
            tminy = (boundy[0] - start[1])*divy
            tmaxy = (boundy[1] - start[1])*divy
        else:
            tminy = (boundy[1] - start[1])*divy
            tmaxy = (boundy[0] - start[1])*divy
    else:
        tminy = float('-Inf')
        tmaxy = float('Inf')

    if tminy > tmin:
        tmin = tminy
    if tmaxy < tmax:
        tmax = tmaxy

    if tmin> tmaxy or tminy> tmax :
        string1 = 'no intersection'
        return string1

    if vray[2] != 0:
        divz= 1/vray[2]
        if divz>0:
            tminz = (boundz[0] - start[2])*divz
            tmaxz = (boundz[1] - start[2])*divz
        else:
            tminz = (boundz[1] - start[2])*divz
            tmaxz = (boundz[0] - start[2])*divz

    else:
        tminz= float('-Inf')
        tmaxz = float('Inf')

    if tmin > tmaxz or tminz > tmax:
        string1 = 'no intersection'
        return string1

    if tminz > tmin:
        tmin = tminz
    if tmaxy < tmax:
        tmax = tmaxz
    if tmin < 0.0001 and tmin > 0:
       tmin = 0
    if tmax < 0.0001 and tmax > 0:
       tmax = 0
    if tmin > - 0.0001 and tmin < 0:
       tmin = 0
    if tmax > - 0.0001 and tmax < 0:
       tmax = 0




    # return tmin<t1 and tmax>t0, tmin,tmax
    return tmin, tmax


def intersection(start, vray, top, actual_index, final_geometry, initial_geometry):

    possible_obj = [] # list with the possible object
    possible_int = []  # possible lambda
    possible_nn = []  # normal
    for o in top:  # check in all the object in topology
        if o.geometry.type == 'plane':  # if geometry is a plane

            int = inter(start, vray, o.geometry.x, o.geometry.y, o.geometry.z) # call int -> smith algorithm
            if int != 'no intersection':  # if there is an intersection
                if int[0] == 0 or int[1] == 0:  # if at least one lambda is zero -> the ray start in this wall
                    if int[0] > 0 or int[1] > 0:   # if at least one is positive -> the wall is not behind the ray
                        int = np.asarray(int)  # make it as an array
                        possible_obj.append(top.index(o))  # fill the list with the index of the object
                        possible_int.append(np.amax(int[int >= 0]))  # append the max of the lambda (only lambda >0)
                        if vray[0] > 0:
                            possible_nn.append(np.array([-1, 0,0]))
                        else:
                            possible_nn.append(np.array([1, 0, 0]))
                else:  # no lambda is zero -> maybe I am considering the next wall
                    if int[0] > 0 or int[1] > 0:  # at least one is more than zero
                        int = np.asarray(int)
                        possible_obj.append(top.index(o))
                        possible_int.append(np.amin(int[int >= 0]))  # this time I have to take the minimum intersection
                        if vray[0] > 0:
                            possible_nn.append(np.array([-1, 0, 0]))
                        else:
                            possible_nn.append(np.array([1, 0, 0]))

    possible_int = np.asarray(possible_int)

    if len(possible_int) != 0:  # if there are intersections
        index_min = np.where(np.asarray(possible_int) == np.asarray(possible_int).min()) # find indices(always >1)of the min lambdas
        rng = np.asmatrix(index_min).shape  # indices of the minimum
        lambda_int = possible_int[np.argmin(possible_int)]  # lambda_int:Not important obj belongs: take just min lambda
        point_x = start[0] + vray[0] * lambda_int  # the point of the intersection
        for i in range(0, rng[1]):  # Check in the index_min to find the right object!
            if possible_obj[index_min[0][i]] != actual_index:  # if the object is not == to the actual_index
                index_min_new = possible_obj[index_min[0][i]]  # we found the right object!
                possible_nn_new = possible_nn[index_min[0][i]]
            else:  # if it's the same index
                if top[actual_index].is_contained_in != []: index_min_new = top[actual_index].is_contained_in #if the object is contained in another object-> it's the other object
                if np.abs(point_x - final_geometry)<0.0001:  # if we are at the end of the geometry
                    index_min_new = actual_index  # actual_index is the right object
                    possible_nn_new = 'final'
                if np.abs(point_x - initial_geometry)<0.0001:  # if we are at the beginning of the geometry
                    index_min_new = actual_index  # actual_index is the right object
                    possible_nn_new = 'final'
    else:
        index_min_new = False
        lambda_int = False
    if lambda_int == False:
        index_min_new = False
        point = False
    else:
        point = start + vray * lambda_int

    if 'index_min_new' not in locals(): # bug to solve: with lambda_max really big, almost vertical rays, point_x is slightly different than final geometry
        index_min_new = actual_index  # actual_index is the right object
        possible_nn_new = 'final'
        print('buggy ray')


    return index_min_new, point, possible_nn_new, lambda_int #return the index of the object intersected and the lambda_int







