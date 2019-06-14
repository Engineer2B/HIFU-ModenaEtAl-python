import itertools as itool
import numpy as np
from numpy.linalg import norm
import math

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

def inter_ct(start, vray, tree):
    start = np.real(start)
    m = 50  # magnitude of vector to calculate destination point
    destination = [start[0] + vray[0]*m, start[1] + vray[1]*m, start[2] + vray[2]*m]  # destination point
    destination = np.real(destination)
    rayPointList = np.array([[start, destination]], dtype=np.float32)
    # Get intersection points for a single ray
    ray = rayPointList[0]
    inversion = False
    # Get list of intersected triangles
    triLabelList = [i.triLabel for i in tree.rayIntersection(ray)]
    triLabelList
    if triLabelList == []:  # if it is empty, no intersection occurred
        int_point_bingo = False
        normal = False
        lambda_ct = False

    else:  # if triLabelList is NOT empty, intersection(s) occurred
        # Get tris
        triList = [tree.polyList[i] for i in triLabelList]
        triList
        face_int = []
        dist = []
        one_dist_is_zero = False
        one_dist_is_neg = False
        normal_zero = []
        for i in tree.rayIntersection(ray):
            # print('Intersected tri = %d,' % i.triLabel, 'Intersection coords = [%.2f, %.2f, %.2f]' % tuple(i.p),
                  # ',  Parametric. dist. along ray = %.2f' % i.s)
            if i.s < -0.001:
                one_dist_is_neg = True
            elif i.s > -0.001 and i.s < 0.001:  # if one distance is near-zero means that the ray origin is on the bone surface,
                                        # so it's intersection doesn't have to be taken in account.
                one_dist_is_zero = True
            elif i.s > 0.001:  # no distance is near-zero or negative, so ray origin is NOT on (or near) the bone surface
                face_int.append(i.triLabel)
                dist.append(i.s)
        #if one_dist_is_neg is True and one_dist_is_zero is True:
            #int_point_bingo = False
            #normal = False
            #lambda_ct = False
        if dist == []:
            int_point_bingo = False
            normal = False
            lambda_ct = False
        else:
            # Convert ray from start/end points into unit vector

            rayVect = ray[1] - ray[0]
            rayVect /= norm(rayVect)
            rayVect
            # STL could be intersected more than one time: now we need to find the nearest intersection point:
            ind = dist.index(min(dist))
            # face_int[ind] is the number of nearest intersected traingle
            # print(face_int[ind])

            for tri in triList:
                if tri.label == face_int[ind]:
                    normal = tri.N  # normal of nearest triangle
            # angle = math.degrees(math.atan2(np.linalg.norm(np.cross(normal, ray[1] - ray[0])), np.dot(normal, ray[1] - ray[0])))
            # print("#######")
            # print("Normal: ", normal)
            # print("Angle: ", angle, " deegres")
            for i in tree.rayIntersection(ray):
                if i.triLabel == face_int[ind]:
                    int_point_bingo = tuple(i.p)  # nearest intersection point
            # print("Intersection point Bingo: ", int_point_bingo)
            lambda_ct = min(dist)  # smallest lambda (distance between origin and intersection point)
            # If the ray is coming from inside the bone (usually because of a reflection), the above calculated normal has
            # to be inverted
            if np.dot(normal, rayVect) > 0:  # if dot product between calculated normal and ray is positive means that
                                            # the ray is coming from inside the bone, so the normal has to be inverted
                                            # (because normal are always pointing outside
                normal = -normal
                inversion = True

    return int_point_bingo, normal, lambda_ct, inversion


def intersection(start, vray, top, actual_index, final_geometry, initial_geometry):

    possible_obj = []  # list with the possible object
    possible_int = []  # possible lambda
    possible_nn = []  # normal
    vector_point_ct = []
    inversion_matrix = []
    inversion_bone = False
    inversion_marrow = False
    for o in top:  # check in all the object in topology
        if o.geometry.type == 'plane':  # if geometry is a plane
            int = inter(start, vray, o.geometry.x, o.geometry.y, o.geometry.z)  # call int -> smith algorithm
            if int != 'no intersection':  # if there is an intersection
                if int[0] == 0 or int[1] == 0:  # if at least one lambda is zero -> the ray start in this wall
                    if int[0] > 0 or int[1] > 0:   # if at least one is positive -> the wall is not behind the ray
                        int = np.asarray(int)  # make it as an array
                        possible_obj.append(top.index(o))  # fill the list with the index of the object
                        possible_int.append(np.amax(int[int >= 0]))  # append the max of the lambda (only lambda >0)
                        if vray[0] > 0:
                            possible_nn.append(np.array([-1, 0, 0]))
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
            vector_point_ct.append(False)
            inversion_matrix.append(False)
        if o.geometry.type == 'ct':  # if the object is 'ct' instead of 'plane':
            tree = o.geometry.tree  # import the tree object from geometry
            point_ct, normal_ct, dist_ct, inversion = inter_ct(start, vray, tree)  # calculate intersection
            vector_point_ct.append(point_ct)
            inversion_matrix.append(inversion)
            # if o.material.name == 'bone':
            # #     point_bone = point_ct
            # #     inversion_bone = inversion
            #
            # else:
            # #     point_marrow = point_ct
            #     inversion_marrow = inversion
            if point_ct != False:  # intersection occurred with the CT
                possible_obj.append(top.index(o))
                possible_int.append(dist_ct)
                possible_nn.append(normal_ct)


    possible_int = np.asarray(possible_int)
    choose_bone = False
    choose_marrow = False
    choose_ct = False
    same_index = False
    buggy_ray = False
    #calamity = False
    if len(possible_int) != 0:  # if there are intersections
        index_min = np.where(np.asarray(possible_int) == np.asarray(possible_int).min()) # find indices(always >1)of the min lambdas
        rng = np.asmatrix(index_min).shape  # indices of the minimum
        lambda_int = possible_int[np.argmin(possible_int)]  # lambda_int:Not important obj belongs: take just min lambda
        point_x = start[0] + vray[0] * lambda_int  # the point of the intersection
        for i in range(0, rng[1]):  # Check in the index_min to find the right object!
            if possible_obj[index_min[0][i]] != actual_index:  # if the object is not == to the actual_index
                index_min_new = possible_obj[index_min[0][i]]  # we found the right object!
                if top[index_min[0][i]].geometry.type == 'ct': choose_ct = True
                possible_nn_new = possible_nn[index_min[0][i]]

                if (np.abs(point_x - final_geometry) < 0.0001 or np.abs(point_x - initial_geometry) < 0.0001) and top[actual_index].material.name == 'bone':
                    buggy_ray = True
                    # print("ERROR1")
                elif inversion_matrix[index_min_new] is True and choose_ct is True:
                    buggy_ray = True
                    # print("ERROR2")
                elif top[actual_index].material.name == 'muscle' and top[index_min_new].material.name == 'muscle':
                    buggy_ray = True
                    # print("ERROR3")
                else:
                    buggy_ray = False
            else:  # if it's the same index
                same_index = True
                if top[actual_index].is_contained_in != []:  # if the object is contained in another object-> it's the other object
                    index_min_new = top[actual_index].is_contained_in
                    if top[index_min_new].geometry.type == 'ct': choose_ct = True
                    if inversion_matrix[index_min_new] is False and choose_ct is True:
                        buggy_ray = True
                        # print("ERROR4")
                    possible_nn_new = possible_nn[index_min[0][i]]

                if np.abs(point_x - final_geometry)<0.0001:  # if we are at the end of the geometry
                    index_min_new = actual_index  # actual_index is the right object
                    # if top[index_min[0][i]].geometry.type == 'ct': choose_ct = True
                    possible_nn_new = 'final'
                if np.abs(point_x - initial_geometry)<0.0001:  # if we are at the beginning of the geometry
                    index_min_new = actual_index  # actual_index is the right object
                    # if top[index_min[0][i]].geometry.type == 'ct': choose_ct = True
                    possible_nn_new = 'final'
    else:
        index_min_new = False
        lambda_int = False
    if lambda_int is False:
        index_min_new = False
        point = False
    # elif choose_bone is True:
    #     point = np.asarray(point_bone)
    # elif choose_marrow is True:
    #     point = np.asarray(point_marrow)
    elif choose_ct is True:
        if same_index is False:
            point = np.asarray(vector_point_ct[index_min_new])
        else:
            point = np.asarray(vector_point_ct[actual_index])
    else:
        point = start + vray * lambda_int

    if 'index_min_new' not in locals():  # bug to solve: with lambda_max really big, almost vertical rays, point_x is slightly different than final geometry
        index_min_new = actual_index  # actual_index is the right object
        possible_nn_new = 'final'
        #print('buggy ray')
    return index_min_new, point, possible_nn_new, lambda_int, buggy_ray  # return the index of the object intersected and the lambda_int







