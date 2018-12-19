#used only to generate rays from the trd element till the end of the lossless
from ray import Ray
import create_topology as tp
import numpy as np
from scipy import special
import intersection as inters
import reflection_refraction as rf
import intersection as inter
from numpy import linalg as LA
# trd is an object of the Class transducer
# r_1 is the first zero bessel 3.8317
# top is the list of objects of the class Topology
# This Function is called for every trd element -> 256 times for Sonalleve transducer

def generation_rays_lossless(trd, top,  Nrays, final_geometry, initial_geometry, index_trd, AA,k_r,index): # r_1 first zero bessel
    ptrd_1 = 0
    start = ([trd.coord[index_trd, 1], trd.coord[index_trd, 2], trd.coord[index_trd, 3]])
    start = np.asanyarray(start)
    normal = -  start / LA.norm(start)
    v2 = ([-normal[2], 0, normal[0]])
    v2_norm = v2 / LA.norm(v2)
    v3 = np.cross(v2_norm, normal)
    v3 = v3 / LA.norm(v3)
    r_loss=[]
    for j in range(0, Nrays):
        rd = np.random.random_sample()
        #rd = 0.07
        theta = np.arccos(1 - AA*rd)
        phi = 2 * np.pi * np.random.random_sample()
        #phi = 2 * np.pi * 0.06
        vray = normal + np.tan(theta)*(np.cos(phi)*v2_norm + np.sin(phi)*v3)
        vray = vray / LA.norm(vray)
        B = k_r * np.sin(theta)
        if B == 0:
            scaling = 1
        else:
            scaling = (2 * special.j1(B) / B )**2
        I0 = scaling
        ptrd_1 = ptrd_1 + I0
        next_obj, end, nn, lambda_int = inter.intersection(start, vray, top,
                                                                    index,
                                                                    final_geometry,
                                                                    initial_geometry)
        path = []
        path.append(index)
        phi_initial = 0
        phase_shift = 0
        real_dist = np.dot((end -start), nn)/np.dot(vray, nn)
        phi_final = phi_initial + top[index].material.k * lambda_int
        r = Ray(start, end, vray, path, I0, phi_initial, phi_final, phase_shift,
                                         0, index, next_obj, nn, I0)
        r_loss.append(r)
    return r_loss, ptrd_1


def Generation_ray(top, ray, final_geometry, initial_geometry):
    son = []
    I_limit = 0.0001
    control = ray

    while len(control)!= 0:
        processing = control
        control = []
        for o in range(0, len(processing)):
            # NEXT MATERIAL IS SOFT?
            if top[processing[o].next_obj].material.type == 'liquid': # next material is soft
                # ACTUAL MATERIAL IS SOFT?
                if top[processing[o].obj_index].material.type == 'liquid':  # actual material soft SOFT->SOFT
                    nn = processing[o].nn
                    refr, v_out = rf.refraction0(processing[o].vray, nn, top[processing[o].obj_index].material.c * 0.01, top[processing[o].next_obj].material.c * 0.01)
                    if top[processing[o].obj_index].material.name == 'lossless':  # ray is in lossless
                        T = 1
                        R = 0
                        Ph_refl = 0
                        Ph_tr = 0
                    else:
                        T, R, Ph_refl, Ph_tr = rf.coefficient_ll(processing[o].vray, nn,
                                                               top[processing[o].obj_index].material.c * 1e-2,
                                                               top[processing[o].next_obj].material.c * 1e-2,
                                                               top[processing[o].obj_index].material.rho * 1e6,
                                                               top[processing[o].next_obj].material.rho * 1e6)
                    #refraction In soft tissue
                    if processing[o].IF > I_limit:
                        if refr == True and top[processing[o].next_obj].material.name != 'lossless': #I have refraction
                            start = processing[o].end
                            I0 = processing[o].IF * T
                            vray = v_out / LA.norm(v_out)
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].next_obj,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            if isinstance(possible_nn, str): # if possible nn is a string and not a vector
                                IF = 0
                            else:
                                IF = I0 * np.exp(-2 * top[processing[o].next_obj].material.alpha * lambda_int)
                            lst = []
                            lst.append(processing[o].next_obj)
                            path = (processing[o].path)+ lst
                            phi_initial = processing[o].phase_final + Ph_tr
                            phi_final = phi_initial + top[processing[o].next_obj].material.k * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_tr
                            sn = Ray(start, end, vray, path, I0, phi_initial, phi_final, phase_shift, 0, processing[o].next_obj, next_obj,possible_nn, IF,0, False)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)
                        if R != 0 and top[processing[o].next_obj].material.name != 'lossless':
                            if processing[o].IF > I_limit:
                                v_out = rf.reflection(processing[o].vray, nn)
                                vray = v_out / LA.norm(v_out)
                                start = processing[o].end
                                I0 = processing[o].IF * R
                                next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                            processing[o].obj_index,
                                                                                            final_geometry,
                                                                                            initial_geometry)
                                if isinstance(possible_nn, str):  # if possible nn is a string and not a vector
                                    IF = 0
                                else:
                                    IF = I0 * np.exp(-2 * top[processing[o].obj_index].material.alpha * lambda_int)

                                lst = []
                                lst.append(processing[o].obj_index)
                                path = (processing[o].path) + lst
                                phi_initial = processing[o].phase_final + Ph_refl
                                phi_final = phi_initial + top[processing[o].obj_index].material.k * lambda_int
                                phase_shift = processing[o].phase_shift + Ph_refl
                                sn = Ray(start, end, vray, path, I0, phi_initial, phi_final, phase_shift,
                                         0, processing[o].obj_index, next_obj, possible_nn, IF,0, False)
                                son.append(sn)
                                if IF > I_limit:
                                    control.append(sn)
                # ACTUAL MATERIAL IS NOT SOFT! RAY IS IN BONE!!
                else:

                    nn = processing[o].nn
                    alpha_in = np.arccos(np.abs(np.dot(nn, processing[o].vray)))
                    rho_liquid = top[processing[o].next_obj].material.rho * 1e6
                    rho_solid = top[processing[o].obj_index].material.rho * 1e6
                    c_long_liquid = top[processing[o].next_obj].material.c * 1e-2
                    c_long_solid = top[processing[o].obj_index].material.cL * 1e-2
                    c_shear_solid = top[processing[o].obj_index].material.cS * 1e-2
                    # IT's a LONGITUDINAL RAY: REFL LONG - REFL SHEAR -REFRACT LONG
                    if processing[o].shear == False:  # IT's a LONGITUDINAL RAY: REFL LONG - REFL SHEAR -REFRACT LONG
                        Refl_long, Refl_shear, Transm_long = rf.B2MCoef_long(alpha_in, rho_liquid,
                                                                             rho_solid, c_long_liquid,
                                                                             c_long_solid, c_shear_solid)
                        intens_long_reflect = processing[o].IF * Refl_long
                        intens_shear_reflec = processing[o].IF * Refl_shear
                        intens_long_refract = processing[o].IF * Transm_long
                        #REFLECTED LONG RAY
                        if intens_long_reflect > I_limit:
                            v_out = rf.reflection(processing[o].vray, nn)
                            vray = v_out / LA.norm(v_out)
                            start = processing[o].end
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].obj_index,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            IF = intens_long_reflect * np.exp(-2 * top[processing[o].obj_index].material.alphaL * lambda_int)
                            lst = []
                            lst.append(processing[o].obj_index)
                            path = processing[o].path + lst
                            Ph_refl_long = 0  # to change!!!! Maybe there's a phase shift
                            phi_initial = processing[o].phase_final + Ph_refl_long
                            phi_final = phi_initial + top[processing[o].obj_index].material.kL* lambda_int
                            phase_shift = processing[o].phase_shift + Ph_refl_long
                            sn = Ray(start, end, vray, path, intens_long_reflect, phi_initial, phi_final, phase_shift,
                                     0, processing[o].obj_index, next_obj, possible_nn, IF, 0, False)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)
                        # REFLECTED SHEAR
                        if intens_shear_reflec > I_limit:
                            refl, v_out, poldir = rf.reflection2(processing[o].vray, nn, c_long_solid, c_long_liquid)
                            vray = v_out / LA.norm(v_out)
                            start = processing[o].end
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].obj_index,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            IF = intens_long_reflect * np.exp(-2 * top[processing[o].obj_index].material.alphaS * lambda_int)
                            lst = []
                            lst.append(processing[o].obj_index)
                            path = (processing[o].path) + lst
                            Ph_refl_shear = 0  # to change!!!! Maybe there's a phase shift
                            phi_initial = processing[o].phase_final + Ph_refl_shear
                            phi_final = phi_initial + top[processing[o].obj_index].material.kS * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_refl_shear
                            sn = Ray(start, end, vray, path, intens_long_reflect, phi_initial, phi_final, phase_shift,
                                     0, processing[o].obj_index, next_obj, possible_nn, IF, poldir, True)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)
                        # REFRACTED LONGITUDINAL
                        if intens_long_refract > I_limit:

                            refr, v_out, piero = rf.refraction(processing[o].vray, nn, c_long_solid, c_long_liquid)
                            if refr == True:  # I have refraction
                                start = processing[o].end
                                vray = v_out / LA.norm(v_out)
                                next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                            processing[o].next_obj,
                                                                                            final_geometry,
                                                                                            initial_geometry)

                                if isinstance(possible_nn, str):  # if possible nn is a string and not a vector
                                    IF = 0
                                else:
                                    IF = intens_long_refract * np.exp(-2 * top[processing[o].next_obj].material.alpha * lambda_int)
                                lst = []
                                lst.append(processing[o].next_obj)
                                path = (processing[o].path) + lst
                                Ph_tr_shift = 0 #TO CHANGE!!! CAN BE DIFFERENT
                                phi_initial = processing[o].phase_final + Ph_tr_shift
                                phi_final = phi_initial + top[processing[o].next_obj].material.k * lambda_int
                                phase_shift = processing[o].phase_shift + Ph_tr_shift
                                sn = Ray(start, end, vray, path, intens_long_refract, phi_initial, phi_final, phase_shift, 0,
                                         processing[o].next_obj, next_obj, possible_nn, IF, 0, False)
                                son.append(sn)
                                if IF > I_limit:
                                    control.append(sn)
                    else: #  it's a shear wave. Shear: reflected shear horiz pol, reflected shear vert pol, reflected long, refracted long

                        v_out, vpol_dir_in, vpol_dir_refl, hpol_dir = rf.reflection3(processing[o].vray, nn)
                        pol_dir_in = processing[o].poldir
                        cosvert = np.dot(pol_dir_in, vpol_dir_in)
                        coshor = np.dot(pol_dir_in, hpol_dir)
                        intensity_horcomp = processing[o].IF * coshor ** 2
                        intensity_vertcomp = processing[o].IF * cosvert ** 2

                        # SHEAR HORIZONATAL REFLECTED
                        if intensity_horcomp > I_limit:  # SHEAR HORIZONATAL REFLECTED
                            vray = v_out / LA.norm(v_out)
                            start = processing[o].end
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].obj_index,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            IF = intensity_horcomp * np.exp(
                                -2 * top[processing[o].obj_index].material.alphaS * lambda_int)
                            lst = []
                            lst.append(processing[o].obj_index)
                            path = (processing[o].path) + lst
                            Ph_refl_shear = 0  # to change!!!! Maybe there's a phase shift
                            phi_initial = processing[o].phase_final + Ph_refl_shear
                            phi_final = phi_initial + top[processing[o].obj_index].material.kS * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_refl_shear
                            sn = Ray(start, end, vray, path, intensity_horcomp, phi_initial, phi_final, phase_shift,
                                     0, processing[o].obj_index, next_obj, possible_nn, IF, hpol_dir, True)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)

                        Refl_long, Refl_shear, Transm_long = rf.B2MCoef_shear(alpha_in, rho_liquid,
                                                                                          rho_solid, c_long_liquid,
                                                                                          c_long_solid,
                                                                                          c_shear_solid)

                        intensity_vert_shear_reflected = Refl_shear * intensity_vertcomp

                        # SHEAR VERTICAL REFLECTED
                        if intensity_vert_shear_reflected > I_limit:  # SHEAR VERTICAL REFLECTED
                            vray = v_out / LA.norm(v_out)
                            start = processing[o].end
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].obj_index,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            IF = intensity_vert_shear_reflected * np.exp(
                                -2 * top[processing[o].obj_index].material.alphaS * lambda_int)
                            lst = []
                            lst.append(processing[o].obj_index)
                            path = (processing[o].path) + lst
                            Ph_refl_shear = 0  # to change!!!! Maybe there's a phase shift
                            phi_initial = processing[o].phase_final + Ph_refl_shear
                            phi_final = phi_initial + top[processing[o].obj_index].material.kS * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_refl_shear
                            sn = Ray(start, end, vray, path, intensity_horcomp, phi_initial, phi_final, phase_shift,
                                     0, processing[o].obj_index, next_obj, possible_nn, IF, vpol_dir_refl, True)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)
                        print(processing[o].start)
                        # longitudinal REFLECTED
                        longrefl_possible, v_out, poldir = rf.reflection2(processing[o].vray, nn, c_shear_solid,
                                                                         c_long_solid)
                        intensity_longit_reflected = Refl_long * intensity_vertcomp
                        if longrefl_possible == 1 and intensity_longit_reflected > I_limit:
                            v_out = rf.reflection(processing[o].vray, nn)
                            vray = v_out / LA.norm(v_out)
                            start = processing[o].end
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].obj_index,
                                                                                        final_geometry,
                                                                                        initial_geometry)
                            if isinstance(possible_nn, str):  # if possible nn is a string and not a vector
                                IF = 0
                            else:
                                IF = intensity_longit_reflected * np.exp(
                                    -2 * top[processing[o].obj_index].material.alphaL * lambda_int)
                            lst = []
                            lst.append(processing[o].obj_index)
                            path = processing[o].path + lst
                            Ph_refl_long = 0  # to change!!!! Maybe there's a phase shift
                            phi_initial = processing[o].phase_final + Ph_refl_long
                            phi_final = phi_initial + top[processing[o].obj_index].material.kL * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_refl_long
                            sn = Ray(start, end, vray, path, intensity_longit_reflected, phi_initial, phi_final, phase_shift,
                                     0, processing[o].obj_index, next_obj, possible_nn, IF, 0, False)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)

                        # LONGITUDINAL REFRACTED
                        refr, v_out, poldir = rf.refraction(processing[o].Vray, nn, c_shear_solid, c_long_liquid)
                        intensity_longit_refracted = Transm_long * intensity_vertcomp
                        if intensity_longit_refracted > I_limit and refr ==1:
                            start = processing[o].end
                            vray = v_out / LA.norm(v_out)
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].next_obj,
                                                                                        final_geometry,
                                                                                        initial_geometry)

                            if isinstance(possible_nn, str):  # if possible nn is a string and not a vector
                                IF = 0
                            else:
                                IF = intensity_longit_refracted * np.exp(
                                    -2 * top[processing[o].next_obj].material.alpha * lambda_int)
                            lst = []
                            lst.append(processing[o].next_obj)
                            path = (processing[o].path) + lst
                            Ph_tr_shift = 0  # TO CHANGE!!! CAN BE DIFFERENT
                            phi_initial = processing[o].phase_final + Ph_tr_shift
                            phi_final = phi_initial + top[processing[o].next_obj].material.k * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_tr_shift
                            sn = Ray(start, end, vray, path, intensity_longit_refracted, phi_initial, phi_final, phase_shift, 0,
                                     processing[o].next_obj, next_obj, possible_nn, IF, 0, False)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)



                        # SOFT -> BONE
            else:
                # NEXT MATERIAL IS BONE
                nn = processing[o].nn
                alpha_in = np.arccos(np.abs(np.dot(nn,processing[o].vray)))
                rho_liquid = top[processing[o].obj_index].material.rho * 1e6
                rho_solid = top[processing[o].next_obj].material.rho * 1e6
                c_long_liquid = top[processing[o].obj_index].material.c * 1e-2
                c_long_solid = top[processing[o].next_obj].material.cL * 1e-2
                c_shear_solid = top[processing[o].next_obj].material.cS * 1e-2

                Refl, Transm_long, Transm_shear, Ph_refl, Ph_tr_long, Ph_tr_shear = rf.M2BCoef(alpha_in,
                                                                                               rho_liquid, rho_solid,
                                                                                               c_long_liquid,
                                                                                               c_long_solid,
                                                                                               c_shear_solid)
                # REFLECTED RAY IN SOFT

                intens_long_refl = processing[o].IF * Refl
                if intens_long_refl > I_limit:
                    v_out = rf.reflection(processing[o].vray, nn)
                    vray = v_out / LA.norm(v_out)
                    start = processing[o].end
                    next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                processing[o].obj_index,
                                                                                final_geometry,
                                                                                initial_geometry)
                    if isinstance(possible_nn, str):  # if possible nn is a string and not a vector
                        IF = 0
                    else:
                        IF = intens_long_refl * np.exp(-2 * top[processing[o].obj_index].material.alpha * lambda_int)
                    lst = []
                    lst.append(processing[o].obj_index)
                    path = (processing[o].path) + lst
                    phi_initial = processing[o].phase_final + Ph_refl
                    phi_final = phi_initial + top[processing[o].obj_index].material.k * lambda_int
                    sn = Ray(start, end, vray, path, intens_long_refl, phi_initial, phi_final, Ph_refl, 0, processing[o].obj_index,
                             next_obj, possible_nn, IF,0, False)
                    son.append(sn)

                    if IF > I_limit:
                        control.append(sn)

                    # REFRACTED LONGITUDINAL RAY IN BONE
                    longrefrac_possible, v_out, pp = rf.refraction(processing[o].vray, nn, c_long_liquid, c_long_solid)
                    intens_long_refrac = processing[o].IF * Transm_long
                    if longrefrac_possible == True and intens_long_refrac > I_limit:
                        start = processing[o].end
                        I0 = intens_long_refrac
                        vray = v_out / LA.norm(v_out)
                        next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                    processing[o].next_obj,
                                                                                    final_geometry,
                                                                                    initial_geometry)
                        if isinstance(possible_nn, str):
                            IF = 0
                        else:
                            IF = I0 * np.exp(-2 * top[processing[o].next_obj].material.alphaL * lambda_int)
                        lst = []
                        lst.append(processing[o].next_obj)
                        path = (processing[o].path) + lst
                        phi_initial = processing[o].phase_final + Ph_tr_long
                        phi_final = phi_initial + top[processing[o].next_obj].material.kL * lambda_int
                        phase_shift = processing[o].phase_shift + Ph_tr_long
                        sn = Ray(start, end, vray, path, I0, phi_initial, phi_final, phase_shift, 0, processing[o].next_obj,
                                 next_obj, possible_nn, IF,0, False)
                        son.append(sn)
                        if IF > I_limit:
                            control.append(sn)

                        # if np.abs(start[0] - end[0]) < 0.00001:
                        #   print('something wrong in long')

                    # REFRACTED SHEAR RAY IN BONE
                    shearrefrac_possible, v_out, poldir = rf.refraction(processing[o].vray, nn, c_long_liquid, c_shear_solid)
                    if shearrefrac_possible == True:
                        intens_shear_refrac = processing[o].IF * Transm_shear
                        if intens_shear_refrac > I_limit:
                            start = processing[o].end
                            I0 = intens_shear_refrac
                            vray = v_out / LA.norm(v_out)
                            poldir = poldir / LA.norm(poldir)
                            next_obj, end, possible_nn, lambda_int = inter.intersection(start, vray, top,
                                                                                        processing[o].next_obj,
                                                                                        final_geometry,
                                                                                        initial_geometry)
                            if isinstance(possible_nn, str): # if possible_nn is a string
                                IF = 0
                            else:
                                IF = I0 * np.exp(-2 * top[processing[o].next_obj].material.alphaS * lambda_int)
                            lst = []
                            lst.append(processing[o].next_obj)
                            path = (processing[o].path) + lst
                            phi_initial = processing[o].phase_final + Ph_tr_shear
                            phi_final = phi_initial + top[processing[o].next_obj].material.kL * lambda_int
                            phase_shift = processing[o].phase_shift + Ph_tr_shear
                            sn = Ray(start, end, vray, path, I0, phi_initial, phi_final, phase_shift, 0,
                                     processing[o].next_obj,
                                     next_obj, possible_nn, IF, poldir, True)
                            son.append(sn)
                            if IF > I_limit:
                                control.append(sn)
                            # if np.abs(start[0] - end[0]) < 0.00001:
                            #    print('something wrong in shear')

    return son
