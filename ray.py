class Ray:
    def __init__(self, star=0, en=0, dir=0, path=0, i0=0, phi=0, phi_final=0, phi_shift=0, check=0, obj=-99, next_obj=-99, nn=0, IF=-99, poldir = 0, shear= False):
        self.start = star  # start point
        self.end = en  # end point
        self.vray = dir  # direction
        self.path = path  # memory of the ray -> all the objects it went across
        self.I0 = i0  # initial power
        self.phase_initial = phi  # initial phase
        self.phase_final = phi_final  # initial phase
        self.phase_shift = phi_shift  # memory of the phase shift accumulated
        self.check = check  # is it already processed to get the power production? 1 yes 0 no
        self.obj_index = obj  #index of the object the ray is
        self.next_obj = next_obj #next index object the ray intersects
        self.nn = nn #the normal in the end point of the ray
        self.IF = IF #final intensity
        self.poldir = poldir #pol dir for sh waves
        self.shear = shear #True if it's a shear

