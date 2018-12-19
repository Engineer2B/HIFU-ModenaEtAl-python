import numpy as np
class Cube:
    def __init__(self, center=0, sid=0):
        self.cent = center
        self.side = sid
        self.boundsx= np.array([center[0]-sid/2, center[0]+sid/2])
        self.boundsy = np.array([center[1] - sid/2, center[1] +sid/2])
        self.boundsz = np.array([center[2] - sid/2, center[2] + sid/2])
