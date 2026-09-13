import pygame
import numpy as np
import random
import sys
import time
import json
from math import *



# Colour Palette
DEEP_OCEAN = np.array([10, 60, 120], dtype=np.float32)
OCEAN = np.array([30, 100, 160], dtype=np.float32)
LOW_LAND = np.array([60, 120, 50], dtype=np.float32)
HIGH_LAND = np.array([90, 160, 50], dtype=np.float32)
MOUNTAIN = np.array([140, 120, 100], dtype=np.float32)
SNOW = np.array([240, 245, 255], dtype=np.float32)
DESERT = np.array([200, 175, 90], dtype=np.float32)
ICE = np.array([210, 235, 255], dtype=np.float32)

#-------------------------------
# SATELLITE GESTION
#-------------------------------

class Satellite():
    def __init__(self, inclinaison, raan, altitude_km, phase=0.0):
        self.inclinaison = np.radians(inclinaison)
        self.raan = np.radians(raan)
        self.phase = phase

        #self.omega = 0.3 / (self.r ** 1.5)      # Kepler law simplification
        GM = 3.986e14  # m³/s²
        R_EARTH = 6.371e6  # m
        a = R_EARTH + altitude_km * 1000
        self.omega_real = np.sqrt(GM / a ** 3)  # rad/s
        self.r = 1.0 + altitude_km / 6371.0     # for vizualisation

    def position(self, t):
        angle = self.omega_real * t + self.phase

        # Position on the orbital plan
        x_orb = self.r * np.cos(angle)
        y_orb = self.r * np.sin(angle)

        # Rotation by inclinaison then by RAAN
        x = np.cos(self.raan) * x_orb - np.sin(self.raan) * np.cos(self.inclinaison) * y_orb
        y = np.sin(self.raan) * x_orb + np.cos(self.raan) * np.cos(self.inclinaison) * y_orb
        z = np.sin(self.inclinaison) * y_orb

        return np.array([x, y, z])