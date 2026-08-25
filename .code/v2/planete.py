import pygame
import numpy as np
import random
import sys
import time
import json
from math import *


nres = 1
WIDTH, HEIGHT = 800, 800
FPS = 60
TIME_SCALE = 100
#RADIUS = 280
FOCAL = 900
LAT_STEP = int(120 * nres)
LON_STEP = int(240 * nres)
SEED = 0


# Colour Palette
DEEP_OCEAN = np.array([10, 60, 120], dtype=np.float32)
OCEAN = np.array([30, 100, 160], dtype=np.float32)
LOW_LAND = np.array([60, 120, 50], dtype=np.float32)
HIGH_LAND = np.array([90, 160, 50], dtype=np.float32)
MOUNTAIN = np.array([140, 120, 100], dtype=np.float32)
SNOW = np.array([240, 245, 255], dtype=np.float32)
DESERT = np.array([200, 175, 90], dtype=np.float32)
ICE = np.array([210, 235, 255], dtype=np.float32)

class Perlin():

# -------------------------------
# PROCEDURAL GENERATION OF THE PERLIN NOISE
# -------------------------------

    _perm = np.random.RandomState(SEED).permutation(256)
    _perm = np.concatenate([_perm, _perm]).astype(np.int32)

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(self, a, b, t):
        return a + t * (b - a)

    def grad2(self, h, x, y):
        h &= 3
        u = np.where(h < 2, x, y)
        v = np.where(h < 2, y, x)
        return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)

    def noise2(self, x, y):
        X = np.floor(x).astype(np.int32) & 255
        Y = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = self.fade(xf);
        v = self.fade(yf)
        aa = self._perm[self._perm[X] + Y]
        ab = self._perm[self._perm[X] + Y + 1]
        ba = self._perm[self._perm[X + 1] + Y]
        bb = self._perm[self._perm[X + 1] + Y + 1]
        return self.lerp(self.lerp(self.grad2(aa, xf, yf), self.grad2(ba, xf - 1, yf), u),
                         self.lerp(self.grad2(ab, xf, yf - 1), self.grad2(bb, xf - 1, yf - 1), u), v)

    def fbm(self, x, y, octaves=6):
        val, amp, freq = 0.0, 0.5, 1.0
        for _ in range(octaves):
            val += amp * self.noise2(x * freq, y * freq)
            amp *= 0.5
            freq *= 2.0
        return val

class Planet():

    def __init__(self, seed):
        self.SEED = seed
        SEED = self.SEED
        self.perlin = Perlin()

#-------------------------------
# TEXTURES
#-------------------------------

    def buildTexture(self):
        lats = np.linspace(-np.pi/2, np.pi/2, LAT_STEP)
        lons = np.linspace(-np.pi, np.pi, LON_STEP)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        # Unit sphere coordinates
        scale = 1.8
        nx = np.cos(lat_grid) * np.cos(lon_grid) * scale
        ny = np.cos(lat_grid) * np.sin(lon_grid) * scale
        nz = np.sin(lat_grid) * scale

        # fBm in 3D via projection (approximation)
        h = self.perlin.fbm(nx+nz*0.3, ny+nz*0.3, octaves=7)

        # Base fbm for continents
        continent_base = self.perlin.fbm(nx * 0.7, ny * 0.5, octaves=4)     # low frequencies
        h_combined = continent_base * 0.65 + h * 0.35
        land_mask = h_combined > 0.008

        # Land creation
        lat_d = np.degrees(lat_grid)
        lon_d = np.degrees(lon_grid)

        # Coast precision to make them more realistic
        coast_noise = self.perlin.fbm(nx*2, ny*2, octaves=4)*0.15
        land_mask = (land_mask & (h > -0.05 + coast_noise)) | (h > 0.27)

        # Color creation
        colors = np.zeros((LAT_STEP, LON_STEP, 3), dtype=np.float32)

        # Ocean
        ocean_t = np.clip((h + 0.4) / 0.5, 0, 1)
        ocean_col = DEEP_OCEAN[None, None, :] + ocean_t[:, :, None]*(OCEAN-DEEP_OCEAN)

        # Land
        elevation = np.clip((h_combined - 0.05) / 0.35, 0, 1)
        land_col = np.where(elevation[:, :, None] < 0.35, LOW_LAND + (elevation[:, :, None]/0.35) * (HIGH_LAND - LOW_LAND),
                            np.where(elevation[:, :, None] < 0.65,
                                     HIGH_LAND + ((elevation[:, :, None]-0.35)/0.3) * (MOUNTAIN - HIGH_LAND),
                                     MOUNTAIN + ((elevation[:, :, None]-0.65)/0.35) * (SNOW - MOUNTAIN)))

        # Deserts
        tropical_factor = np.exp(-((lat_d / 25.0) ** 2))        # gaussian centered on 0 degrees
        desert_noise = self.perlin.fbm(nx * 1.5 + 5.3, ny * 1.5 + 2.7)      # noise to modulate the limit between lands and deserts
        desert_mask = land_mask & (desert_noise * tropical_factor < -0.01)
        land_col = np.where(desert_mask[:, :, None], DESERT, land_col)

        # Polar ice
        polar_noise_N = self.perlin.fbm(nx * 2.0 + 8.1, ny * 2.0 + 4.6, octaves=4)  # North
        polar_noise_S = self.perlin.fbm(nx * 2.0 + 13.7, ny * 2.0 + 9.2, octaves=4) # South

        # gaussians to make a limitation for the poles
        polar_factor_N = np.exp(-(((lat_d - 90) / 22.0) ** 2))  # North
        polar_factor_S = np.exp(-(((lat_d + 90) / 22.0) ** 2))  # South

        ice_mask = (polar_noise_N * polar_factor_N > 0.03) & (lat_d > 50) | (polar_noise_S * polar_factor_S > 0.005) & (lat_d < -50) | (lat_d < -75)
        colors = np.where(ice_mask[:, :, None], ICE, np.where(land_mask[:, :, None], land_col, ocean_col))

        # Simple light
        Light = False   # True to activate
        if Light:
            sun = np.array([1.0, 0.3, 0.2])
            sun /= np.linalg.norm(sun)
            nx_n = np.cos(lat_grid) * np.cos(lon_grid)
            ny_n = np.cos(lat_grid) * np.sin(lon_grid)
            nz_n = np.sin(lat_grid)
            diffuse = (nx_n * sun[0] + ny_n * sun[1] + nz_n * sun[2])
            light = np.clip(0.25 + 0.75 * diffuse, 0, 1)
            colors *= light[:, :, None]

        return np.clip(colors, 0, 255).astype(np.uint8), lats, lons

#-------------------------------
# PRE-CALCULATION OF THE SPHERE
#-------------------------------

    def buildSpherePoints(self, lats, lons):
        lon_grid, lat_grid = np.meshgrid(lons, lats)
        x = np.cos(lat_grid) * np.cos(lon_grid)
        y = np.cos(lat_grid) * np.sin(lon_grid)
        z = np.sin(lat_grid)
        return np.stack([x, y, z], axis=-1) # (LAT, LON, 3)

#-------------------------------
# ROTATION MATRIX
#-------------------------------

    def rotX(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)
    def rotY(self, a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

#-------------------------------
# PROJECTION AND PERSPECTIVE
#-------------------------------

    def projection(self, pts3d, rot, cx, cy, RADIUS):
        r = (rot @ pts3d.T).T           # rotation
        cam_z = 3.0                     # camera on the z axis
        dz = cam_z - r[:, 2]
        scale = FOCAL / (FOCAL + dz)    # perspective
        px = cx + r[:, 0] * RADIUS * scale
        py = cy - r[:, 1] * RADIUS * scale
        return px, py, r[:, 2]
