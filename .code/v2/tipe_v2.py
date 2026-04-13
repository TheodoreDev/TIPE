import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
#plt.ion()       # passage en mode interractif de matplotlib # NE MARCHE PAS
import matplotlib.patches as mpatches
from pygame.examples.go_over_there import screen

#-------------------------------
# SETTINGS
#-------------------------------
nres = 1
WIDTH, HEIGHT = 800, 800
FPS = 60
RADIUS = 280
FOCAL = 900
LAT_STEP = int(120 * nres)
LON_STEP = int(240 * nres)
SEED = int(random.random() * 100000000)
print(f'Seed: {SEED}')

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

    def __init__(self):
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

    def projection(self, pts3d, rot, cx, cy):
        r = (rot @ pts3d.T).T           # rotation
        cam_z = 3.0                     # camera on the z axis
        dz = cam_z - r[:, 2]
        scale = FOCAL / (FOCAL + dz)    # perspective
        px = cx + r[:, 0] * RADIUS * scale
        py = cy - r[:, 1] * RADIUS * scale
        return px, py, r[:, 2]

class Satellite():
    def __init__(self, inclinaison, raan, altitude, phase=0.0):
        self.inclinaison = np.radians(inclinaison)
        self.raan = np.radians(raan)
        self.r = 1.0 + altitude
        self.phase = phase
        self.omega = 0.3 / (self.r ** 1.5)      # Kepler law simplification

    def position(self, t):
        angle = self.omega * t + self.phase

        # Position on the orbital plan
        x_orb = self.r * np.cos(angle)
        y_orb = self.r * np.sin(angle)

        # Rotation by inclinaison then by RAAN
        x = np.cos(self.raan) * x_orb - np.sin(self.raan) * np.cos(self.inclinaison) * y_orb
        y = np.sin(self.raan) * x_orb + np.cos(self.raan) * np.cos(self.inclinaison) * y_orb
        z = np.sin(self.inclinaison) * y_orb

        return np.array([x, y, z])

class TextureVisualizer():

#-------------------------------
# TEXTURE VISUALIZER
#-------------------------------

    def showTexture2D(self, texture, lats, lons, seed=None):
        fig, ax = plt.subplots(figsize=(14, 7), facecolor='#05050F')
        ax.set_facecolor('#05050F')

        # Texture displaying
        lat_deg = np.degrees(lats)
        lon_deg = np.degrees(lons)
        extent = [lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]]

        ax.imshow(texture, origin='lower', extent=extent, aspect='auto', interpolation='bilinear')

        # Coordinate grid
        ax.set_xticks(np.arange(-180, 181, 30))
        ax.set_yticks(np.arange(-90, 91, 30))
        ax.grid(color='white', linewidth=0.3, alpha=0.3, linestyle='--')

        # Tropical and polar circle
        special_lats = {
            'Cercle Arctique': 66.5,
            'Tropique Cancer': 23.5,
            'Équateur': 0.0,
            'Tropique Capricorne': -23.5,
            'Cercle Antarctique': -66.5,
        }
        colors_lat = {
            'Équateur': '#FF6B6B',
            'Tropique Cancer': '#FFD93D',
            'Tropique Capricorne': '#FFD93D',
            'Cercle Arctique': '#6BCFFF',
            'Cercle Antarctique': '#6BCFFF',
        }
        for name, lat_val in special_lats.items():
            ax.axhline(lat_val, color=colors_lat[name], linewidth=0.8,
                       linestyle=':', alpha=0.7)
            ax.text(182, lat_val, name, color=colors_lat[name],
                    fontsize=6.5, va='center', clip_on=False)

        # Axis
        ax.set_xlabel('Longitude', color='#A0C8FF', fontsize=10)
        ax.set_ylabel('Latitude',  color='#A0C8FF', fontsize=10)
        ax.tick_params(colors='#A0C8FF', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#A0C8FF')

        # Title
        title = f'Projection de Mercator — seed {seed}'
        ax.set_title(title, color='white', fontsize=13, pad=10)

        # Biomes legende
        legend_items = [
            mpatches.Patch(color=np.array([10,  60, 120])/255, label='Deep Ocean'),
            mpatches.Patch(color=np.array([30, 100, 160])/255, label='Ocean'),
            mpatches.Patch(color=np.array([60, 120,  50])/255, label='Plains'),
            mpatches.Patch(color=np.array([90, 150,  60])/255, label='Forest'),
            mpatches.Patch(color=np.array([140,120, 100])/255, label='Mountain'),
            mpatches.Patch(color=np.array([240,245, 255])/255, label='Snow'),
            mpatches.Patch(color=np.array([200,175,  90])/255, label='Desert'),
            mpatches.Patch(color=np.array([210,235, 255])/255, label='Ice'),
        ]
        ax.legend(handles=legend_items, loc='lower left', facecolor='#0D0D1A', edgecolor='#A0C8FF',
                  labelcolor='white', fontsize=8, framealpha=0.8)

        plt.tight_layout()
        #plt.pause(0.001)
        plt.show()

class Main():

    def __init__(self):
        self.pln = Planet()
        self.tv = TextureVisualizer()

#-------------------------------
# MAIN RENDER
#-------------------------------

    def render(self, screen, texture, sphere, rot, cx, cy):
        screen.fill((5, 5, 15))

        # Flattening the points (LAT*LON, 3)
        pts_flat = sphere.reshape(-1, 3)
        colors_flat = texture.reshape(-1, 3)
        px, py, pz = self.pln.projection(pts_flat, rot, cx, cy)

        # Painter algorithm
        order = np.argsort(pz)
        px = px[order]; py = py[order]
        pz = pz[order]; colors_flat = colors_flat[order]

        # Point size according to distance and resolution
        base_size = max(1, RADIUS*2 // max(LAT_STEP, LON_STEP) + 1)

        # Optimization by drawing only what is visible
        visible = pz > -0.05
        px_v = px[visible].astype(int)
        py_v = py[visible].astype(int)
        col_v = colors_flat[visible]

        # Drawing with surface array
        surf_arr = pygame.surfarray.pixels3d(screen)
        mask_x = (px_v >= 0) & (px_v < WIDTH)
        mask_y = (py_v >= 0) & (py_v < HEIGHT)
        mask = mask_x & mask_y

        for s in range(base_size, 0, -1):
            for dx in range(-s+1, s):
                for dy in range(-s+1, s):
                    if dx * dx + dy * dy <= s*s:
                        xc = np.clip(px_v[mask]+dx, 0, WIDTH-1)
                        yc = np.clip(py_v[mask]+dy, 0, HEIGHT-1)
                        surf_arr[xc, yc] = col_v[mask]

        del surf_arr        # Unlock the pygame Surface

        # Atmosphere (blue halo) (Only for style)
        atmo_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        for r_off in range(8, 0, -1):
            alpha = int(18 - r_off * 1.8)
            if alpha > 0:
                pygame.draw.circle(atmo_surf, (80, 160, 255, alpha),
                                   (cx, cy), RADIUS + r_off, 2)
        screen.blit(atmo_surf, (0, 0))

        # HUD
        font = pygame.font.SysFont('monospace', 14)
        hints = ["3D Viewer", "R : reset", "Z : top view", "S : down vue", "Q, D : Side vue"]
        for i, txt in enumerate(hints):
            surf = font.render(txt, True, (160, 200, 255))
            screen.blit(surf, (12, 12+i*20))

# -------------------------------
# CHECK THE VISIBILITY OF SAT
# -------------------------------

    def isHidden(self, pos_rot, radius=0.7):
        x, y, z = pos_rot
        cam_z = 3.0
        #dx, dy, dz = x, y, cam_z - z
        ox, oy, oz = 0.0, 0.0, cam_z
        dirx, diry, dirz = x - ox, y - oy, z - oz

        a = dirx**2 + diry**2 + dirz**2
        b = 2 * (ox*dirx + oy*diry + oz*dirz)
        c = ox**2 + oy**2 + oz**2 - radius**2

        if b**2 - 4*a*c < 0:
            return False
        t1 = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)
        return 0 < t1 < 1.0

# -------------------------------
# SATELLITE RENDERING
# -------------------------------

    def drawSat(self, screen, satellites, rot, cx, cy, t):
        for sat in satellites:
            pos = sat.position(t)
            if self.isHidden(rot @ pos):
                continue
            px, py, pz = self.pln.projection(pos.reshape(1, 3), rot, cx, cy)
            pygame.draw.circle(screen, (255, 50, 50), (int(px[0]), int(py[0])), 4)

# -------------------------------
# MAIN LOOP
# -------------------------------

    def main(self):
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('TIPE 3D VISUALIZATION')
        clock = pygame.time.Clock()

        print("Texture generation ...")
        start = time.time()
        texture, lats, lons = self.pln.buildTexture()
        sphere = self.pln.buildSpherePoints(lats, lons)
        end = time.time()
        print(f'Build time : {end - start}')
        print("Ready to render")

        # Initial camera state
        rot = self.pln.rotY(np.radians(-37)) @ self.pln.rotX(np.radians(-70))
        cx, cy = WIDTH//2, HEIGHT//2
        dragging = False
        last_mouse = (0, 0)
        zoom_scale = 1.0            # Not used anymore

        global RADIUS
        base_radius = RADIUS

        satellites = [
            Satellite(0, 0, 0.6),
        ]
        ACTIVATE_ROTATION = True
        t = 0.0

        SHOW_MAP = False  # False to desactivate visualization
        if SHOW_MAP:
            self.tv.showTexture2D(texture, lats, lons, seed=SEED)

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        rot = self.pln.rotY(np.radians(-37)) @ self.pln.rotX(np.radians(-70))        # Reset de la rotation
                        RADIUS = base_radius
                    elif event.key == pygame.K_z:
                        rot = self.pln.rotY(np.radians(0)) @ self.pln.rotX(np.radians(0))
                    elif event.key == pygame.K_s:
                        rot = self.pln.rotY(np.radians(0)) @ self.pln.rotX(np.radians(180))
                    elif event.key == pygame.K_q:
                        rot = self.pln.rotY(np.radians(0)) @ self.pln.rotX(np.radians(-90))
                    elif event.key == pygame.K_d:
                        rot = self.pln.rotY(np.radians(180)) @ self.pln.rotX(np.radians(-90))
                    elif event.key == pygame.K_SPACE:
                        ACTIVATE_ROTATION = False if ACTIVATE_ROTATION else True

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        dragging = True
                        last_mouse = event.pos
                    elif event.button == 4:     # mouse wheel up : zoom in
                        RADIUS = min(int(RADIUS*1.08), 420)
                    elif event.button == 5:     # mouse wheel down : zoom out
                        RADIUS = max(int(RADIUS*0.93), 80)

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        dragging = False

                elif event.type == pygame.MOUSEMOTION:
                    if dragging:        # follow of the "mouse position" by the camera
                        dx = event.pos[0] - last_mouse[0]
                        dy = event.pos[1] - last_mouse[1]
                        last_mouse = event.pos
                        sens = 0.005
                        rot = self.pln.rotY(dx*sens) @ self.pln.rotX(dy*sens) @ rot

            self.render(screen, texture, sphere, rot, cx, cy)
            self.drawSat(screen, satellites, rot, cx, cy, t)
            if ACTIVATE_ROTATION:
                t += 0.01
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    pygame.init()
    ml = Main()
    ml.main()
