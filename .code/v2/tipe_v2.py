import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
#plt.ion()       # passage en mode interractif de matplotlib # NE MARCHE PAS
import matplotlib.patches as mpatches
import json
from math import *
from planete import Perlin, Planet
from satellites import Satellite

#-------------------------------
# SETTINGS
#-------------------------------
nres = 1
WIDTH, HEIGHT = 800, 800
FPS = 60
TIME_SCALE = 100
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
        plt.savefig("./map-result/map.png", dpi=150, bbox_inches="tight")
        #plt.pause(0.001)
        plt.show()

class Main():

    def __init__(self):
        self.pln = Planet(SEED)
        self.tv = TextureVisualizer()

#-------------------------------
# MAIN RENDER
#-------------------------------

    def render(self, screen, texture, sphere, rot, cx, cy):
        screen.fill((5, 5, 15))

        # Flattening the points (LAT*LON, 3)
        pts_flat = sphere.reshape(-1, 3)
        colors_flat = texture.reshape(-1, 3)
        px, py, pz = self.pln.projection(pts_flat, rot, cx, cy, RADIUS)

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
# LOAD/CREATE THE SATELLITE CONSTELLATION
# -------------------------------

    def loadConstellation(self, filepath, altitude=1200):
        with open(filepath) as f:
            data = json.load(f)
        print(len(data))
        return [
            Satellite(
                inclinaison=sat["inclination_deg"],
                raan=sat["raan_deg"],
                altitude_km=altitude,
                phase=sat["phase_deg"],
            )
            for sat in data
        ]

    def generateOneWebWalker(self, altitude=1200):
        N_PLANES = 12
        N_PER_PLANE = 49  # 12 * 49 = 588, + some more in reserve to arrive to 648
        INCLINATION = 87.9

        satellites = []
        for p in range(N_PLANES):
            raan = p * (180.0 / N_PLANES)  # Spaced plan of de 180°/12 = 15°
            for s in range(N_PER_PLANE):
                # Shift between two plans (F parametre of Walker)
                walker_offset = (p * 360.0 / N_PER_PLANE) / N_PLANES
                phase = s * (360.0 / N_PER_PLANE) + walker_offset + p*25
                satellites.append(Satellite(
                    inclinaison=INCLINATION,
                    raan=raan,
                    altitude_km=altitude,
                    phase=np.radians(phase)
                ))
        return satellites

# -------------------------------
# CHECK THE VISIBILITY OF SAT
# -------------------------------

    def isHidden(self, pos_rot, radius=0.8):
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
            px, py, pz = self.pln.projection(pos.reshape(1, 3), rot, cx, cy, RADIUS)
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

        MODEL = True        # False to use real positions of the satellites
        if MODEL == True:
            satellites = self.generateOneWebWalker()
        else :
            satellites = self.loadConstellation("./sat-data/oneweb_constellation.json")
        ACTIVATE_ROTATION = True
        t = 0.0

        SHOW_MAP = True  # False to desactivate visualization
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
            #i = random.randint(0,len(texture)-1)
            #j = random.randint(0,len(texture[i])-1)
            #texture[i, j] = [255, 0, 0]
            if ACTIVATE_ROTATION:
                t += (1 / FPS) * TIME_SCALE
            pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    pygame.init()
    ml = Main()
    ml.main()
