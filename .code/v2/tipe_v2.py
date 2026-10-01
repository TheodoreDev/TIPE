import pygame
import numpy as np
import random
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import json
from math import *


from planete import Perlin, Planet
from satellites import Satellite
from utils import *
from simu import *

#-------------------------------
# SETTINGS / CONST
#-------------------------------
nres = 1
TEST_V = 7.0
WIDTH, HEIGHT = 800, 800
FPS = 240
TIME_SCALE = 100
RADIUS = 280
FOCAL = 900
LAT_STEP = int(120 * nres)
LON_STEP = int(240 * nres)
OMEGA_EARTH = (2 * pi/86164.0)
MIN_ELEVATION_DEG = 45
SEED = int(random.random() * 100000000)
print(f'Seed: {SEED}')

# Colour Palette
#DEEP_OCEAN = np.array([10, 60, 120], dtype=np.float32)
#OCEAN = np.array([30, 100, 160], dtype=np.float32)
#LOW_LAND = np.array([60, 120, 50], dtype=np.float32)
#HIGH_LAND = np.array([90, 160, 50], dtype=np.float32)
#MOUNTAIN = np.array([140, 120, 100], dtype=np.float32)
#SNOW = np.array([240, 245, 255], dtype=np.float32)
#DESERT = np.array([200, 175, 90], dtype=np.float32)
#ICE = np.array([210, 235, 255], dtype=np.float32)


#-------------------------------
# TEXTURE VISUALIZER
#-------------------------------

class TextureVisualizer():

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

    def render(self, screen, texture, sphere, rot, cx, cy, t):
        screen.fill((5, 5, 15))

        # Flattening the points (LAT*LON, 3)
        pts_flat = sphere.reshape(-1, 3)
        colors_flat = texture.reshape(-1, 3)
        rot_earth = rot @ self.pln.rotZ(OMEGA_EARTH * t)
        px, py, pz = self.pln.projection(pts_flat, rot_earth, cx, cy, RADIUS, FOCAL)

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

    def generateOneWebWalker(self, N_planes, N_per_plane, orbital_params):
        N_PLANES = N_planes
        N_PER_PLANE = N_per_plane  # 12 * 49 = 588, + some more in reserve to arrive to 648
        INCLINATION = orbital_params["INCLINATION"]

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
                    altitude_km=orbital_params['H'],
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
# SATELLITE RENDERING AND MATH
# -------------------------------

    def drawSat(self, screen, satellites, rot, cx, cy, t):
        for sat in satellites:
            pos = sat.position(t)
            if self.isHidden(rot @ pos):
                continue
            px, py, pz = self.pln.projection(pos.reshape(1, 3), rot, cx, cy, RADIUS, FOCAL)
            pygame.draw.circle(screen, (255, 50, 50), (int(px[0]), int(py[0])), 4)


    def connectTwoPoints(self, sa_point, satellites, rot, cx, cy, t, screen):       # Not used anymore
        sat_path = []
        distances = []
        pseudo_latency = 0
        for sat in satellites:
            sat_pos = sat.position(t)
            d = sqrt((sa_point[0][0] - sat_pos[0])**2 + (sa_point[0][1] - sat_pos[1])**2 + (sa_point[0][2] - sat_pos[2])**2)
            distances.append(d)
        min_index_start = 0
        for i in range(len(distances)):
            if distances[i] < distances[min_index_start]:
                min_index_start = i

        pseudo_latency += distances[min_index_start]
        start_point = np.array(sa_point[0])
        sat_point = satellites[min_index_start].position(t)
        sat_path.append(start_point)
        sat_path.append(sat_point)
        previous_distance_end = sqrt((sa_point[1][0] - sat_point[0])**2 + (sa_point[1][1] - sat_point[1])**2 + (sa_point[1][2] - sat_point[2])**2)

        new_sat_point = sat_point
        is_com_end = False

        while not is_com_end:
            sat_distances = []
            for sat in range(len(satellites)):
                sat_pos = satellites[sat].position(t)
                d = sqrt((new_sat_point[0] - sat_pos[0])**2 + (new_sat_point[1] - sat_pos[1])**2 + (new_sat_point[2] - sat_pos[2])**2)
                sat_distances.append([d, sat])
            near_sat = satellitesDistanceSorting(sat_distances)[1:15]

            end_distances = []
            for sat in near_sat:
                sat_pos = satellites[sat[1]].position(t)
                d = sqrt((sa_point[1][0] - sat_pos[0])**2 + (sa_point[1][1] - sat_pos[1])**2 + (sa_point[1][2] - sat_pos[2])**2)
                end_distances.append(d)
            min_index_end = 0
            for i in range(len(end_distances)):
                if end_distances[i] < end_distances[min_index_end]:
                    min_index_end = i

            d_current_to_end = sqrt((sa_point[1][0] - new_sat_point[0]) ** 2 + (sa_point[1][1] - new_sat_point[1]) ** 2 + (sa_point[1][2] - new_sat_point[2]) ** 2)
            if end_distances[min_index_end] < previous_distance_end and end_distances[min_index_end] < d_current_to_end:
                new_sat_point = satellites[near_sat[min_index_end][1]].position(t)
                sat_path.append(new_sat_point)
                pseudo_latency += end_distances[min_index_end]
                previous_distance_end = end_distances[min_index_end]
            else:
                is_com_end = True
                end_point = np.array(sa_point[1])
                f_sat = sat_path[-1]
                pseudo_latency += sqrt((sa_point[1][0] - f_sat[0])**2 + (sa_point[1][1] - f_sat[1])**2 + (sa_point[1][2] - f_sat[2])**2)
                sat_path.append(end_point)

        return sat_path, pseudo_latency

    def connectTwoPointsAst(self, sa_point, satellites, rot, cx, cy, t, N_sat, min_elevation_deg):
        isl_range = 2500/6371                         # 2500km is max range for modern sat

        positions = np.array([sat.position(t) for sat in satellites])
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
        d_mat = np.sqrt(np.sum(diff**2, axis=-1))
        adj_sat_mat = np.where(d_mat < isl_range, d_mat, np.inf)
        np.fill_diagonal(adj_sat_mat, np.inf)

        sat_path = []
        pseudo_latency = 0

        start_point = np.array(sa_point[0])
        stop_point = np.array(sa_point[1])

        min_index_sa = []
        for point in sa_point:
            distances = []
            current = np.array(point)
            best_index = -1
            best_dist = np.inf
            for i, sat in enumerate(satellites):
                sat_pos = sat.position(t)
                d = sqrt((current[0] - sat_pos[0]) ** 2 + (current[1] - sat_pos[1]) ** 2 + (
                            current[2] - sat_pos[2]) ** 2)
                dot_PS = current[0]*sat_pos[0] + current[1]*sat_pos[1] + current[2]*sat_pos[2]
                sin_E = (dot_PS - 1.0) / d if d > 0 else -1.0
                if sin_E >= np.sin(np.radians(min_elevation_deg)) and d < best_dist:
                    best_dist = d
                    best_index = i
            if best_index == -1:
                return [], np.inf

            pseudo_latency += best_dist
            min_index_sa.append(best_index)

        sat_path, sat_distances = Astar(positions, min_index_sa, adj_sat_mat, N_sat)

        if len(sat_path) == 0:
            return [], np.inf

        path = [start_point] + [satellites[i].position(t) for i in sat_path] + [stop_point]
        pseudo_latency += sat_distances[min_index_sa[1]]
        latency = ((float(pseudo_latency) * 6371) / 300000) + (len(path) * 0.005)

        return path, latency

    def coverageCalculation(self, satellites, t, min_elevation_deg, n_lat=30, n_lon=60): #min_elev_deg=27.6
        lats = np.linspace(-np.pi/2, np.pi/2, n_lat)
        lons = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
        lon_grid, lat_grid = np.meshgrid(lons, lats)

        rz = self.pln.rotZ(OMEGA_EARTH*t)
        px_fix = np.cos(lat_grid) * np.cos(lon_grid)
        py_fix = np.cos(lat_grid) * np.sin(lon_grid)
        pz = np.sin(lat_grid)
        points_fix = np.stack([px_fix, py_fix, pz], axis=-1)
        points = points_fix @ rz.T

        weight = np.cos(lat_grid)
        is_covered = np.zeros(lat_grid.shape, dtype=bool)

        for sat in satellites:
            diff = sat.position(t)[None, None, :] - points
            d = np.linalg.norm(diff, axis=-1)
            dot_PS = np.tensordot(points, sat.position(t), axes=([-1], [0]))

            with np.errstate(divide='ignore', invalid='ignore'):
                sin_E = np.where(d > 0, (dot_PS - 1.0) / d, -1.0)

            is_covered |= (sin_E >= np.sin(np.radians(min_elevation_deg)))
            if is_covered.all():
                break

        return np.sum(weight * is_covered) / np.sum(weight)

    def satCostEff(self, avg_coverage, N_planes, n_per_plane, cost_per_sat=1e6):
        if avg_coverage <= 0 :
            return np.inf
        total_cost = N_planes * n_per_plane * cost_per_sat + (ceil(N_planes*n_per_plane / 34)) * 100e6
        return total_cost / avg_coverage


# -------------------------------
# MAIN LOOP
# -------------------------------

    def main(self):
        screen = None
        clock = pygame.time.Clock()
        sa_point_angle = []
        mesure_number = 0
        N_hsimu = 4         # 0 if sim from start
        direct_distance = 0
        latencies = []
        pseudo_speed_ratios = []
        sat_coverages = []
        avg_pseudo_speed_ratio = []
        costs_for_Ns = []
        result_mesure_point = []
        sa_point = []
        N_planes = 8       # 8 to start
        N_per_plane = 19    # 19 to start
        orbital_params = {
            "INCLINATION" : 87.9,
            "RAAN" : None,
            "H" : 1400,      #600 to start
        }
        ACTIVATE_DISPLAY = False
        ACTIVATE_REFRESH = False

        # Creation of needed folders
        if not check_files('./', 'simu-saves'):
            os.mkdir('./simu-saves')
        if not check_files('./', 'result-curves'):
            os.mkdir('./result-curves')
        if not check_files('./', 'map-result'):
            os.mkdir('./map-result')

        if ACTIVATE_DISPLAY:
            ACTIVATE_REFRESH = True
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption('TIPE 3D VISUALIZATION')

        print("Texture generation ...")
        start = time.time()
        texture, lats, lons = self.pln.buildTexture(LAT_STEP, LON_STEP)
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

        # satellite generation
        satellites = self.generateOneWebWalker(N_planes, N_per_plane, orbital_params)

        ACTIVATE_ROTATION = True
        t = 0.0

        SHOW_MAP = True  # False to desactivate visualization
        if SHOW_MAP:
            self.tv.showTexture2D(texture, lats, lons, seed=SEED)

        print(f'Constellation : ({N_planes}, {N_per_plane}), Orbital : {orbital_params}')
        main_start = time.time()

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
                    elif event.key == pygame.K_a and ACTIVATE_DISPLAY:
                        ACTIVATE_REFRESH = False if ACTIVATE_REFRESH else True

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

            if ACTIVATE_REFRESH:
                self.render(screen, texture, sphere, rot, cx, cy, t)
                self.drawSat(screen, satellites, rot, cx, cy, t)

            if mesure_number % 250 == 0 and ACTIVATE_ROTATION:
                if mesure_number != 0:
                    # calcul of the ratio between pseudo latency and direct distance
                    valid_latencies = [i for i in latencies if np.isfinite(i)]
                    average_latency = np.average(valid_latencies) if len(valid_latencies) > 0 else np.inf
                    print(f'[{t}] Average pseudo latency between the two points : {average_latency}')
                    print(f'[{t}] Direct distance between the two points : {direct_distance}')
                    pseudo_speed_ratios.append(average_latency / direct_distance)     # / direct_distance

                if True:        # True = mesure on different distance for same N_planes
                    # Choose two point for communication + create [start, arrivial]
                    latencies = []
                    sa_point_angle = []
                    sa_point = []
                    for _ in range(2):
                        lat_index = random.randint(30, LAT_STEP - 30)
                        lon_index = random.randint(0, LON_STEP - 1)
                        lat_co = lats[lat_index]
                        lon_co = lons[lon_index]
                        sa_point_angle.append((lat_co, lon_co))
                        texture[lat_index, lon_index] = [0, 0, 0]
                        sa_point.append([np.cos(lat_co) * np.cos(lon_co), np.cos(lat_co) * np.sin(lon_co), np.sin(lat_co)])

                    # Direct distance between the 2 points on the globe (Haversine)
                    delta_lat = sa_point_angle[1][0] - sa_point_angle[0][0]
                    delta_lon = sa_point_angle[1][1] - sa_point_angle[0][1]
                    a_inter = np.sin(delta_lat/2)**2 + np.cos(sa_point_angle[0][0])*np.cos(sa_point_angle[1][0])*np.sin(delta_lon/2)**2
                    c_angle = 2 * atan2(sqrt(a_inter), sqrt(1-a_inter))
                    direct_distance = 6371 * c_angle

            # earth rotation managing
            rz = self.pln.rotZ(OMEGA_EARTH * t)
            sa_point = []
            for lat_co, lon_co in sa_point_angle:
                v = np.array([np.cos(lat_co) * np.cos(lon_co), np.cos(lat_co) * np.sin(lon_co), np.sin(lat_co)])
                sa_point.append((rz @ v).tolist())

            if mesure_number % 12500  == 0 and mesure_number != 0 and ACTIVATE_ROTATION: #12500
                #avg_sat_coverage = np.average(sat_coverages)
                avg_sat_coverage = min(sat_coverages)
                valid_pseudo_speed_ratios = [r for r in pseudo_speed_ratios if np.isfinite(r)]
                avg_pseudo_speed_ratio.append(np.average(valid_pseudo_speed_ratios))
                cost_for_N = self.satCostEff(avg_sat_coverage, N_planes, N_per_plane)
                costs_for_Ns.append(cost_for_N)
                result_mesure_point.append([(np.average(valid_pseudo_speed_ratios), cost_for_N), N_planes, N_per_plane, avg_sat_coverage])

                sat_coverages = []
                pseudo_speed_ratios = []

                if N_planes == 15 and N_per_plane == 69:
                    #simuScoreNplanes(avg_pseudo_speed_ratio, costs_for_Ns, N_planes, TEST_V)
                    #simuScoreNplanesNperplane(result_mesure_point, TEST_V, MIN_ELEVATION_DEG, 8e-14)
                    simuScoreNplanesNperplaneH(result_mesure_point, TEST_V, MIN_ELEVATION_DEG, 8e-14, orbital_params['H'], N_hsimu)
                    result_mesure_point = []

                    if N_hsimu == 8:        # 8 simu : H 600 -> 2000
                        main_end = time.time()
                        print(f"[SUCCESS] simulation completed successfully (duration : {main_end - main_start}s)")
                        ACTIVATE_ROTATION = False

                    N_hsimu += 1
                    mesure_number = 0
                    orbital_params["H"] += 200

                if N_per_plane == 69:
                    N_per_plane = 19
                    N_planes = N_planes + 1 if mesure_number != 0 else 8
                else :
                    N_per_plane += 10
                
                print(f'Config shifting : {N_planes}, {N_per_plane} \n',
                     f'Orbital param : {orbital_params} \n',
                     "Reset constellation position")
                satellites = self.generateOneWebWalker(N_planes, N_per_plane, orbital_params)


            sat_path, latency = self.connectTwoPointsAst(sa_point, satellites, rot, cx, cy, t, N_planes * N_per_plane, MIN_ELEVATION_DEG)
            if ACTIVATE_REFRESH:
                for k in range(len(sat_path) - 1):
                    pxS1, pyS1, pzS1 = self.pln.projection(sat_path[k].reshape(1, 3), rot, cx, cy, RADIUS, FOCAL)
                    pxS2, pyS2, pzS2 = self.pln.projection(sat_path[k + 1].reshape(1, 3), rot, cx, cy, RADIUS, FOCAL)
                    pygame.draw.line(screen, (255, 0, 255), (int(pxS1[0]), int(pyS1[0])), (int(pxS2[0]), int(pyS2[0])), 4)

            if ACTIVATE_ROTATION:
                latencies.append(latency)
                sat_coverages.append(self.coverageCalculation(satellites, t, MIN_ELEVATION_DEG))

                mesure_number += 1
                t += (1 / 60) * TIME_SCALE

            if ACTIVATE_REFRESH:
                pygame.display.flip()
            clock.tick(FPS)

if __name__ == "__main__":
    pygame.init()
    ml = Main()
    ml.main()
