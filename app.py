import io
import base64
import sys
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from scipy.integrate import solve_ivp
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.code', 'v1'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.code', 'v2'))


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/forest', methods=['POST'])
def run_forest():
    data = request.get_json(silent=True) or {}
    temperature = float(data.get('temperature', 15.0))
    co2 = float(data.get('co2', 0.04))
    duration = int(data.get('duration', 100))
    n_trees = int(data.get('n_trees', 10))
    surface = float(data.get('surface', 1.0))

    from tree import TreeSpecies, ForestSimulation, ESPECES

    simulation = ForestSimulation(
        especes=ESPECES,
        temperature=temperature,
        co2=co2,
        duree=duration,
        surface_initiale=surface,
        n_arbres_initial=n_trees,
    )
    resultats = simulation.simuler()

    t = resultats['temps']
    esp = simulation.especes
    couleurs = ["#2e7d32", "#1565c0", "#b71c1c"]

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        f"Simulation de croissance forestière\n"
        f"T = {temperature} °C  |  CO₂ = {co2*100:.2f} %  |  Durée = {duration} ans",
        fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, resultats["surface_totale"], color="#2e7d32", linewidth=2)
    ax1.fill_between(t, resultats["surface_totale"], alpha=0.15, color="#2e7d32")
    ax1.set_xlabel("Temps (années)")
    ax1.set_ylabel("Surface (ha)")
    ax1.set_title("Surface totale de la forêt")
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, resultats["biomasse_totale"], color="#795548", linewidth=2)
    ax2.fill_between(t, resultats["biomasse_totale"], alpha=0.15, color="#795548")
    ax2.set_xlabel("Temps (années)")
    ax2.set_ylabel("Biomasse (tonnes)")
    ax2.set_title("Biomasse totale de la forêt")
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 0])
    for i, (e, c) in enumerate(zip(esp, couleurs)):
        ax3.plot(t, resultats["N"][i], color=c, linewidth=2, label=e.nom)
    ax3.set_xlabel("Temps (années)")
    ax3.set_ylabel("Nombre d'arbres")
    ax3.set_title("Évolution par espèce — population")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    l1, = ax4.plot(t, resultats["densite"], color="#6a1b9a", linewidth=2, label="Densité (arbres/ha)")
    l2, = ax4_twin.plot(t, resultats["croissance_moyenne"], color="#f57c00", linewidth=2,
                        linestyle="--", label="Croissance moy. (t/arbre/an)")
    ax4.set_xlabel("Temps (années)")
    ax4.set_ylabel("Densité (arbres/ha)", color="#6a1b9a")
    ax4_twin.set_ylabel("Croissance moy. (t/arbre/an)", color="#f57c00")
    ax4.set_title("Densité & croissance moyenne par arbre")
    lines = [l1, l2]
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax4.grid(True, alpha=0.3)

    img = fig_to_base64(fig)

    t_fin = -1
    summary = {
        'surface': round(float(resultats['surface_totale'][t_fin]), 3),
        'biomasse': round(float(resultats['biomasse_totale'][t_fin]), 2),
        'densite': round(float(resultats['densite'][t_fin]), 1),
        'n_total': int(resultats['N_total'][t_fin]),
        'croissance': round(float(resultats['croissance_moyenne'][t_fin]), 4),
        'species': [
            {
                'nom': e.nom,
                'N': int(resultats['N'][i, t_fin]),
                'B': round(float(resultats['B'][i, t_fin]), 2),
                'fT': round(float(simulation.f_temperature(e)), 3),
                'fCO2': round(float(simulation.f_co2(e)), 3),
            }
            for i, e in enumerate(esp)
        ]
    }

    return jsonify({'image': img, 'summary': summary})


@app.route('/api/planet', methods=['POST'])
def run_planet():
    data = request.get_json(silent=True) or {}
    seed = int(data.get('seed', 42))

    class Perlin:
        def __init__(self, seed):
            perm = np.random.RandomState(seed).permutation(256)
            self._perm = np.concatenate([perm, perm]).astype(np.int32)

        def fade(self, t):
            return t * t * t * (t * (t * 6 - 15) + 10)

        def lerp(self, a, b, t):
            return a + t * (b - a)

        def grad2(self, h, x, y):
            h = h & 3
            u = np.where(h < 2, x, y)
            v = np.where(h < 2, y, x)
            return np.where(h & 1, -u, u) + np.where(h & 2, -v, v)

        def noise2(self, x, y):
            X = np.floor(x).astype(np.int32) & 255
            Y = np.floor(y).astype(np.int32) & 255
            xf = x - np.floor(x)
            yf = y - np.floor(y)
            u = self.fade(xf)
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

    class PlanetSim:
        def __init__(self, seed):
            self.perlin = Perlin(seed)

        def buildTexture(self):
            LAT_STEP = 120
            LON_STEP = 240
            lats = np.linspace(-np.pi/2, np.pi/2, LAT_STEP)
            lons = np.linspace(-np.pi, np.pi, LON_STEP)
            lon_grid, lat_grid = np.meshgrid(lons, lats)

            scale = 1.8
            nx = np.cos(lat_grid) * np.cos(lon_grid) * scale
            ny = np.cos(lat_grid) * np.sin(lon_grid) * scale
            nz = np.sin(lat_grid) * scale

            h = self.perlin.fbm(nx+nz*0.3, ny+nz*0.3, octaves=7)
            continent_base = self.perlin.fbm(nx * 0.7, ny * 0.5, octaves=4)
            h_combined = continent_base * 0.65 + h * 0.35
            land_mask = h_combined > 0.008

            lat_d = np.degrees(lat_grid)

            coast_noise = self.perlin.fbm(nx*2, ny*2, octaves=4)*0.15
            land_mask = (land_mask & (h > -0.05 + coast_noise)) | (h > 0.27)

            DEEP_OCEAN = np.array([10, 60, 120], dtype=np.float32)
            OCEAN = np.array([30, 100, 160], dtype=np.float32)
            LOW_LAND = np.array([60, 120, 50], dtype=np.float32)
            HIGH_LAND = np.array([90, 160, 50], dtype=np.float32)
            MOUNTAIN = np.array([140, 120, 100], dtype=np.float32)
            SNOW = np.array([240, 245, 255], dtype=np.float32)
            DESERT = np.array([200, 175, 90], dtype=np.float32)
            ICE = np.array([210, 235, 255], dtype=np.float32)

            ocean_t = np.clip((h + 0.4) / 0.5, 0, 1)
            ocean_col = DEEP_OCEAN[None, None, :] + ocean_t[:, :, None]*(OCEAN-DEEP_OCEAN)

            elevation = np.clip((h_combined - 0.05) / 0.35, 0, 1)
            land_col = np.where(elevation[:, :, None] < 0.35,
                                LOW_LAND + (elevation[:, :, None]/0.35) * (HIGH_LAND - LOW_LAND),
                                np.where(elevation[:, :, None] < 0.65,
                                         HIGH_LAND + ((elevation[:, :, None]-0.35)/0.3) * (MOUNTAIN - HIGH_LAND),
                                         MOUNTAIN + ((elevation[:, :, None]-0.65)/0.35) * (SNOW - MOUNTAIN)))

            tropical_factor = np.exp(-((lat_d / 25.0) ** 2))
            desert_noise = self.perlin.fbm(nx * 1.5 + 5.3, ny * 1.5 + 2.7)
            desert_mask = land_mask & (desert_noise * tropical_factor < -0.01)
            land_col = np.where(desert_mask[:, :, None], DESERT, land_col)

            polar_noise_N = self.perlin.fbm(nx * 2.0 + 8.1, ny * 2.0 + 4.6, octaves=4)
            polar_noise_S = self.perlin.fbm(nx * 2.0 + 13.7, ny * 2.0 + 9.2, octaves=4)

            polar_factor_N = np.exp(-(((lat_d - 90) / 22.0) ** 2))
            polar_factor_S = np.exp(-(((lat_d + 90) / 22.0) ** 2))

            ice_mask = (polar_noise_N * polar_factor_N > 0.03) & (lat_d > 50) | \
                       (polar_noise_S * polar_factor_S > 0.005) & (lat_d < -50) | \
                       (lat_d < -75)

            colors = np.where(ice_mask[:, :, None], ICE,
                              np.where(land_mask[:, :, None], land_col, ocean_col))

            return np.clip(colors, 0, 255).astype(np.uint8), lats, lons

    ps = PlanetSim(seed)
    texture, lats, lons = ps.buildTexture()

    fig, ax = plt.subplots(figsize=(14, 7), facecolor='#05050F')
    ax.set_facecolor('#05050F')

    lat_deg = np.degrees(lats)
    lon_deg = np.degrees(lons)
    extent = [lon_deg[0], lon_deg[-1], lat_deg[0], lat_deg[-1]]

    ax.imshow(texture, origin='lower', extent=extent, aspect='auto', interpolation='bilinear')
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.grid(color='white', linewidth=0.3, alpha=0.3, linestyle='--')

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
        ax.axhline(lat_val, color=colors_lat[name], linewidth=0.8, linestyle=':', alpha=0.7)
        ax.text(182, lat_val, name, color=colors_lat[name], fontsize=6.5, va='center', clip_on=False)

    ax.set_xlabel('Longitude', color='#A0C8FF', fontsize=10)
    ax.set_ylabel('Latitude', color='#A0C8FF', fontsize=10)
    ax.tick_params(colors='#A0C8FF', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#A0C8FF')

    ax.set_title(f'Projection de Mercator — seed {seed}', color='white', fontsize=13, pad=10)

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
    ax.legend(handles=legend_items, loc='lower left', facecolor='#0D0D1A',
              edgecolor='#A0C8FF', labelcolor='white', fontsize=8, framealpha=0.8)

    plt.tight_layout()
    img = fig_to_base64(fig)

    return jsonify({'image': img, 'seed': seed})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
