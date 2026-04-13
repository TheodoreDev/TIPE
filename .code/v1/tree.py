import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
import pickle


# FONCTION D'ECRITURE ET DE LECTURE DES FICHIERS DE SAUVEGARDE

def writing(filepath, obj):
    file_obj = open(filepath, "wb")
    pickle.dump(obj, file_obj)

def reading(filepath):
    file_obj = open(filepath, "rb")
    obj = pickle.load(file_obj)
    print(obj)

# PARAMÈTRES GLOBAUX DE L'ENVIRONNEMENT
# Ces paramètres définissent les conditions et les paramètres de la simulation.

TEMPERATURE_MOYENNE = 15.0      # °C — température ambiante moyenne
CO2_CONCENTRATION = 0.04        # % volumique (400 ppm ≈ 0.04 %)
O2_CONCENTRATION = 20.6         # % volumique
N2_CONCENTRATION = 79.0         # % volumique

DUREE_SIMULATION = 100         # années
SURFACE_INITIALE = 1.0          # ha — surface de départ de la forêt
N_ARBRES_INITIAL = 10          # nombre total d'arbres à t=0

# CLASSE TreeSpecies — Paramètres biologiques par espèce

class TreeSpecies:

    def __init__(self, nom, taux_croissance_max, temperature_optimale,
                 sigma_temperature, sensibilite_co2, taux_mortalite_naturelle,
                 biomasse_initiale_par_arbre, surface_par_arbre, capacite_charge):
        self.nom = nom
        self.taux_croissance_max = taux_croissance_max
        self.temperature_optimale = temperature_optimale
        self.sigma_temperature = sigma_temperature
        self.sensibilite_co2 = sensibilite_co2
        self.taux_mortalite_naturelle = taux_mortalite_naturelle
        self.biomasse_initiale = biomasse_initiale_par_arbre
        self.surface_par_arbre = surface_par_arbre
        self.capacite_charge = capacite_charge  # arbres/ha

# DÉFINITION DES ESPÈCES (3 types distincts)

ESPECES = [
    TreeSpecies(
        nom = "Chêne pédonculé",
        taux_croissance_max = 0.08,             # croissance lente, bois dense
        temperature_optimale = 14.0,
        sigma_temperature = 6.0,                # tolérant aux variations thermiques
        sensibilite_co2 = 1.2,                  # modérément sensible au CO₂
        taux_mortalite_naturelle = 0.005,       # très résistant (0.5 %/an)
        biomasse_initiale_par_arbre = 0.5,      # tonnes/arbre
        surface_par_arbre = 0.006,              # ha/arbre (60 m2)
        capacite_charge = 200,                  # arbres/ha max
    ),
    TreeSpecies(
        nom = "Pin sylvestre",
        taux_croissance_max = 0.14,         # croissance rapide
        temperature_optimale = 10.0,        # espèce de montagne / froide
        sigma_temperature = 5.0,
        sensibilite_co2 = 1.5,              # très sensible au CO₂ (fertilisation)
        taux_mortalite_naturelle = 0.012,   # moins robuste (1.2 %/an)
        biomasse_initiale_par_arbre = 0.3,
        surface_par_arbre = 0.004,          # ha/arbre (40 m2))
        capacite_charge = 350,
    ),
    TreeSpecies(
        nom = "Hêtre commun",
        taux_croissance_max = 0.10,         # croissance intermédiaire
        temperature_optimale = 12.0,
        sigma_temperature = 4.0,            # assez sensible aux écarts de T
        sensibilite_co2 = 1.3,
        taux_mortalite_naturelle = 0.008,   # mortalité intermédiaire (0.8 %/an)
        biomasse_initiale_par_arbre = 0.4,
        surface_par_arbre = 0.005,          # ha/arbre (50 m2))
        capacite_charge = 250,
    ),
]

# CLASSE ForestSimulation — Intégration temporelle et résultats

class ForestSimulation:

    def __init__(self, especes, temperature, co2, duree,
                 surface_initiale=1.0, n_arbres_initial=200):
        self.especes = especes
        self.temperature = temperature
        self.co2 = co2
        self.duree = duree
        self.surface_initiale = surface_initiale
        self.n_arbres_initial = n_arbres_initial
        self.n_especes = len(especes)
        self.dt = 1.0   # pas de temps annuel (années)

        # Résultats stockés ici après simulation
        self.resultats = None

    # FONCTIONS DU MODÈLE MATHÉMATIQUE

    def f_temperature(self, espece):

        T = self.temperature
        T_opt = espece.temperature_optimale
        sigma = espece.sigma_temperature
        return np.exp(-((T - T_opt) ** 2) / (2 * sigma ** 2))

    def f_co2(self, espece):

        CO2_ref = 0.028   # % volumique — niveau pré-industriel de référence
        delta = (self.co2 - CO2_ref) / CO2_ref
        return max(0.0, 1.0 + espece.sensibilite_co2 * delta)

    # INTÉGRATION TEMPORELLE (schéma d'Euler explicite)

    def simuler(self):

        n_steps = int(self.duree / self.dt) + 1
        temps = np.linspace(0, self.duree, n_steps)

        # Répartition équilibrée des arbres entre les espèces
        n_par_espece = self.n_arbres_initial // self.n_especes

        # Tableaux de résultats : shape (n_especes, n_steps)
        N = np.zeros((self.n_especes, n_steps))   # nombre d'arbres
        B = np.zeros((self.n_especes, n_steps))   # biomasse (tonnes)

        # Conditions initiales
        for i, esp in enumerate(self.especes):
            N[i, 0] = n_par_espece
            B[i, 0] = n_par_espece * esp.biomasse_initiale

        # Pré-calcul des facteurs environnementaux (constants ici)
        fT = np.array([self.f_temperature(esp) for esp in self.especes])
        fCO2 = np.array([self.f_co2(esp) for esp in self.especes])

        # Auler explicite
        for t in range(1, n_steps):
            for i, esp in enumerate(self.especes):

                # Capacité de charge effective (arbres, pas arbres/ha)
                K = esp.capacite_charge * self.surface_initiale

                # Facteur logistique : limite la croissance quand la densité → K
                # Modélise la compétition pour la lumière, l'eau, les nutriments
                facteur_logistique = max(0.0, 1.0 - N[i, t-1] / K)

                # Taux de croissance effectif
                r_eff = esp.taux_croissance_max * fT[i] * fCO2[i]

                # Équation discrète pour N (Euler)
                # dN/dt = (r_eff - mu) · N · (1 - N/K)
                dN = (r_eff - esp.taux_mortalite_naturelle) \
                     * N[i, t-1] * facteur_logistique
                N[i, t] = max(0.0, N[i, t-1] + self.dt * dN)

                # Équation discrète pour B (Euler)
                # dB/dt = r_eff · B · (1 - N/K) - mu · B
                dB = r_eff * B[i, t-1] * facteur_logistique \
                     - esp.taux_mortalite_naturelle * B[i, t-1]
                B[i, t] = max(0.0, B[i, t-1] + self.dt * dB)

        # Surface totale (ha) = somme sur les espèces de N × surface/arbre
        surfaces = np.array([
            N[i] * esp.surface_par_arbre
            for i, esp in enumerate(self.especes)
        ])
        surface_totale = surfaces.sum(axis=0)

        # Biomasse totale (tonnes)
        biomasse_totale = B.sum(axis=0)

        # Nombre total d'arbres
        N_total = N.sum(axis=0)

        # Densité d'arbres (arbres/ha)
        densite = np.where(surface_totale > 0, N_total / surface_totale, 0)

        # Croissance moyenne par arbre (tonnes/arbre/an) — dérivée de B/N
        with np.errstate(invalid='ignore', divide='ignore'):
            biomasse_par_arbre = np.where(N_total > 0, biomasse_totale / N_total, 0)
        croissance_moyenne = np.gradient(biomasse_par_arbre, self.dt)

        self.resultats = {
            "temps" : temps,
            "N" : N,                        # arbres par espèce
            "B" : B,                        # biomasse par espèce
            "surface_totale" : surface_totale,            
            "biomasse_totale" : biomasse_totale,
            "N_total" : N_total,
            "densite" : densite,
            "croissance_moyenne" : croissance_moyenne,
        }
        return self.resultats

# VISUALISATION

def visualiser(sim, resultats):

    t = resultats["temps"]
    esp = sim.especes
    couleurs = ["#2e7d32", "#1565c0", "#b71c1c"]

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Simulation de croissance forestière\n"
        f"T = {sim.temperature} °C  |  CO₂ = {sim.co2*100:.2f} %  |  Durée = {sim.duree} ans",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # Graphe 1 : Surface totale vs temps
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, resultats["surface_totale"], color="#2e7d32", linewidth=2)
    ax1.fill_between(t, resultats["surface_totale"], alpha=0.15, color="#2e7d32")
    ax1.set_xlabel("Temps (années)")
    ax1.set_ylabel("Surface (ha)")
    ax1.set_title("Surface totale de la forêt")
    ax1.grid(True, alpha=0.3)

    # Graphe 2 : Biomasse totale vs temps
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, resultats["biomasse_totale"], color="#795548", linewidth=2)
    ax2.fill_between(t, resultats["biomasse_totale"], alpha=0.15, color="#795548")
    ax2.set_xlabel("Temps (années)")
    ax2.set_ylabel("Biomasse (tonnes)")
    ax2.set_title("Biomasse totale de la forêt")
    ax2.grid(True, alpha=0.3)

    # Graphe 3 : Comparaison des espèces (nombre d'arbres)
    ax3 = fig.add_subplot(gs[1, 0])
    for i, (e, c) in enumerate(zip(esp, couleurs)):
        ax3.plot(t, resultats["N"][i], color=c, linewidth=2, label=e.nom)
    ax3.set_xlabel("Temps (années)")
    ax3.set_ylabel("Nombre d'arbres")
    ax3.set_title("Évolution par espèce — population")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Graphe 4 : Densité et croissance moyenne par arbre
    ax4 = fig.add_subplot(gs[1, 1])
    ax4_twin = ax4.twinx()
    l1, = ax4.plot(t, resultats["densite"],
                   color="#6a1b9a", linewidth=2, label="Densité (arbres/ha)")
    l2, = ax4_twin.plot(t, resultats["croissance_moyenne"],
                         color="#f57c00", linewidth=2, linestyle="--",
                         label="Croissance moy. (t/arbre/an)")
    ax4.set_xlabel("Temps (années)")
    ax4.set_ylabel("Densité (arbres/ha)", color="#6a1b9a")
    ax4_twin.set_ylabel("Croissance moy. (t/arbre/an)", color="#f57c00")
    ax4.set_title("Densité & croissance moyenne par arbre")
    lines = [l1, l2]
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=9)
    ax4.grid(True, alpha=0.3)

    plt.savefig("./result/forest_simulation.png",
                dpi=150, bbox_inches="tight")
    print("Figure sauvegardée : forest_simulation.png")
    plt.show()

# AFFICHAGE DES RÉSULTATS NUMÉRIQUES

def afficher_resume(sim, resultats):

    t_fin = -1   # dernier pas de temps
    print("\n" + "="*60)
    print(f"  RÉSULTATS À t = {sim.duree} ans")
    print("="*60)
    print(f"  Surface totale : {resultats['surface_totale'][t_fin]:.3f} ha")
    print(f"  Biomasse totale : {resultats['biomasse_totale'][t_fin]:.2f} t")
    print(f"  Densité moyenne : {resultats['densite'][t_fin]:.1f} arbres/ha")
    print(f"  Nombre total arbres : {resultats['N_total'][t_fin]:.0f}")
    print(f"  Croissance moy./arbre: {resultats['croissance_moyenne'][t_fin]:.4f} t/arbre/an")
    print("-"*60)
    print("  Détail par espèce :")
    for i, esp in enumerate(sim.especes):
        print(f"    [{esp.nom:20s}] "
              f"N={resultats['N'][i, t_fin]:.0f}  "
              f"B={resultats['B'][i, t_fin]:.2f} t  "
              f"fT={sim.f_temperature(esp):.3f}  "
              f"fCO2={sim.f_co2(esp):.3f}")
    print("="*60)

# SAUVEGARDE DES DONNEES DANS UN FICHIER .ted

def save(resultats, filename):
    writing(f'./result/{filename}.ted', resultats)
    print(f"Fichier de donnée sauvegardée : {filename}.ted")

# POINT D'ENTRÉE PRINCIPAL

if __name__ == "__main__":

    # Instance de la simulation avec les paramètres globaux
    simulation = ForestSimulation(
        especes = ESPECES,
        temperature = TEMPERATURE_MOYENNE,
        co2 = CO2_CONCENTRATION,
        duree = DUREE_SIMULATION,
        surface_initiale = SURFACE_INITIALE,
        n_arbres_initial = N_ARBRES_INITIAL,
    )

    print("Lancement de la simulation forestière...")
    resultats = simulation.simuler()

    afficher_resume(simulation, resultats)
    visualiser(simulation, resultats)
    save([simulation, resultats], filename="foret1")
