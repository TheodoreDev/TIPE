# 🛰️ TIPE — Optimization & Resilience of LEO Satellite Mega-Constellations

> *A French classe préparatoire independent research project (TIPE), conducted with the help of the **CNES** (French national space agency), on the year's theme:* **Sobriété, Efficacité, Optimisation** *(Frugality, Efficiency, Optimization).*

A from-scratch Python simulator that builds a Walker-type LEO mega-constellation, routes traffic across it through a simulated inter-satellite-link (ISL) mesh network, and searches for the constellation design that minimizes cost while guaranteeing a target quality of service, then checks how that answer compares to a real deployed system (OneWeb or other).

---

## Table of Contents

- [Origin story: from forest ecosystems to satellite networks](#origin-story-from-forest-ecosystems-to-satellite-networks)
- [Research question](#research-question)
- [Repository structure](#repository-structure)
- [Architecture, part by part](#architecture-part-by-part)
- [Routing: from a greedy heuristic to A\*](#routing-from-a-greedy-heuristic-to-a)
- [Coverage & the elevation-angle constraint](#coverage--the-elevation-angle-constraint)
- [Costing the constellation](#costing-the-constellation)
- [Building a score that means something](#building-a-score-that-means-something)
- [Three-parameter exploration: `N_planes` × `N_per_plane` × altitude `H`](#three-parameter-exploration-n_planes--n_per_plane--altitude-h)
- [Confronting the model with reality: OneWeb](#confronting-the-model-with-reality-oneweb)
- [Physical realism: Earth actually spins](#physical-realism-earth-actually-spins)
- [A running log of bugs and open problems that shaped the methodology](#a-running-log-of-bugs-and-open-problems-that-shaped-the-methodology)
- [Running it](#running-it)
- [Research context & related work](#research-context--related-work)
- [Roadmap](#roadmap)
- [Acknowledgments](#acknowledgments)

---

## Origin story: from forest ecosystems to satellite networks

This project didn't start as a satellite simulator. It began in **January 2026** as an exploratory phase modeling resource-limited ecosystems, an attempt at simulating symbiotic forest–mycelium relationships (`.code/v1/tree.py`, still in the repo: three competing tree species, explicit Euler integration, Gaussian temperature response, logistic competition for resources). A Barnes–Hut N-body / galaxy-cluster simulation was also considered as an alternative at that stage.

The ecological model kept running into the same wall: the climate and biological sub-models needed to make it rigorous became too complex to responsibly finish within a *classe préparatoire* TIPE timeline, and the topic itself was proving hard to pin down to a crisp, defensible problem statement.

**On April 4, 2026**, the topic pivoted entirely toward **autonomous communication network optimization**, landing on the evaluation and optimization of satellite mega-constellations, a topic that maps naturally onto the year's theme by posing a very concrete dilemma: *the trade-off between network coverage/latency and the number of satellites placed in orbit.*

The forest simulator wasn't deleted, it's kept in the repo (`v1/`) as a record of that earlier direction, and is still exposed through the web interface alongside the current work.

---

## Research question

> **Given a fixed communication-quality requirement, what is the best LEO Walker constellation, in number of orbital planes, satellites per plane, altitude, ... ; that still delivers it, and how does that answer compare to what a real operator (OneWeb) actually built?**

The project is organized as four connected layers:

| Part | Focus                                                                                                 |
|---|-------------------------------------------------------------------------------------------------------|
| 1 | Orbital & geometric modeling : Keplerian mechanics, Walker Star constellations, coverage maps         |
| 2 | The network as a graph : satellites as nodes, ISL links as edges, shortest-path routing               |
| 3 | Combinatorial optimization : searching the design space, building a Pareto-style trade-off            |
| 4 | Resilience analysis : node failures, connectivity robustness *(in progress, see [Roadmap](#roadmap))* |

---

## Repository structure

```
.code/
  v1/
    tree.py             # Abandoned topic: forest growth ODE simulation (kept for reference)
    result/             # Saved outputs from that earlier phase
  v2/                    # Current TIPE: satellite constellation simulator
    tipe_v2.py           # Main simulation loop : orbital mechanics, routing, scoring, Pygame viewer
    satellites.py        # Satellite class: Keplerian position propagation
    planete.py            # Procedural Earth: Perlin-noise texture, sphere geometry, rotations, projection
    utils.py              # Dijkstra / A* pathfinding, misc. helpers
    simu.py               # Result plotting (matplotlib) & save/load of simulation runs
    save-visualizer.py     # Reloads saved .ted runs, re-applies the score/constraint, and plots the winner across all tested altitudes
  test-file/
    neural_net/           # From-scratch neural network (unrelated scratch project)
    function_testing.ipynb   # Jupyter notebook to test functions
```

---

## Architecture, part by part

### The planet

Earth's surface texture (oceans, continents, deserts, ice caps) is generated procedurally with **fractional Brownian motion over Perlin noise**, combined with latitude-dependent masks for tropical and polar biomes, no real map data is loaded, the whole globe is synthesized (`planete.py`, `class Planet`). A perspective projection with a configurable camera (mouse-controlled rotation) renders it as an interactive 3D scene via Pygame.

### The constellation

Each satellite (`satellites.py`, `class Satellite`) is parameterized by **inclination, RAAN (right ascension of the ascending node), and orbital phase**, and propagates its own position over time using real orbital mechanics, angular velocity derived from Kepler's third law:

```
ω = √(GM / a³),   a = R_Earth + altitude
```

The full constellation is generated Walker-Star style (`generateOneWebWalker`): `N_planes` orbital planes with RAAN evenly spread over 180° (the standard spacing for a near-polar Walker Star, since ascending/descending nodes coincide across planes 180° apart), each carrying `N_per_plane` satellites, defaulting to an 87.9° inclination and `H` km altitude.

### The network

At every simulated instant, satellites within a fixed inter-satellite laser range (2500 km) are connected into an adjacency graph, this graph is the network the routing algorithm operates on.

---

## Routing: from a greedy heuristic to A\*

The very first routing algorithm (`connectTwoPoints`, a greedy nearest-satellite-to-destination heuristic) produced a real, documented failure mode : as the constellation densified, it could get trapped in locally reasonable but globally poor choices, causing **latency to paradoxically increase** with more satellites, jagged, zig-zagging paths instead of sensible ones.

Switching to **Dijkstra** fixed that (a provably optimal shortest-path algorithm can't be worse off with more options), giving the expected monotonic-decreasing latency curve. The project then moved to **Astar**, using the Euclidean distance to the destination satellite as a heuristic, a minimal, surgical change on top of the existing Dijkstra implementation (swap the frontier-selection criterion from `g` to `f = g + h`), justified rigorously : since ISL edge weights are Euclidean distances, the straight-line heuristic satisfies the triangle inequality, making it both **admissible and consistent**, A\* is guaranteed to return the same optimal path as Dijkstra while typically exploring far fewer nodes.

---

## Coverage & the elevation-angle constraint

A ground point is only usefully connected to a satellite if that satellite sits above a **minimum elevation angle**, not just above the geometric horizon (0°), but high enough that the signal isn't degraded by atmospheric path length, terrain occlusion, or hardware beamwidth. This is expressed as a closed-form visibility test derived from the orbital geometry:

```
sin(E) = (P·S − 1) / ‖S − P‖   ≥   sin(E_min)
```

Coverage is then estimated by testing this condition over a latitude/longitude grid (weighted by `cos(latitude)` to correct for the pole-ward distortion of a lat/lon grid), averaged over an orbital period to smooth out the constellation's motion. The elevation threshold turns out to matter enormously: at a lenient 0–25°, coverage saturates almost immediately even for sparse constellations; realistic operational thresholds (the literature suggests values from 25° up to 55° depending on the source and system generation) are needed to see any meaningful trade-off at all.

---

## Costing the constellation

Cost is modeled as satellite hardware cost plus launch cost, with launches happening in **fixed-capacity batches** (mirroring a real launch vehicle's payload limit) rather than continuously, meaning cost is a step function of total satellite count, not a smooth line, with real consequences for where an "efficient" configuration sits relative to a launch's fill level.

---

## Building a score that means something

This is where most of the recent methodological work happened. In order:

1. **Normalized weighted sum** (`a·speed_norm + b·cost_norm`) : simple, but the weights are arbitrary and the normalization silently depends on the range of the tested grid.
2. **A monetized score** (`cost(€) + value_per_ms_lost(€/ms) × excess_latency(ms)`) : both terms in the same unit, avoiding arbitrary normalization; the free constant is explicit and its influence can be probed with a proper sensitivity sweep rather than hidden inside a blend ratio.
3. **A hard coverage constraint** replacing a third weighted penalty term: rather than stacking yet another calibrated constant, candidate configurations are first filtered to those meeting a minimum coverage requirement, and then the cheapest one among survivors is selected, closer to how a regulated operator actually reasons (meet the service requirement, then minimize cost), and it doesn't add a parameter that needs its own justification.



---

## Three-parameter exploration: `N_planes` * `N_per_plane` * altitude `H`

Sweeping `N_planes` alone, at a fixed satellites-per-plane count, can only ever describe a single cost/performance trajectory, there is no second configuration at the same cost to compare against, so no real trade-off can ever emerge from a 1D sweep. The project first moved to a 2D grid search over `(N_planes, N_per_plane)`, then extended it to a third dimension: orbital **altitude**, since the earlier ISL-range/altitude finding (see bug log) showed that altitude alone can flip a constellation from well-connected to badly fragmented.

Each altitude is run as its own full `(N_planes, N_per_plane)` sweep and appended to an indexed save file (`simuScoreNplanesNperplaneH`, `utils.adding`), rather than overwriting a single result, so a whole batch of altitude scenarios can be queued and run unattended.

`save-visualizer.py` reloads that batch afterwards: for **each altitude**, it re-applies the hard coverage constraint and picks the best `(N_planes, N_per_plane)` under it, then compares those per-altitude winners against each other to report a single global optimum, effectively a nested optimization, altitude on the outside, plane configuration on the inside. Results are visualized as 3D scatter plots (score and coverage against `N_planes`/`N_per_plane`), titled with the winning altitude, with the global optimum marked explicitly.

---

## Confronting the model with reality: OneWeb

OneWeb's actual deployed configuration (12 planes × 49 satellites/plane, +spares to 648) is used throughout as a real-world reference point. The optimizer's answer doesn't match it exactly, and that gap turned into a research question in its own right rather than a bug to chase away:

- With the model's coverage constraint as a **minimum** (in space and time) and 2 parameter (N_planes, N_per_plane) ; we get the best constellation at (12 planes, 39 satellites per plane) which is near reality. The difference can be explain by the fact that OneWebb has more satellite than, needed to be sure that every place on earth is constantly covered.
- The elevation-angle threshold tested may still be more lenient than OneWeb's real operational value.
- The model has no notion of **capacity** (throughput per satellite, user demand per region) or **handover redundancy** (needing two satellites in view at once for a seamless switch), both push real designs toward more satellites than pure geometric coverage requires.

Testing **3-parameter** exploration and implementing **satellite capacity**, are the next concrete steps to quantify how much of the gap each factor explains.

---

## Physical realism: Earth actually spins

An early version of the simulator implicitly treated ground points as fixed in the same inertial frame as the satellites' orbits, correct for the satellites (whose RAAN/inclination are properly defined in an Earth-Centered Inertial frame), wrong for ground points, which are fixed in the **rotating** Earth frame. A `rotZ` rotation (Earth's own spin, using the *sidereal* day : 86164 s, not the 24 h solar day) is now applied to ground points and to the visible texture before satellite visibility and routing are computed, so a fixed city on the map now correctly sweeps underneath the orbiting constellation instead of staying artificially locked to it.

---

## A running log of bugs and open problems that shaped the methodology

Two different kinds of issues showed up along the way : plain implementation bugs, and bigger modeling/methodology dead ends that took real reasoning (not just a code fix) to work through. Both mattered enough to the project's direction to be worth keeping on record.

### Implementation bugs

| Bug | Symptom | Root cause |
|---|---|---|
| Pixel radius vs. real Earth radius | Nonsensical absolute latency values | Display variable (`RADIUS`, in pixels) used in place of the real Earth radius in a physics calculation |
| Speed-of-light unit mismatch | Propagation delay ~1000× too small | Distance in km divided by *c* in m/s instead of km/s |
| Longitude range bug | Coverage grid only sampled half the globe | `linspace(-π/2, π/2, ...)` instead of `(-π, π, ...)` |
| No horizon/elevation check | Latency looked artificially good at low satellite counts | Nearest-satellite search never verified the satellite was actually above the visibility threshold |
| Accumulator lists never reset | Every result curve looked artificially smooth and monotonic | Running cumulative averages across *all* previous `N_planes` values, never cleared between sweep steps |
| Ground points not rotating with Earth | Coverage/latency computed in the wrong frame | Satellites correctly propagated in an inertial frame; ground points stayed static instead of spinning with the sidereal day |
| Hardcoded satellites-per-plane | `ValueError` on any 2D sweep | Constellation generator ignored the new `N_per_plane` parameter, silently always building 49/plane |
| Floor instead of ceiling for launch count | Systematic ~100M€ under-costing | `N_sat // sat_per_launch` instead of `ceil(N_sat / sat_per_launch)` |
| Inconsistent return type on pathfinding failure | `TypeError: 'float' object is not subscriptable` | `Astar` returned a bare `np.inf` instead of `([], np.inf)` in the no-path-found case, only ever triggered once altitude changes made that case common |
| Coverage saturated across the whole test grid | 3D coverage plot looked completely flat near 1.0 | Default elevation threshold too lenient (27.6°) *combined with* a `N_per_plane` sweep that never tested sparse values below 29 |

### Modeling & methodology dead ends

| Problem                                                    | What went wrong                                                                                      | Resolution / finding                                                                                                                                                                                                                                                                           |
|------------------------------------------------------------|------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Greedy routing                                             | Latency *increased* as the constellation got denser : jagged, zig-zagging paths                      | Replaced by Dijkstra (provably optimal -> can't get worse with more options), then by A\* with an admissible Euclidean-distance heuristic                                                                                                                                                      |
| `latency × cost` never showed a U-shaped minimum           | Score just kept climbing, however the two curves were combined                                       | Elasticity analysis showed coverage's elasticity w.r.t. total satellite count stays below 1 across the whole practically-testable range, cost's relative growth structurally outpaces latency's relative decline everywhere tested, so no interior minimum can exist in a 1-parameter sweep    |
| Arbitrary weighted-sum score                               | Weights `a`/`b` had no justification, and normalization silently depended on the tested grid's range | Replaced with a monetized score (`cost(€) + value_per_ms × excess_latency`), same unit throughout, explicit and probeable via a sensitivity sweep instead of a hidden blend ratio                                                                                                              |
| Score completely insensitive to latency                    | Optimum didn't move even with a huge weighting constant (2×10¹¹)                                     | Quantified that cost represented >99% of the score at that scale; constant raised ~3 orders of magnitude (to 8×10¹⁴) to get a genuinely interior optimum                                                                                                                                       |
| Coverage absent from the score                             | The "best" configuration found didn't actually reach full coverage                                   | Coverage never entered the score at all, only a separate display, fixed by filtering to configurations meeting a minimum coverage requirement *before* minimizing cost, instead of stacking another arbitrary weighted penalty                                                                 |
| Optimum didn't match real OneWeb's per-plane density       | `(12, 19)` came out on top instead of anything close to `(12, 49)`                                   | A suspected launch-cost bug was checked and ruled out (it discounts nearly the whole grid uniformly); more likely explanation is a genuine Walker-Star geometry effect, more planes can fill longitude coverage gaps more efficiently than more satellites per plane, for the same total count |
| Optimum constellation far lighter than real OneWeb overall | Even after all fixes, the model's answer stayed well under real deployed satellite counts            | Traced to the coverage constraint being an *average* (space & time) rather than a *worst-case* guarantee, a possibly still-too-lenient elevation threshold, and the complete absence of capacity/throughput and handover-redundancy requirements in the model                                  |
| A* mistake on high altitude                                | Bizarre, zig-zagging (or outright broken) A\* paths at higher altitude                               | A fixed 2500 km range connects proportionally fewer satellite pairs as orbital radius, and therefore real inter-satellite spacing, grows with altitude; confirmed by directly counting isolated satellites in the ISL graph at two altitudes. It's not a bug but something normal              |

---

## Running it

### Dependencies

| Package | Used for                                                                                |
|---|-----------------------------------------------------------------------------------------|
| `numpy` | Vectorized geometry, orbital mechanics, ISL adjacency matrices, coverage grids          |
| `pygame` | Interactive 3D viewer (planet rendering, satellite/route display, mouse controls)       |
| `matplotlib` | Result plots, forest growth curves, 3D score/coverage scatter plots, Pareto-style plots |
| `scipy` | ODE integration (legacy forest-growth simulation, `v1/tree.py`)                         |

Everything else imported across the codebase (`math`, `random`, `sys`, `time`, `json`, `pickle`, `os`, `ast`, `re`) is Python standard library, no separate install needed. `mpl_toolkits.mplot3d` ships with `matplotlib` itself, not a separate package.

**Python version :** 3.12

Install everything with :
```bash
pip install numpy pygame matplotlib scipy
```

Or, with a `requirements.txt` :
```
numpy
pygame
matplotlib
scipy
```
```bash
pip install -r requirements.txt
```

### Starting interactive 3D viewer (desktop, Pygame)
```bash
cd .code/v2
python3 tipe_v2.py
```

---

## Research context & related work

Early industrial grounding came from a discussion with **Jean-Philippe Taisant**, SatCom Program Manager at CNES, which helped anchor the simulation in real constraints operators actually face, fleet maintenance (OneWeb, Iris²), collision risk (Kessler syndrome), and failure resilience.

A literature search turned up closely related academic work worth citing for comparison, most notably a shortest-path (Dijkstra-based) ISL latency simulator very close in spirit to this project's routing layer, a mega-constellation network performance platform benchmarking Starlink/OneWeb/Telesat directly, and several Pareto-front studies optimizing LEO constellation design over the same parameters used here (number of planes, satellites, altitude). None combine this project's exact mix of a hand-built A\*/Dijkstra router, an elevation-angle coverage constraint, and a realistic batched-launch cost model swept over a full `(N_planes, N_per_plane)` grid against a real deployed reference, which is this project's own methodological contribution.

---

## Roadmap

- [ ] **Three-parameter** simulation to see what altitude is the best according to the algorithm
- [ ] **Resilience analysis** (Part 4): random vs. targeted node removal, algebraic connectivity (λ₂ of the graph Laplacian), max-flow / k-connectivity between regions (Ford–Fulkerson)
- [ ] Worst-case (rather than time-averaged) coverage as the hard constraint, to test the OneWeb-gap hypothesis directly
- [ ] Settle on a single, well-sourced elevation threshold (current runs have used several values across different tests : 25°, 40°, 45°, 55°, needs to be pinned down and justified once, not switched between)
- [ ] Make the ISL laser range altitude-aware (or explicitly document it as a fixed hardware constant) inside the `(N_planes, N_per_plane, H)` joint sweep, rather than treating altitude's connectivity impact as a separate side-finding
- [ ] Statistical rigor: uncertainty bars on sampled metrics, systematic sensitivity analysis on the monetized score's free constant
- [ ] Validation against publicly reported real-world Starlink/OneWeb latency figures
- [ ] Capacity/throughput and handover-redundancy constraints
- [ ] J2 perturbation (nodal precession) for longer-timescale orbital realism
 
---

## Acknowledgments

---

[![forthebadge](http://forthebadge.com/images/badges/built-with-love.svg)](#)  [![forthebadge](http://forthebadge.com/images/badges/powered-by-electricity.svg)](#)
