# TIPE — Simulations Scientifiques

## Overview

An application presenting scientific simulations developed as a TIPE (Travail d'Initiative Personnelle Encadré — French preparatory class project). The app wraps Python simulation scripts.

## Project Structure

```
app.py                  # Flask web server (main entry point) (Replit)
templates/index.html    # Web UI (single-page app) (Replit)
.code/
  v1/
    tree.py             # Forest growth simulation (Euler integration)
    result/             # Saved simulation outputs
  v2/
    main.py             # 3D planet visualizer (Pygame-based, not used in web)
    visualizer.py       # Supporting visualizer code
  test-file/
    neural_net/         # Neural network from scratch
    earth-model/        # File management utilities
```

## Technologies

- **Python 3.12** with Flask (web server for Replit)
- **NumPy** — numerical computation
- **SciPy** — ODE integration
- **Matplotlib** (Agg backend) — plot generation as base64 PNG images
- **Gunicorn** — production WSGI server (Replit)

## Simulations

### Forest Growth (v1)
- Models three tree species: Oak (Chêne), Pine (Pin sylvestre), Beech (Hêtre)
- Uses explicit Euler integration for ODE solving
- Environmental factors: temperature (Gaussian response), CO₂ concentration
- Logistic competition between species for resources

### Procedural Planet (v2)
- Generates planets using Perlin noise / fBm (Fractional Brownian Motion)
- Biomes: ocean, plains, forest, mountain, snow, desert, ice
- Mercator projection 2D map rendering
- Original code uses Pygame for 3D interactive viewer; web version uses matplotlib

## Running

### Development
In the ./.code/v2 directory :

```
python3 tipe_v2.py
```
Runs on `0.0.0.0:5000` (Replit)