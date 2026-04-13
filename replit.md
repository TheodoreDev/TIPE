# TIPE — Simulations Scientifiques

## Overview

A Flask web application presenting scientific simulations developed as a TIPE (Travail d'Initiative Personnelle Encadré — French preparatory class project). The app wraps Python simulation scripts in a browser-friendly interactive interface.

## Project Structure

```
app.py                  # Flask web server (main entry point)
templates/index.html    # Web UI (single-page app)
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

- **Python 3.12** with Flask (web server)
- **NumPy** — numerical computation
- **SciPy** — ODE integration
- **Matplotlib** (Agg backend) — plot generation as base64 PNG images
- **Gunicorn** — production WSGI server

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
```
python3 app.py
```
Runs on `0.0.0.0:5000`

### Production
```
gunicorn --bind=0.0.0.0:5000 --reuse-port app:app
```

## API Endpoints

- `GET /` — Main web interface
- `POST /api/forest` — Run forest simulation, returns base64 PNG + summary stats
- `POST /api/planet` — Generate planet map, returns base64 PNG
