import requests
import json
import math

def parse_tle_to_json(output_file="oneweb_constellation.json"):
    """
    Télécharge les TLE OneWeb depuis CelesTrak et extrait
    inclinaison, RAAN et anomalie moyenne (phase sur l'orbite)
    """

    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=oneweb&FORMAT=tle"
    response = requests.get(url)
    lines = response.text.strip().splitlines()

    satellites = []

    # Les TLE sont groupés par blocs de 3 lignes : nom / ligne1 / ligne2
    for i in range(0, len(lines) - 2, 3):
        name  = lines[i].strip()
        line1 = lines[i+1].strip()
        line2 = lines[i+2].strip()

        # Extraction depuis la ligne 2 du TLE
        # Format TLE ligne 2 :
        # 2 NNNNN III.IIII RRR.RRRR EEEEEEE WWW.WWWW MMM.MMMM NN.NNNNNNNNNNNNNN
        inclination = float(line2[8:16])    # degrés
        raan        = float(line2[17:25])   # degrés — Right Ascension of Ascending Node
        mean_anomaly = float(line2[43:51])  # degrés — position sur l'orbite

        satellites.append({
            "name": name,
            "inclination_deg": round(inclination, 4),
            "raan_deg":        round(raan, 4),
            "phase_deg":       round(mean_anomaly, 4)   # anomalie moyenne = phase initiale
        })

    with open(output_file, "w") as f:
        json.dump(satellites, f, indent=2)

    print(f"{len(satellites)} satellites exportés dans {output_file}")

parse_tle_to_json()