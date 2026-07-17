"""
Task 1 - Récupération des données depuis la PokéAPI (aucune clé requise).

Stratégie :
  1. On appelle l'endpoint liste (/pokemon) en gérant la PAGINATION via le champ
     "next" renvoyé par l'API (on boucle jusqu'à avoir assez de Pokémon).
  2. La liste ne donne que le nom + une URL de détail : on fait donc un 2e appel
     par Pokémon pour récupérer ses stats de base (nested JSON à aplatir).
  3. On aplatit le JSON imbriqué (stats[i].base_stat -> colonnes hp, attack, ...).
  4. On construit un DataFrame et on l'enregistre dans data/raw_data.csv.

Le fichier est 100% reproductible : il suffit de relancer ce script.
"""

import time
from pathlib import Path

import pandas as pd
import requests

# Dossier du projet (le script marche quel que soit le répertoire courant)
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATA_DIR.mkdir(exist_ok=True)

BASE_URL = "https://pokeapi.co/api/v2/pokemon"
N_POKEMON = 1000          # nombre de Pokémon à récupérer (>> 200 demandés)
PAGE_SIZE = 100           # taille d'une page pour la liste
SLEEP = 0.15              # pause entre appels pour ne pas surcharger l'API gratuite

# Les 6 stats de base qui nous serviront de features numériques
STAT_KEYS = ["hp", "attack", "defense", "special-attack", "special-defense", "speed"]


def get_json(session, url, params=None, retries=3):
    """Appel GET avec quelques tentatives en cas d'erreur réseau ponctuelle."""
    for attempt in range(retries):
        try:
            resp = session.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            print(f"  ! erreur ({exc}) tentative {attempt + 1}/{retries}")
            time.sleep(1.0)
    raise RuntimeError(f"Échec définitif sur {url}")


def list_pokemon_urls(session, n):
    """Parcourt les pages de la liste (pagination via 'next') et renvoie n URLs de détail."""
    urls = []
    url, params = BASE_URL, {"limit": PAGE_SIZE, "offset": 0}
    while url and len(urls) < n:
        data = get_json(session, url, params=params)
        urls.extend(item["url"] for item in data["results"])
        url = data.get("next")      # PokéAPI donne l'URL de la page suivante, ou None
        params = None               # 'next' contient déjà les paramètres
    return urls[:n]


def extract_record(detail):
    """Aplatit le JSON d'un Pokémon en un dictionnaire plat (une ligne du DataFrame)."""
    # stats -> dictionnaire {nom_stat: valeur}
    stats = {s["stat"]["name"]: s["base_stat"] for s in detail["stats"]}

    # types : le 1er est le type principal (notre cible), le 2e est optionnel
    types = [t["type"]["name"] for t in detail["types"]]

    record = {
        "id": detail["id"],
        "name": detail["name"],
        "height": detail["height"],
        "weight": detail["weight"],
        "base_experience": detail.get("base_experience"),
        "type_1": types[0] if len(types) >= 1 else None,   # <-- cible de classification
        "type_2": types[1] if len(types) >= 2 else None,
    }
    # une colonne par stat (les '-' remplacés par '_' pour des noms de colonnes propres)
    for key in STAT_KEYS:
        record[key.replace("-", "_")] = stats.get(key)
    return record


def fetch_all_records():
    records = []
    with requests.Session() as session:
        print(f"Récupération de la liste des {N_POKEMON} premiers Pokémon...")
        urls = list_pokemon_urls(session, N_POKEMON)
        print(f"{len(urls)} URLs de détail récupérées. Appels de détail en cours...")

        for i, url in enumerate(urls, start=1):
            detail = get_json(session, url)
            records.append(extract_record(detail))
            if i % 50 == 0:
                print(f"  {i}/{len(urls)} Pokémon traités")
            time.sleep(SLEEP)
    return records


if __name__ == "__main__":
    records = fetch_all_records()
    df = pd.DataFrame(records)
    out_path = DATA_DIR / "raw_data.csv"
    df.to_csv(out_path, index=False)
    print(f"\nOK -> {len(df)} lignes enregistrées dans {out_path}")
