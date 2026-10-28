"""Anwendung des Passmodells auf Spieler und Teams (Notebook 03).

Kennzahl: Pässe über Erwartung (PüE) = angekommene Pässe − Summe der xPass-Werte.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from kloppy._providers.sportec import get_IDSSE_url  # URL der DFL-Rohdatei, wie kloppy sie selbst nutzt
from kloppy.io import open_as_file                    # liest aus ~/kloppy_cache, falls schon geladen

# DFL-Positionscodes (PlayingPosition in den Metadaten) -> Rollengruppe.
# kloppy übersetzt nicht alle Codes (IVZ, DMZ, HL, HR, LM, RM fehlen) und führt die Schienenspieler
# einer Dreierkette (DLM, DRM) als Sechser. Deshalb lesen wir die Codes selbst.
ROLLE = {
    "TW": "Torwart",
    "IVL": "Innenverteidiger", "IVR": "Innenverteidiger", "IVZ": "Innenverteidiger",
    "LV": "Außenverteidiger", "RV": "Außenverteidiger", "DLM": "Außenverteidiger", "DRM": "Außenverteidiger",
    "DML": "Zentrales Mittelfeld", "DMR": "Zentrales Mittelfeld", "DMZ": "Zentrales Mittelfeld",
    "HL": "Zentrales Mittelfeld", "HR": "Zentrales Mittelfeld",
    "LM": "Flügel", "RM": "Flügel", "OLM": "Flügel", "ORM": "Flügel", "LA": "Flügel", "RA": "Flügel",
    "ZO": "Zehner/Sturm", "STL": "Zehner/Sturm", "STR": "Zehner/Sturm", "STZ": "Zehner/Sturm",
}
ROLLEN = ["Torwart", "Innenverteidiger", "Außenverteidiger", "Zentrales Mittelfeld", "Flügel", "Zehner/Sturm"]


def positionen(match_id):
    """Eine Zeile pro Spieler eines Spiels: DFL-Positionscode und Rollengruppe."""
    with open_as_file(get_IDSSE_url(match_id, "meta")) as datei:
        wurzel = ET.parse(datei).getroot()
    zeilen = []
    for spieler in wurzel.iter("Player"):
        code = spieler.get("PlayingPosition")
        zeilen.append(dict(match_id=match_id, player_id=spieler.get("PersonId"),
                           position_dfl=code, rolle=ROLLE.get(code, "unbekannt")))
    return pd.DataFrame(zeilen)


def mit_rollen(paesse, match_ids):
    """Hängt jedem Pass die Rolle des Passgebers in diesem Spiel an (Spalte rolle)."""
    pos = pd.concat([positionen(m) for m in match_ids])
    paesse = paesse.merge(pos, on=["match_id", "player_id"], how="left")
    paesse["rolle"] = paesse["rolle"].fillna("unbekannt")
    return paesse


def hauptrolle(paesse):
    """Rolle, in der ein Spieler die meisten seiner Pässe gespielt hat (ohne 'unbekannt')."""
    bekannt = paesse[paesse["rolle"] != "unbekannt"]
    return bekannt.groupby("player_id")["rolle"].agg(lambda s: s.value_counts().index[0])


def bilanz(paesse, nach):
    """Pässe über Erwartung (PüE) je Gruppe, z. B. je Spieler, Team oder Spiel.

    Spalten: paesse, angekommen, erwartet, pue, pue_100 (PüE pro 100 Pässe) und
    zufall_100: so stark schwankt pue_100 allein durch Zufall (1 Standardabweichung), wenn jeder Pass
    ein unabhängiger Münzwurf mit Wahrscheinlichkeit xPass wäre (Varianz je Pass = xPass · (1 − xPass)).
    """
    tabelle = (paesse.assign(varianz=paesse["xpass"] * (1 - paesse["xpass"]))
               .groupby(nach)
               .agg(paesse=("erfolg", "size"), angekommen=("erfolg", "sum"),
                    erwartet=("xpass", "sum"), varianz=("varianz", "sum")))
    tabelle["pue"] = tabelle["angekommen"] - tabelle["erwartet"]
    tabelle["pue_100"] = 100 * tabelle["pue"] / tabelle["paesse"]
    tabelle["zufall_100"] = 100 * np.sqrt(tabelle["varianz"]) / tabelle["paesse"]
    return tabelle.drop(columns="varianz")
