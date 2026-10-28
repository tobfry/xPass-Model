"""Merkmale und Bewertung des Passmodells (Notebook 02): erstes Modell A (Ort linear) und Zonenmodell.

Koordinaten (aus passdaten.py): Meter, Ursprung im Mittelpunkt, das passende Team greift nach rechts an.
x = Längsachse (0 = Mittellinie, +52,5 = gegnerische Torlinie), y = Querachse (0 = Mitte, positiv = links).
Richtung θ = DFL-PlayAngle in Grad, im Uhrzeigersinn, 0° = Richtung gegnerisches Tor.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

DRITTEL_GRENZE = 17.5        # 105 m in drei Drittel zu je 35 m
STRAFRAUM_X = 52.5 - 16.5    # Strafraum beginnt 16,5 m vor der Torlinie
STRAFRAUM_Y = 20.16          # halbe Strafraumbreite (40,32 m)
PUNKT_IN_PASSRICHTUNG = 15   # gedachter Punkt 15 m in Passrichtung (Median Passlänge: 16,7 m)

AUSGESCHLOSSENE_STANDARDS = ["KICK_OFF", "THROW_IN", "CORNER_KICK"]

BEZUGSZONE = "mittleres_zentrum"
ZONEN = ["erstes_zentrum", "erstes_fluegel", "mittleres_fluegel",
         "letztes_zentrum", "letztes_fluegel", "strafraum"]

ZONEN_MERKMALE = (
    ["flanke", "freistoss", "abstoss", "hoch", "tiefe", "breite"]
    + ["zone_" + z for z in ZONEN]
    + ["zur_mitte", "letztes_drittel_x_zurueck", "letztes_drittel_x_zur_mitte",
       "erstes_drittel_x_tiefe", "flach_richtung_strafraum"]
)


def pass_auswahl(paesse):
    """Unsere Pass-Definition: ohne Anstöße, Einwürfe und Ecken; nur Pässe mit Richtungsangabe."""
    auswahl = paesse[~paesse["standard"].isin(AUSGESCHLOSSENE_STANDARDS)]
    return auswahl.dropna(subset=["play_angle"]).reset_index(drop=True)


def zone(x, y):
    """Zone des Abspielorts: 3 Drittel × (Zentrum | Flügel) + gegnerischer Strafraum."""
    x, y = np.asarray(x), np.asarray(y)
    drittel = np.where(x < -DRITTEL_GRENZE, "erstes", np.where(x > DRITTEL_GRENZE, "letztes", "mittleres"))
    bahn = np.where(np.abs(y) > STRAFRAUM_Y, "fluegel", "zentrum")
    name = np.char.add(np.char.add(drittel, "_"), bahn)
    strafraum = (x > STRAFRAUM_X) & (np.abs(y) <= STRAFRAUM_Y)
    return np.where(strafraum, "strafraum", name)


def zonen_merkmale(paesse):
    """Die 17 Merkmale des Zonenmodells (nur Event-Daten, Passrichtung = DFL-PlayAngle)."""
    w = np.radians(paesse["play_angle"].to_numpy())
    tiefe = np.cos(w)                 # +1 in die Tiefe, 0 quer, -1 zurück
    quer = -np.sin(w)                 # seitlicher Anteil, positiv = nach links
    x, y = paesse["x"].to_numpy(), paesse["y"].to_numpy()
    hoch = (paesse["hoehe"] == "high").astype(int).to_numpy()

    # Gedachter Punkt 15 m in Passrichtung (nur Geometrie, nicht das echte Ziel)
    px, py = x + PUNKT_IN_PASSRICHTUNG * tiefe, y + PUNKT_IN_PASSRICHTUNG * quer
    zur_mitte = (np.abs(y) - np.abs(py)) / PUNKT_IN_PASSRICHTUNG
    richtung_strafraum = (px > STRAFRAUM_X) & (px <= 52.5) & (np.abs(py) <= STRAFRAUM_Y)

    letztes_drittel = (x > DRITTEL_GRENZE).astype(int)
    erstes_drittel = (x < -DRITTEL_GRENZE).astype(int)
    z = zone(x, y)

    X = pd.DataFrame({
        "flanke": paesse["flanke"].astype(int).to_numpy(),
        "freistoss": (paesse["standard"] == "FREE_KICK").astype(int).to_numpy(),
        "abstoss": (paesse["standard"] == "GOAL_KICK").astype(int).to_numpy(),
        "hoch": hoch,
        "tiefe": tiefe,
        "breite": np.abs(quer),
    })
    for name in ZONEN:
        X["zone_" + name] = (z == name).astype(int)
    X["zur_mitte"] = zur_mitte
    X["letztes_drittel_x_zurueck"] = letztes_drittel * np.clip(-tiefe, 0, None)
    X["letztes_drittel_x_zur_mitte"] = letztes_drittel * zur_mitte
    X["erstes_drittel_x_tiefe"] = erstes_drittel * tiefe
    X["flach_richtung_strafraum"] = (richtung_strafraum & (hoch == 0)).astype(int)
    X.index = paesse.index
    return X[ZONEN_MERKMALE]


def modell_a_merkmale(paesse):
    """Die 8 Merkmale des ersten Modells A """
    w = np.radians(paesse["play_angle"].to_numpy())
    return pd.DataFrame({
        "x_10m": paesse["x"].to_numpy() / 10,
        "abstand_mitte_10m": np.abs(paesse["y"].to_numpy()) / 10,
        "flanke": paesse["flanke"].astype(int).to_numpy(),
        "freistoss": (paesse["standard"] == "FREE_KICK").astype(int).to_numpy(),
        "abstoss": (paesse["standard"] == "GOAL_KICK").astype(int).to_numpy(),
        "hoch": (paesse["hoehe"] == "high").astype(int).to_numpy(),
        "tiefe": np.cos(w),
        "breite": np.abs(np.sin(w)),
    }, index=paesse.index)


def neues_modell():
    """Logistische Regression wie im Projekt verwendet (scikit-learn, leichte L2-Regularisierung)."""
    return LogisticRegression(max_iter=5000)#logistische regression = unser Modell; leave_one_match_out nutzt es, wenn kein anderes übergeben wird

def leave_one_match_out(X, y, spiele, modell_fabrik=neues_modell):
    """Vorhersagen, bei denen jedes Spiel von einem Modell geschätzt wird, das dieses Spiel nicht kennt."""
    y = np.asarray(y) # Ergebnis je Pass (1 = angekommen, 0 = nicht) als einfaches Array
    vorhersage = np.zeros(len(y)) #leere liste mit 0 pro pass, wird in der schleife befüllt
    spiele = np.asarray(spiele)   # Spiel je Pass
    # 7 Runden (eine pro Spiel): train = Pässe aus 6 Spielen, test = Pässe aus dem ausgelassenen Spiel
    for train, test in GroupKFold(n_splits=len(np.unique(spiele))).split(X, y, spiele):
        # neues Modell nur mit den Pässen der 6 Trainingsspiele trainieren
        modell = modell_fabrik().fit(X.iloc[train], y[train])
        # für das ausgelassene Spiel die Wahrscheinlichkeit "angekommen" (Spalte 1) vorhersagen
        vorhersage[test] = modell.predict_proba(X.iloc[test])[:, 1]
    return vorhersage             # jeder Pass hat jetzt einen xPass von einem Modell, das sein Spiel nicht kannte
