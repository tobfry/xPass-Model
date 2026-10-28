"""Empfänger-Modell (Notebook 04): Erweiterung des Zonenmodells um Merkmale aus den Tracking-Daten.

Idee: Das Modell beschreibt jeden Pass so, als wäre der Empfänger bekannt. Wie weit ist er weg, wie frei steht er,
wie viele Gegner stehen im Weg zu ihm? Dazu kommt der Druck auf den Passgeber. Der Empfänger wird mit einer festen
Regel geschätzt, für angekommene und Fehlpässe gleich, ohne den Ausgang zu kennen.

Grundlage ist die Momentaufnahme aus tracking.py: je Pass die Positionen und Laufgeschwindigkeiten aller Spieler
relativ zum Passgeber im Abspielmoment (dx nach vorn, dy nach links, in m; vx, vy in m/s).
"""

import numpy as np
import pandas as pd

from modell import zonen_merkmale

HORIZONT = 1.0           # Spieler werden so viele Sekunden mit ihrem Lauftempo vorausgerechnet
STRAFE_JE_METER = 0.5    # Punkte = Winkel zur Passrichtung (Grad) + 0,5 je Meter Entfernung
KORRIDOR_WINKEL = 15     # halber Öffnungswinkel des Korridors um die Linie zum Empfänger (Grad)
KORRIDOR_KAPPE = 3       # Gegner im Korridor bei 3 gedeckelt (ab 3 Gegnern sinkt die Erfolgsquote kaum weiter)
DRUCK_KAPPUNG = 15.0     # nächster Gegner zum Passgeber, darüber kaum noch Pässe
DISTANZ_KAPPUNG = 70.0   # Entfernung zum Empfänger
FREI_KAPPUNG = 20.0      # Abstand des nächsten Gegners zum Empfänger
EINDEUTIG_AB = 10        # Punkte-Abstand zum Zweitbesten, ab dem die Wahl des Empfängers als klar gilt


def passrichtung(paesse):
    """Passrichtung für das Modell: Ballweg der ersten 0,12 s nach dem Abschlag (Spalte richtung_ballweg).

    Bei Sync-Stufe 3 wurde der Abspielmoment nur grob gefunden, der Ballweg ist dann unzuverlässig; dort gilt der
    DFL-PlayAngle. (Diese Pässe bekommen im Modell ohnehin die Vorhersage des Zonenmodells.)
    """
    stufe3 = (paesse["sync_stufe"] == 3) & paesse["play_angle"].notna()
    return pd.Series(np.where(stufe3, paesse["play_angle"], paesse["richtung_ballweg"]), index=paesse.index)


def winkel_zur_linie(x, y, richtung_x, richtung_y):
    """Winkel (Grad, 0 bis 180) zwischen den Punkten (x, y) und einer Linie vom Passgeber in Richtung (richtung_x,
    richtung_y). Außerdem: wie weit die Punkte entlang dieser Linie liegen (m, negativ = hinter dem Passgeber)."""
    entlang = x * richtung_x + y * richtung_y
    seitlich = x * -richtung_y + y * richtung_x
    return np.degrees(np.abs(np.arctan2(seitlich, entlang))), entlang


def merkmale_eines_passes(s, richtung):
    """Tracking-Merkmale eines Passes.

    s: Momentaufnahme dieses Passes, eine Zeile je Spieler (Spalten mitspieler, passgeber, dx, dy, vx, vy).
    richtung: Passrichtung in Grad (im Uhrzeigersinn, 0 = nach vorn).

    Empfänger-Regel:
    1. Jeder Spieler wird HORIZONT s mit seinem Lauftempo vorausgerechnet.
    2. Kandidaten: Mitspieler, die danach in Passrichtung vor dem Passgeber stehen.
    3. Punkte = Winkel zur Passrichtung (Grad) + STRAFE_JE_METER × Entfernung (m); wenigste Punkte = Empfänger.
    """
    gegner = s[s["mitspieler"] == 0]
    ergebnis = {
        # Druck: Abstand des nächsten Gegners zum Passgeber im Abspielmoment
        "druck_abstand": np.hypot(gegner["dx"], gegner["dy"]).min(),
        "empf": None, "empf_distanz": np.nan, "empf_gegner_abstand": np.nan,
        "korridor_gegner": 0, "empf_winkel": np.nan, "eindeutigkeit": np.nan,
    }

    # 1. Positionen in 1 s
    bx = s["dx"] + HORIZONT * s["vx"]
    by = s["dy"] + HORIZONT * s["vy"]

    # 2. + 3. Empfänger: Mitspieler vor dem Ball mit den wenigsten Punkten
    w = np.radians(richtung)
    winkel, entlang = winkel_zur_linie(bx, by, np.cos(w), -np.sin(w))
    punkte = winkel + STRAFE_JE_METER * np.hypot(bx, by)
    kandidat = (s["mitspieler"] == 1) & (s["passgeber"] == 0) & (entlang > 0) & bx.notna() & by.notna()
    if not kandidat.any():
        return ergebnis                                   # kein Mitspieler vor dem Ball (0,4 % der Pässe)
    rangfolge = punkte[kandidat].sort_values(kind="stable")
    e = rangfolge.index[0]
    ex, ey = bx[e], by[e]                                 # Empfänger in 1 s
    ergebnis["empf"] = s.loc[e, "player_id"]
    ergebnis["empf_distanz"] = np.hypot(ex, ey)
    ergebnis["empf_winkel"] = winkel[e]
    ergebnis["eindeutigkeit"] = min(rangfolge.iloc[1] - rangfolge.iloc[0], 99.0) if len(rangfolge) > 1 else 99.0

    # Freiraum: Abstand des nächsten Gegners zum Empfänger (beide vorausgerechnet)
    ergebnis["empf_gegner_abstand"] = np.hypot(bx[gegner.index] - ex, by[gegner.index] - ey).min()

    # Korridor: Gegner (Position im Abspielmoment) zwischen Passgeber und Empfänger, höchstens 15° neben der Linie
    laenge = np.hypot(ex, ey)
    g_winkel, g_entlang = winkel_zur_linie(gegner["dx"], gegner["dy"], ex / laenge, ey / laenge)
    ergebnis["korridor_gegner"] = int(((g_entlang > 0) & (g_entlang < laenge) & (g_winkel < KORRIDOR_WINKEL)).sum())
    return ergebnis


def tracking_merkmale(paesse, spieler):
    """Tracking-Merkmale für alle Pässe (eine Zeile je Pass, Index wie paesse).

    paesse: Spalten event_id und richtung; spieler: Momentaufnahme aus tracking.momentaufnahmen_spiel.
    Spalten: druck_abstand, empf (geschätzter Empfänger), empf_distanz, empf_gegner_abstand, korridor_gegner,
    empf_winkel (Winkel Passrichtung – Empfänger; nicht im Modell, nur für den Test in Notebook 04),
    eindeutigkeit (Punkte-Abstand des Zweitbesten, 99 = nur ein Kandidat).
    """
    richtung_je_pass = dict(zip(paesse["event_id"], paesse["richtung"]))
    je_pass = {event_id: merkmale_eines_passes(s, richtung_je_pass[event_id])
               for event_id, s in spieler.groupby("event_id")}
    return pd.DataFrame([je_pass[e] for e in paesse["event_id"]], index=paesse.index)


def modell_merkmale(daten, mit_winkel=False):
    """Die 21 Merkmale des Empfänger-Modells für alle Pässe.

    - die 17 Zonen-Merkmale (modell.zonen_merkmale), mit der Passrichtung aus dem Ballweg (Spalte richtung)
    - druck_abstand: nächster Gegner zum Passgeber, gekappt bei 15 m
    - empf_distanz: Entfernung zum Empfänger, gekappt bei 70 m
    - empf_gegner_abstand: wie frei der Empfänger steht, gekappt bei 20 m
    - korridor_gegner: Gegner im Korridor zum Empfänger, bei 3 gedeckelt
    Ohne Mitspieler vor dem Ball (0,4 % der Pässe): Entfernung 60 m, Abstand 20 m, Korridor 0.
    mit_winkel: zusätzlich empf_winkel (fehlend = 90°). Nur für den Test in Notebook 04, nicht im Modell.
    """
    X = zonen_merkmale(daten.assign(play_angle=daten["richtung"])).copy()
    X["druck_abstand"] = daten["druck_abstand"].clip(upper=DRUCK_KAPPUNG).to_numpy()
    X["empf_distanz"] = daten["empf_distanz"].fillna(60.0).clip(upper=DISTANZ_KAPPUNG).to_numpy()
    X["empf_gegner_abstand"] = daten["empf_gegner_abstand"].fillna(FREI_KAPPUNG).clip(upper=FREI_KAPPUNG).to_numpy()
    X["korridor_gegner"] = daten["korridor_gegner"].clip(upper=KORRIDOR_KAPPE).to_numpy()
    if mit_winkel:
        X["empf_winkel"] = daten["empf_winkel"].fillna(90.0).to_numpy()
    return X


# ---------------------------------------------------------------------------------------------------
# Bewertung
# ---------------------------------------------------------------------------------------------------

def leave_one_match_out_mit_rueckfall(X, y, spiele, auswahl, rueckfall, modell_fabrik):
    """Leave-one-match-out nur auf den Pässen der auswahl (Training und Vorhersage); alle anderen Pässe bekommen
    die Vorhersage rueckfall (hier: Zonenmodell für Pässe mit ungenau gefundenem Abspielmoment)."""
    y, spiele, auswahl = np.asarray(y), np.asarray(spiele), np.asarray(auswahl, bool)
    vorhersage = np.asarray(rueckfall, float).copy()
    for m in np.unique(spiele):
        train, test = (spiele != m) & auswahl, (spiele == m) & auswahl
        modell = modell_fabrik().fit(X[train], y[train])
        vorhersage[test] = modell.predict_proba(X[test])[:, 1]
    return vorhersage


def logloss_differenz(y, p_neu, p_alt, spiele, n_boot=2000, seed=0):
    """Log-Loss(neu) − Log-Loss(alt) mit 95-%-Intervall (negativ = neu ist besser).

    Intervall aus dem Spiele-Bootstrap: n_boot-mal so viele Spiele mit Zurücklegen ziehen, wie es gibt, und die
    Differenz auf den Pässen der gezogenen Spiele berechnen. So geht die Unsicherheit durch die wenigen Spiele ein.
    """
    y, spiele = np.asarray(y), np.asarray(spiele)

    def logloss_je_pass(p):
        p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p))

    d = logloss_je_pass(p_neu) - logloss_je_pass(p_alt)
    je_spiel = {m: d[spiele == m] for m in np.unique(spiele)}
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        gezogen = rng.choice(list(je_spiel), size=len(je_spiel), replace=True)
        boot.append(np.concatenate([je_spiel[m] for m in gezogen]).mean())
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)
