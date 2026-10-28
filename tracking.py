"""Momentaufnahme im Abspielmoment aus den Tracking-Daten (Notebook 04, Empfänger-Modell).

Ablauf je Pass:
1. abspiel_frame: den Tracking-Frame finden, in dem der Ball den Passgeber verlässt. Event-Zeit und Tracking
   liegen bis zu ca. 2–3 s auseinander, deshalb wird im Zeitfenster um das Event gesucht.
2. abschlag_richtung: Richtung des Balls in den ersten 0,12 s nach dem eigentlichen Abschlag.
3. momentaufnahme: Positionen und Laufgeschwindigkeiten aller Spieler im Abspiel-Frame.

Alle Positionen werden so gedreht, dass das passende Team nach rechts angreift (wie in passdaten.py).
Richtungen wie beim DFL-Attribut PlayAngle: Grad im Uhrzeigersinn, 0 = nach vorn (Richtung gegnerisches Tor).
"""

import numpy as np
import pandas as pd
from kloppy import sportec

FPS = 25                # Bilder pro Sekunde
ABSCHLAG_SUCHE = 6      # der Abschlag wird in den ersten 6 Frames nach dem Abspiel-Frame gesucht
RICHTUNG_FRAMES = 3     # Passrichtung aus dem Ballweg der ersten 3 Frames (0,12 s) nach dem Abschlag


def lade_tracking(match_id):
    return sportec.load_open_tracking_data(match_id=match_id, coordinates="secondspectrum")


def tracking_arrays(tracking):
    """Tracking je Halbzeit als numpy-Arrays: Zeit, Ball (x, y, z), Spieler (Frames x Spieler x 2).

    Gibt außerdem die Liste der Spieler-IDs (Spaltenreihenfolge) und ein Dict Spieler -> Team zurück.
    """
    df = tracking.to_df()
    spieler, team_von = [], {}
    for team in tracking.metadata.teams:
        for p in team.players:
            if f"{p.player_id}_x" in df.columns:
                spieler.append(p.player_id)
                team_von[p.player_id] = team.team_id
    halbzeiten = {}
    for period, g in df.groupby("period_id"):
        xy = np.stack(
            [g[[f"{pid}_x" for pid in spieler]].to_numpy(float),
             g[[f"{pid}_y" for pid in spieler]].to_numpy(float)],
            axis=-1,
        )
        halbzeiten[period] = dict(
            t=g["timestamp"].dt.total_seconds().to_numpy(),
            ball=g[["ball_x", "ball_y", "ball_z"]].to_numpy(float),
            xy=xy,
        )
    return halbzeiten, spieler, team_von


def _abschlag(hz, k, t_event, fenster, tempo_min, n=5):
    """Frames, in denen der Ball am Fuß des Passgebers ist (< 1,5 m) und schnell wegfliegt."""
    idx = np.where(np.abs(hz["t"] - t_event) <= fenster)[0]
    idx = idx[idx + n < len(hz["t"])]
    if len(idx) == 0:
        return None
    ball = hz["ball"][:, :2]
    abstand = np.linalg.norm(ball[idx] - hz["xy"][idx, k], axis=1)
    tempo = np.linalg.norm(ball[idx + n] - ball[idx], axis=1) / (n / FPS)   # Balltempo der nächsten 0,2 s
    abstand_spaeter = np.linalg.norm(ball[idx + n] - hz["xy"][idx, k], axis=1)
    kandidaten = idx[(abstand < 1.5) & (tempo > tempo_min) & (abstand_spaeter > abstand + 0.8)]
    if len(kandidaten) == 0:
        return None
    # Zusammenhängende Blöcke = ein Abschlag. Den Block nehmen, der zeitlich am nächsten am Event liegt,
    # und davon den ersten Frame.
    bloecke = np.split(kandidaten, np.where(np.diff(kandidaten) > 1)[0] + 1)
    block = min(bloecke, key=lambda b: np.min(np.abs(hz["t"][b] - t_event)))
    return int(block[0])


def abspiel_frame(hz, spieler, passgeber, t_event, fenster=3.0):
    """Index des Abspiel-Frames, gesucht im Zeitfenster ±fenster s um das Event (Kaskade):
    1. Ball am Fuß (< 1,5 m) und danach schneller als 8 m/s
    2. falls nicht gefunden: schneller als 5 m/s (z. B. kurze, langsame Pässe)
    3. falls nicht gefunden: Frame mit dem kleinsten Abstand Ball–Passgeber (Abspielmoment nur grob bekannt)
    Rückgabe: (Index, Abstand Ball–Passgeber in m, Zeitversatz Frame − Event in s, Stufe 1–3) oder None.
    """
    if passgeber not in spieler:
        return None
    k = spieler.index(passgeber)
    i, stufe = None, None
    for stufe_nr, tempo_min in [(1, 8.0), (2, 5.0)]:
        i = _abschlag(hz, k, t_event, fenster, tempo_min)
        if i is not None:
            stufe = stufe_nr
            break
    if i is None:
        idx = np.where(np.abs(hz["t"] - t_event) <= fenster)[0]
        if len(idx) == 0:
            return None
        d = np.linalg.norm(hz["ball"][idx, :2] - hz["xy"][idx, k], axis=1)
        if np.all(np.isnan(d)):
            return None
        i, stufe = int(idx[np.nanargmin(d)]), 3
    abstand = float(np.linalg.norm(hz["ball"][i, :2] - hz["xy"][i, k]))
    return i, abstand, float(hz["t"][i] - t_event), stufe


def abschlag_richtung(hz, i, vz):
    """Abschlag und Passrichtung ab dem Abspiel-Frame i.

    Der Abspiel-Frame ist der erste Frame, ab dem der Ball in den nächsten 0,2 s schnell wird; der Fuß trifft
    den Ball meist erst 1–3 Frames später. Abschlag = der Frame k (0 bis ABSCHLAG_SUCHE nach i) mit dem größten
    Tempo-Sprung des Balls, also Tempo(k -> k+1) − Tempo(k−1 -> k). Richtung = Ballweg der folgenden
    RICHTUNG_FRAMES Frames (0,12 s): kurz genug, dass früh geblockte Bälle kaum eingehen.
    Rückgabe: (Abschlag in Frames nach i, Richtung in Grad).
    """
    frames = np.arange(i - 1, i + ABSCHLAG_SUCHE + RICHTUNG_FRAMES + 1)
    ball = np.full((len(frames), 2), np.nan)
    ok = (frames >= 0) & (frames < len(hz["t"]))
    ball[ok] = hz["ball"][frames[ok], :2] * vz
    tempo = np.hypot(*np.diff(ball, axis=0).T) * FPS                 # tempo[j]: von frames[j] nach frames[j+1]
    sprung = tempo[1:ABSCHLAG_SUCHE + 2] - tempo[:ABSCHLAG_SUCHE + 1]
    k0 = int(np.argmax(np.nan_to_num(sprung, nan=-np.inf)))
    v = ball[k0 + 1 + RICHTUNG_FRAMES] - ball[k0 + 1]                  # frames[k0 + 1] = Frame i + k0
    return k0, float((-np.degrees(np.arctan2(v[1], v[0]))) % 360)


def momentaufnahme(hz, spieler, team_von, i, passgeber, team_id, angriff_rechts):
    """Abschlag, Passrichtung und alle Spieler im Abspiel-Frame i.

    Rückgabe: (pass_zeile, spieler_zeilen)
    - pass_zeile (dict): Abspiel-Frame, Abschlag (Frames danach), Passrichtung aus dem Ballweg (Grad)
    - spieler_zeilen (Liste von dicts): je Spieler auf dem Platz die Position relativ zum Passgeber (dx nach vorn,
      dy nach links, in m) und die Laufgeschwindigkeit (vx, vy in m/s) aus den 0,2 s VOR dem Abspiel.
    """
    vz = 1 if angriff_rechts else -1
    xy = hz["xy"][i] * vz
    tempo = (xy - hz["xy"][max(i - 5, 0)] * vz) / (5 / FPS)
    k = spieler.index(passgeber)
    p = xy[k]
    abschlag, richtung = abschlag_richtung(hz, i, vz)
    zeile = dict(frame=int(i), abschlag_frame=abschlag, richtung_ballweg=richtung)
    spieler_zeilen = []
    for s, pid in enumerate(spieler):
        if np.isnan(xy[s, 0]):
            continue                              # nicht auf dem Platz
        spieler_zeilen.append(dict(
            player_id=pid,
            mitspieler=int(team_von[pid] == team_id),
            passgeber=int(pid == passgeber),
            dx=float(xy[s, 0] - p[0]), dy=float(xy[s, 1] - p[1]),
            vx=float(tempo[s, 0]), vy=float(tempo[s, 1]),
        ))
    return zeile, spieler_zeilen


def momentaufnahmen_spiel(match_id, paesse):
    """Momentaufnahmen aller Pässe eines Spiels (liest das Roh-Tracking einmal ein, ca. 1 Minute, ca. 4 GB Speicher).

    paesse: Pass-Tabelle dieses Spiels aus passdaten.pass_tabelle.
    Rückgabe: zwei DataFrames (pässe, spieler), verbunden über event_id. Pässe ohne gefundenen Abspiel-Frame fehlen.
    """
    halbzeiten, spieler, team_von = tracking_arrays(lade_tracking(match_id))
    p_zeilen, s_zeilen = [], []
    for p in paesse.itertuples():
        hz = halbzeiten.get(p.period)
        if hz is None or p.player_id is None:
            continue
        treffer = abspiel_frame(hz, spieler, p.player_id, p.t)
        if treffer is None:
            continue
        i, abstand, versatz, stufe = treffer
        zeile, sz = momentaufnahme(hz, spieler, team_von, i, p.player_id, p.team_id, p.angriff_rechts)
        zeile.update(event_id=p.event_id, sync_abstand=abstand, sync_versatz=versatz, sync_stufe=stufe)
        p_zeilen.append(zeile)
        s_zeilen += [dict(event_id=p.event_id, **z) for z in sz]
    return pd.DataFrame(p_zeilen), pd.DataFrame(s_zeilen)
