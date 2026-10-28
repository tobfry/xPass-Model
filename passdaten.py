"""Pass-Tabelle aus den DFL-Eventdaten (geladen mit kloppy).

- Eine Zeile pro Pass (DFL-Event "Play", also Pässe und Flanken inkl. Standards), in der Reihenfolge, in der kloppy
  die Events liefert.
- Koordinaten in Metern, Ursprung im Mittelpunkt, und so gedreht, dass das passende Team immer nach rechts
  (Richtung x = +52,5) angreift.
- Erfolg = DFL-Bewertung des Passes (successfullyCompleted / successful).
- Dazu einige Rohangaben der DFL, die kloppy nicht selbst übernimmt (Höhe, Abspielwinkel, Empfänger, ...).
  Begründung, welche davon ins Modell dürfen: notebooks/01_daten_und_definitionen.ipynb.
"""

import numpy as np
import pandas as pd
from kloppy import sportec
from kloppy.domain import SetPieceQualifier

MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]
ERSTE_LIGA = ["J03WMX", "J03WN1"]


def lade_events(match_id):
    """Event-Daten eines Spiels, Koordinaten wie im Tracking (Meter, Ursprung Spielfeldmitte)."""
    return sportec.load_open_event_data(match_id=match_id, coordinates="secondspectrum")


def greift_nach_rechts_an(dataset, team, period_id):
    """True, wenn das Team in dieser Halbzeit Richtung x = +52,5 spielt.

    Laut Metadaten spielt das Heimteam in der 1. Halbzeit nach rechts (HOME_AWAY) oder nach links; zur 2. Halbzeit
    werden die Seiten gewechselt.
    """
    heim_rechts_in_hz1 = dataset.metadata.orientation.name == "HOME_AWAY"
    ist_heim = team.ground.name == "HOME"
    return (ist_heim == heim_rechts_in_hz1) == (period_id == 1)


def _zahl(wert):
    try:
        return float(wert)
    except (TypeError, ValueError):
        return np.nan


def pass_tabelle(match_id, dataset=None):
    """Eine Zeile pro Pass eines Spiels."""
    if dataset is None:
        dataset = lade_events(match_id)
    team_von = {p.player_id: t.team_id for t in dataset.metadata.teams for p in t.players}

    zeilen = []
    for e in dataset.records:
        if e.event_type.name != "PASS":
            continue
        roh = e.raw_event                                   # alle Original-Angaben der DFL zu diesem Event
        vz = 1 if greift_nach_rechts_an(dataset, e.team, e.period.id) else -1   # Vorzeichen zum Drehen
        ende = e.receiver_coordinates                       # kloppys Endpunkt, nur bei angekommenen Pässen
        standard = e.get_qualifier_value(SetPieceQualifier)

        zeilen.append(dict(
            match_id=match_id,
            liga=1 if match_id in ERSTE_LIGA else 2,
            event_id=e.event_id,
            period=e.period.id,
            t=e.timestamp.total_seconds(),
            team_id=e.team.team_id,
            team=e.team.name,
            player_id=e.player.player_id if e.player else None,
            player=e.player.name if e.player else None,
            angriff_rechts=vz == 1,
            x=vz * e.coordinates.x, y=vz * e.coordinates.y,
            end_x=vz * ende.x if ende else np.nan, end_y=vz * ende.y if ende else np.nan,
            standard=standard.name if standard else "OPEN_PLAY",
            # Flanken ("Cross") haben kein FreeKickLayup-Attribut, normale Pässe ("Pass") immer
            flanke="FreeKickLayup" not in roh,
            erfolg=int(roh.get("Evaluation") in ("successfullyCompleted", "successful")),
            bewertung=roh.get("Evaluation"),
            kloppy_ergebnis=e.result.name if e.result else None,
            recipient=roh.get("Recipient"),
            recipient_team=team_von.get(roh.get("Recipient")),
            play_angle=_zahl(roh.get("PlayAngle")),
            hoehe=roh.get("Height"),
            laenge_dfl=roh.get("Distance"),
            direction_dfl=roh.get("Direction"),
        ))
    return pd.DataFrame(zeilen)


def lade_pass_tabelle(pfad):
    """Die in Notebook 01 gespeicherte Pass-Tabelle (data/paesse.csv) einlesen; IDs bleiben Text."""
    return pd.read_csv(pfad, dtype={"event_id": str, "team_id": str, "player_id": str,
                                    "recipient": str, "recipient_team": str})
