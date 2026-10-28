# Passmodell: Wie wahrscheinlich kommt ein Pass an?

Case Study für das Werkstudium Data Science. Ziel ist ein Modell, das für jeden Pass die Wahrscheinlichkeit schätzt, dass er beim Mitspieler ankommt (xPass). Grundlage sind die offenen Tracking- und Eventdaten der DFL/Sportec (7 Spiele der Saison 2022/23, 2 aus der 1. und 5 aus der 2. Bundesliga). Das Hauptmodell (Zonenmodell) nutzt nur die Event-Daten; Notebook 04 erweitert es um Merkmale aus den Tracking-Daten (Empfänger-Modell).

## Setup

Voraussetzung: Python 3.10 oder neuer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Die Daten ([Bassek et al. 2025, Scientific Data](https://doi.org/10.1038/s41597-025-04505-y)) werden beim ersten Ausführen über [kloppy](https://kloppy.pysport.org/user-guide/loading-data/sportec/) heruntergeladen und in `~/kloppy_cache` zwischengespeichert. Sie liegen nicht im Repository.

Hinweis für macOS mit Python von python.org: Schlägt der Download mit „Input file not found“ fehl, fehlen meist die SSL-Zertifikate. Abhilfe: `Install Certificates.command` im Ordner `/Applications/Python 3.x/` ausführen.

## Ausführen

Im Projektordner `jupyter lab` starten und die Notebooks **in der Reihenfolge 01 bis 04** ausführen. Notebook 01 speichert die Pass-Tabelle in `data/`, die folgenden Notebooks lesen sie von dort. Laufzeit: Notebook 01 ca. 1–2 Minuten (Laden der 7 Spiele), 02 und 03 jeweils unter einer Minute. Ausnahme: Notebook 04 liest beim ersten Lauf das Roh-Tracking aller 7 Spiele ein (ca. 7 Minuten, ca. 4 GB Arbeitsspeicher) und speichert die Momentaufnahme je Pass in `data/momentaufnahme_*.csv`; danach liest es nur noch diese Dateien.

## Struktur

| Pfad | Inhalt |
|---|---|
| `notebooks/01_daten_und_definitionen.ipynb` | Überblick über Event-, Kader- und Tracking-Daten; Definition von Pass und Erfolg, Leakage-Prüfung der DFL-Angaben → `data/paesse.csv` |
| `notebooks/02_passmodell.ipynb` | **Aufgabe 1:** schrittweiser Aufbau (Modell A, Ort linear) → Zonenmodell (Hauptmodell, 17 Merkmale); Güte, Kalibrierung, Gewichte, Beispielszenen → `data/vorhersagen_zonen.csv` |
| `notebooks/03_anwendung.ipynb` | **Aufgabe 2:** Pässe über Erwartung je Spieler: Passquote vs. erwartete Passquote, Rangliste innerhalb der Rolle, Belastbarkeit |
| `notebooks/04_erweiterung_tracking.ipynb` | **Erweiterung:** Empfänger-Modell (Zonenmodell + 4 Tracking-Merkmale), Vergleich mit dem Zonenmodell → `data/vorhersagen_empfaenger.csv` |
| `passdaten.py` | Pass-Tabelle aus den Event-Daten: Koordinaten in Angriffsrichtung, Erfolg laut DFL, DFL-Rohangaben (Höhe, Winkel, Empfänger …) |
| `modell.py` | Pass-Auswahl, Zonen, Merkmale von Modell A und Zonenmodell, Leave-one-match-out |
| `auswertung.py` | Positionen aus den DFL-Metadaten, Kennzahl „Pässe über Erwartung“ |
| `tracking.py` | Abspielmoment im Tracking finden, Passrichtung aus dem Ballweg, Momentaufnahme aller Spieler je Pass |
| `empfaenger.py` | Empfänger schätzen, die 4 Tracking-Merkmale, Spiele-Bootstrap |
| `data/` | Zwischenergebnisse der Notebooks (werden beim Ausführen erzeugt, nicht im Repository) |

## Vorgehen und Ergebnisse

- **Daten (01):** 7 Spiele, 6.062 Zuspiele. kloppy liefert die Events als Tabelle; Höhe, Abspielwinkel, Bewertung und Empfänger kommen aus den DFL-Rohangaben (`raw_event`).
- **Pass (01):** alle Zuspiele mit dem Fuß, also auch Freistöße, Abstöße und Flanken; ohne Einwürfe, Ecken und Anstöße (5.664 Pässe, 78,6 % erfolgreich).
- **Erfolg (01):** offizielle DFL-Bewertung des Passes; sie stimmt mit kloppys Ergebnis bis auf 4 Pässe überein.
- **Leakage (01):** Endpunkt, Empfänger, DFL-Passlänge und DFL-Passart beschreiben das Ende des Passes bzw. werden in Kenntnis des Ergebnisses erfasst und bleiben draußen (die Passart ist bei Fehlpässen viermal so oft ausgefüllt). Verwendet werden nur Angaben, die im Moment des Abspiels feststehen: Ort, Situation, Flanke, Richtung, Höhe.
- **Modell A (02):** logistische Regression, schrittweise aufgebaut. Leave-one-match-out Log-Loss: Baseline 0,521 → Ort + Situation 0,503 → + Richtung 0,444 → + Höhe 0,390 (AUC 0,836). Flanken, Freistöße und Abstöße brauchen ein eigenes Kennzeichen, sonst werden Flanken und Abstöße deutlich überschätzt. Im Strafraum liegt Modell A 18 Prozentpunkte daneben.
- **Zonenmodell (02):** 7 Zonen statt linearer Koordinaten und Wechselwirkungen zwischen Richtung und Drittel. Log-Loss 0,376, AUC 0,847; in jeder Zone, Liga und Passart auf höchstens 1 Prozentpunkt kalibriert. Gradient Boosting mit denselben Merkmalen ist nicht besser.
- **Anwendung (03):** Kennzahl „Pässe über Erwartung“ pro 100 Pässe. Die Passquote misst vor allem, *welche* Pässe ein Spieler spielt (r = 0,92 mit der erwarteten Quote); die Kennzahl misst die Ausführung. Sobottka liegt vorn (+8,1, innerhalb der Rolle +7,8). Fair ist der Vergleich nur innerhalb der Rolle (Innenverteidiger +3,2, Zehner/Sturm −5,3), und die Zufallsschwankung (±2 bis ±5 pro 100) ist fast so groß wie die Unterschiede: ein Filter fürs Video-Scouting, kein Urteil.
- **Erweiterung mit Tracking (04):** Der Empfänger wird mit einer festen Regel geschätzt (Mitspieler 1 s vorausgerechnet; Punkte = Winkel zur Passrichtung + 0,5 je Meter). Zu den 17 Zonen-Merkmalen kommen 4: Druck auf den Passgeber, Entfernung zum Empfänger, Freiraum des Empfängers, Gegner im Korridor. Auf den 5.655 Pässen mit Tracking: Log-Loss 0,3765 → 0,3632 (−3,5 %, Bootstrap-Intervall unter 0), AUC 0,847 → 0,865. Der Gewinn entsteht dort, wo der Empfänger klar bestimmt ist; der Engpass ist die Empfänger-Schätzung (82 % richtig bei angekommenen Pässen, 53 % bei hohen Bällen, 23 % bei Steilpässen).
- **Grenzen:** Das Zonenmodell kennt keine Gegnerpositionen (greift Notebook 04 auf); harte Zonengrenzen; nur 7 Spiele. Im Empfänger-Modell ist der Empfänger bei Fehlpässen nur geschätzt.

## Koordinaten

x = Längsachse (0 = Mittellinie, +52,5 = gegnerische Torlinie), y = Querachse (0 = Mitte, positiv = links in Angriffsrichtung). Das passende Team greift immer nach rechts an.
