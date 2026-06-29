# Guide test voiture

## Setup

```bash
cd ~/Documents/robocar_conduite
git pull
source venv/bin/activate
cd Client
```

---

## 1. Verifier la camera + mask

```bash
python3 camera_stream.py
```

Ouvre `http://<IP_DE_LA_PI>:5000` dans le navigateur.
Tu dois voir les bandes blanches en rouge et les rayons bleus.

---

## 2. Tester sans moteurs

```bash
python3 conduite_ia_live.py --dry-run
```

Oriente la voiture a la main, les logs doivent afficher `steering` qui change.

---

## 3. Lancer en autonome

```bash
python3 conduite_ia_live.py
```

Chaque lancement cree automatiquement un dossier de logs:

```text
logs/ia_runs/YYYYMMDD_HHMMSS_live/
  config.json
  events.jsonl
  summary.json
```

Pour un test sans moteurs, le suffixe sera `_dry`.

---

## Reglages - tout est dans live_config.json

| Probleme | Parametre | Action |
|---|---|---|
| Trop rapide | `base_throttle` | Baisser (ex: 0.022) |
| Trop lent | `base_throttle` | Augmenter (ex: 0.035) |
| Vire trop peu au centre de piste | `centerline_gain` / `heading_gain` | Augmenter legerement |
| Evite trop tard une bande proche | `boundary_guard_distance_px` | Augmenter |
| Evite trop fort une bande proche | `boundary_guard_steering` | Baisser |
| Raycast fallback vire trop peu | `steering_gain` | Augmenter legerement |
| Conduite saccadee | `steering_smoothing` | Augmenter (ex: 0.5) |
| Freine trop en virage | `throttle_turn_slowdown` | Baisser (ex: 0.4) |
| Decor detecte en rouge | `min_bottom_fraction` | Augmenter (ex: 0.55) |
| Bandes non vues de loin | `roi_top_fraction` | Baisser (ex: 0.10) |

Apres chaque modif -> relancer `python3 conduite_ia_live.py`.
