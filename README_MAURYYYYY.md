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

---

## 4. Si les virages sont problematiques - activer le modele ML

```bash
# 1. Enregistrer des tours manuels avec la manette
python3 enregistrer_donnees_reelles.py

# 2. Entrainer le modele directement sur la Pi (c'est petit ca devrai aller)
#    Le nom du dossier run_XXXX est affiche dans le terminal a l'etape 1
cd ..
python3 train_corner_model.py data/real/run_XXXXXXXX

# 3. Relancer la voiture -> mode hybride active automatiquement
cd Client
python3 conduite_ia_live.py
```

---

## Reglages - tout est dans live_config.json

| Probleme | Parametre | Action |
|---|---|---|
| Trop rapide | `base_throttle` | Baisser (ex: 0.022) |
| Trop lent | `base_throttle` | Augmenter (ex: 0.035) |
| Vire trop peu | `steering_gain` | Augmenter (ex: 1.4) |
| Vire trop fort | `steering_gain` | Baisser (ex: 0.8) |
| Conduite saccadee | `steering_smoothing` | Augmenter (ex: 0.5) |
| Freine trop en virage | `throttle_turn_slowdown` | Baisser (ex: 0.4) |
| Decor detecte en rouge | `min_bottom_fraction` | Augmenter (ex: 0.55) |
| Bandes non vues de loin | `roi_top_fraction` | Baisser (ex: 0.10) |

Apres chaque modif -> relancer `python3 conduite_ia_live.py`.

