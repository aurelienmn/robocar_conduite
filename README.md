## Architecture

```text
OAK-D Lite camera
  -> Client/live_perception.py      detecte les bandes blanches
  -> mask-generator cast_rays       calcule les raycasts configures
  -> Client/live_controller.py      choisit throttle + steering
  -> Client/vesc_control.py         envoie moteur + servo au VESC
```

Le mode manuel existant reste dans `Client/conduite_manuelle.py`.

## Installation sur la voiture

Depuis la voiture en SSH:

```bash
cd robocar_conduite
python3 -m venv venv
source venv/bin/activate
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -r requirements.txt
```

`requirements.txt` utilise `opencv-python-headless`, adapte pour la voiture/Jetson
en SSH. Les scripts `--headless`, `debug_camera.py` et `conduite_ia_live.py`
fonctionnent avec cette version.

Pour un test local avec fenetre OpenCV sur Mac (`cv2.imshow`, `Client/test_camera.py`):

```bash
python -m pip uninstall -y opencv-python opencv-python-headless
python -m pip install -r requirements-gui.txt
```

Ne pas garder `opencv-python` et `opencv-python-headless` installes ensemble dans
le meme venv: les deux fournissent le module `cv2`.

Le projet est pinne sur `depthai==2.32.0.0`.
Sur Mac/OAK-D Lite, `depthai==2.21.2.0` peut produire des frames uniformes
sans details (`spatial_std=0.0`). Eviter DepthAI v3 pour ce repo car les
scripts utilisent l'API DepthAI v2 (`XLinkOut`).

Le code IA utilise aussi le raycast du repo `robocar/mask-generator`. Garder l'archi :

```text
EPITECH/
  robocar/
    mask-generator/
  robocar_conduite/
```

Si le chemin est different:

```bash
export MASK_GENERATOR_ROOT=/chemin/vers/robocar/mask-generator
```

## Ordre de test

1. Tester la camera seule:

```bash
python Client/test_camera.py
```

Si l'image est uniforme/grise, lancer le diagnostic des sorties OAK:

```bash
python Client/diagnostic_oak_streams.py --attempts 30
```

Pour verifier la detection materielle de l'OAK:

```bash
python Client/diagnostic_oak_info.py
```

Pour forcer quelques reglages RGB et verifier si le flux est bloque par focus/exposition:

```bash
python Client/diagnostic_oak_rgb_controls.py --attempts 30
```

Le script sauvegarde les sorties `rgb_preview`, `rgb_video`, `rgb_isp`,
`mono_left` et `mono_right` dans `debug_oak_streams/`.
Si toutes les sorties ont `warning=flat_stream`, le probleme vient du flux camera
ou du branchement, pas du modele IA.

Pour tester rapidement une source camera dans l'IA sans modifier le JSON:

```bash
python Client/debug_camera.py --camera-source rgb_isp --attempts 30
python Client/preview_ia_live.py --headless --camera-source mono_left --frames 20
```

2. Tester l'IA sans moteurs, en SSH:

```bash
python Client/preview_ia_live.py --headless --frames 20 --max-fps 3
```

Verifier que:

- `raycast` contient le nombre de valeurs configure dans `n_rays`;
- `mask_fraction` n'est pas toujours 0;
- `steering` change quand on oriente la voiture vers une bande blanche.

3. Tester la boucle IA sans ouvrir le VESC:

```bash
python Client/conduite_ia_live.py --dry-run --frames 20 --max-fps 3
```

4. Premier test avec moteurs, voiture levee:

```bash
python Client/conduite_ia_live.py --frames 100
```

5. Premier test au sol: baisser la vitesse dans `Client/live_config.json`:

```json
"max_throttle": 0.04
```

## Reglages importants

Dans `Client/live_config.json`:

- `white_value_min`: augmenter si trop de bruit clair est detecte;
- `saturation_max`: baisser si des couleurs sont prises pour du blanc;
- `min_frame_std`: rejette les frames uniformes/grises de la camera;
- `roi_top_fraction`: augmenter si le haut de l'image perturbe;
- `roi_bottom_fraction`: ignore une petite bande en bas pour eviter que des reflets/bruits juste devant la camera declenchent tous les raycasts a 1 px;
- `min_component_area`: supprime les petits points blancs isoles avant les raycasts;
- `max_throttle`: limite de vitesse pour les tests;
- `emergency_distance_px`: distance minimale avant ralentissement/evasion.

## Mode bas FPS

La config par defaut est volontairement prudente pour une Jetson autour de 2-3 FPS:

- la camera est demandee a 6 FPS pour limiter la latence;
- les premieres frames OAK sont ignorees (`warmup_frames`) pour eviter les images grises de demarrage;
- la queue OAK garde seulement la frame la plus recente;
- la direction est lissee avec le temps reel, pas avec le nombre de frames;
- si une frame arrive trop tard (`stale_frame_timeout_s`), la voiture stoppe;
- si le FPS baisse, le throttle est reduit automatiquement.

Pour le jour J, garder `--max-fps 3` et monter `max_throttle` seulement quand le masque
detecte correctement les bandes blanches.

## Enregistrer des donnees reelles pour entrainer ensuite

Pendant une conduite manuelle sur la vraie piste:

```bash
python Client/enregistrer_donnees_reelles.py
```

Sortie:

```text
data/real/run_YYYYMMDD_HHMMSS/
  manifest.csv
  frames/
    frame_000000.png
```
