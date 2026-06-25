import os
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent.parent


def _candidate_roots():
    # type: () -> List[Path]
    candidates = []
    env_path = os.environ.get("MASK_GENERATOR_ROOT")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        ROOT / "mask-generator",
        ROOT.parent / "robocar" / "mask-generator",
        ROOT.parent / "mask-generator",
    ])
    return candidates


def load_cast_rays():
    # Priorite 1 : raycast.py local dans le meme dossier que ce fichier
    local = Path(__file__).resolve().parent / "raycast.py"
    if local.exists():
        from raycast import cast_rays
        return cast_rays

    # Priorite 2 : repo mask-generator externe
    for candidate in _candidate_roots():
        if (candidate / "src" / "raycast" / "raycast.py").exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            from src.raycast.raycast import cast_rays
            return cast_rays

    searched = "\n".join("- {}".format(p) for p in _candidate_roots())
    raise ImportError(
        "Impossible de trouver le mask-generator. Clone le repo robocar a cote "
        "de robocar_conduite ou exporte MASK_GENERATOR_ROOT.\n"
        "Chemins testes:\n{}".format(searched)
    )
