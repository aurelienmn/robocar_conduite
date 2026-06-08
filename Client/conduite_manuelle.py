import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import Gamepad


def limiter(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def normalize_trigger(value):
    if value < 0:
        return (value + 1.0) / 2.0
    return value


def set_motor(throttle):
    """
    throttle :
    -1.0 = marche arrière
     0.0 = stop
     1.0 = avant max

    À adapter selon ton driver moteur / ESC.
    """
    print(f"MOTEUR: {throttle:.2f}")


def set_steering(steering):
    """
    steering :
    -1.0 = gauche
     0.0 = centre
     1.0 = droite

    À adapter selon ton servo de direction.
    """
    print(f"DIRECTION: {steering:.2f}")


def stop_car():
    set_motor(0.0)
    set_steering(0.0)


def main():
    gamepad = Gamepad.Xbox360()
    gamepad.startBackgroundUpdates()

    print("Conduite manuelle démarrée")
    print("Stick gauche = direction")
    print("RT = accélérer")
    print("LT = freiner / reculer")
    print("START = arrêt")

    try:
        while gamepad.isConnected():
            steering = limiter(gamepad.axis("LX"))

            rt = normalize_trigger(gamepad.axis("RT"))
            lt = normalize_trigger(gamepad.axis("LT"))

            throttle = limiter(rt - lt) * 0.2

            set_steering(steering)
            set_motor(throttle)

            if gamepad.beenPressed("START"):
                break

            time.sleep(0.02)

    finally:
        print("Arrêt sécurité")
        stop_car()
        gamepad.disconnect()


if __name__ == "__main__":
    main()