import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import Gamepad


def safe_axis(gamepad, name):
    try:
        return gamepad.axis(name)
    except Exception:
        return None


def safe_button(gamepad, name):
    try:
        return gamepad.isPressed(name)
    except Exception:
        return False


def main():
    gamepad = Gamepad.Xbox360()
    gamepad.startBackgroundUpdates()

    print("Test manette démarré")
    print("Bouge les sticks, appuie sur les gâchettes et les boutons.")
    print("START pour quitter.")

    axes = ["LX", "LY", "RX", "RY", "LT", "RT", "DX", "DY"]
    buttons = ["A", "B", "X", "Y", "LB", "RB", "BACK", "START", "LTHUMB", "RTHUMB"]

    try:
        while gamepad.isConnected():
            print("\033c", end="")

            print("=== AXES ===")
            for axis in axes:
                value = safe_axis(gamepad, axis)
                if value is not None:
                    print(f"{axis:6} : {value: .3f}")

            print("\n=== BOUTONS ===")
            for button in buttons:
                pressed = safe_button(gamepad, button)
                print(f"{button:6} : {'APPUYÉ' if pressed else '-'}")

            if safe_button(gamepad, "START"):
                print("\nArrêt demandé.")
                break

            time.sleep(0.1)

    finally:
        gamepad.disconnect()
        print("Manette déconnectée.")


if __name__ == "__main__":
    main()