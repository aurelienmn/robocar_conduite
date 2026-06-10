import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import Gamepad
from pyvesc import encode
from pyvesc.messages.setters import SetDutyCycle
import serial


VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200
MAX_SPEED = 0.20


def limiter(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def normalize_trigger(value):
    if value < 0:
        return (value + 1.0) / 2.0
    return value


vesc = serial.Serial(VESC_PORT, VESC_BAUDRATE, timeout=0.1)


def set_motor(throttle):
    throttle = limiter(throttle, -MAX_SPEED, MAX_SPEED)
    packet = encode(SetDutyCycle(throttle))
    vesc.write(packet)
    print(f"MOTEUR: {throttle:.2f}")


def set_steering(steering):
    steering = limiter(steering)
    print(f"DIRECTION: {steering:.2f}")


def stop_car():
    set_motor(0.0)
    set_steering(0.0)


def main():
    gamepad = Gamepad.PS4()
    gamepad.startBackgroundUpdates()

    print("Conduite manuelle démarrée")
    print("Stick gauche = direction")
    print("R2 = accélérer")
    print("L2 = freiner / reculer")
    print("OPTIONS = arrêt")
    print("Vitesse limitée à 20 %")

    try:
        while gamepad.isConnected():
            steering = limiter(gamepad.axis("LEFT-X"))

            rt = normalize_trigger(gamepad.axis("R2"))
            lt = normalize_trigger(gamepad.axis("L2"))

            throttle = limiter(rt - lt) * MAX_SPEED

            set_steering(steering)
            set_motor(throttle)

            if gamepad.beenPressed("OPTIONS"):
                break

            time.sleep(0.02)

    finally:
        print("Arrêt sécurité")
        stop_car()
        gamepad.disconnect()
        vesc.close()


if __name__ == "__main__":
    main()