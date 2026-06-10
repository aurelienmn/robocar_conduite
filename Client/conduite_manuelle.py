# coding: utf-8

import time
import sys
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT / "Gamepad"))

import Gamepad
import serial


VESC_PORT = "/dev/ttyACM0"
VESC_BAUDRATE = 115200

ARDUINO_PORT = "/dev/ttyUSB0"
ARDUINO_BAUDRATE = 115200

MAX_SPEED = 0.08
DEADZONE = 0.03

STEERING_CENTER = 90
STEERING_RANGE = 35
STEERING_MIN = STEERING_CENTER - STEERING_RANGE
STEERING_MAX = STEERING_CENTER + STEERING_RANGE

COMM_SET_DUTY = 5


def limiter(value, min_value=-1.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def apply_deadzone(value):
    if abs(value) < DEADZONE:
        return 0.0
    return value


def normalize_trigger(value):
    if value < 0:
        return (value + 1.0) / 2.0
    return value


def crc16(data):
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = (crc << 1) ^ 0x1021
            else:
                crc <<= 1
            crc &= 0xFFFF
    return crc


def send_vesc_packet(ser, payload):
    packet = bytearray()
    packet.append(2)
    packet.append(len(payload))
    packet.extend(payload)

    crc = crc16(payload)
    packet.append((crc >> 8) & 0xFF)
    packet.append(crc & 0xFF)
    packet.append(3)

    ser.write(packet)


vesc = serial.Serial(VESC_PORT, VESC_BAUDRATE, timeout=0.1)
arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUDRATE, timeout=0.1)
time.sleep(2)


def set_motor(throttle):
    throttle = apply_deadzone(throttle)
    throttle = limiter(throttle, -MAX_SPEED, MAX_SPEED)

    duty_value = int(throttle * 100000)

    payload = bytearray()
    payload.append(COMM_SET_DUTY)
    payload.extend(struct.pack(">i", duty_value))

    send_vesc_packet(vesc, payload)

    print(f"MOTEUR: {throttle:.3f}")


def set_steering(steering):
    steering = apply_deadzone(steering)
    steering = limiter(steering)

    angle = int(STEERING_CENTER + steering * STEERING_RANGE)
    angle = max(STEERING_MIN, min(STEERING_MAX, angle))

    arduino.write(f"S:{angle}\n".encode())

    print(f"DIRECTION: {steering:.2f} | ANGLE: {angle}")


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
    print(f"Vitesse limitée à {MAX_SPEED * 100:.0f} %")

    try:
        while gamepad.isConnected():
            steering = gamepad.axis("LEFT-X")

            rt = normalize_trigger(gamepad.axis("R2"))
            lt = normalize_trigger(gamepad.axis("L2"))

            raw_throttle = rt - lt
            throttle = raw_throttle * MAX_SPEED

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
        arduino.close()


if __name__ == "__main__":
    main()