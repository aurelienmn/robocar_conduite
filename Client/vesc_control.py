import struct
from typing import Union

from live_settings import VescSettings


COMM_SET_DUTY = 5
COMM_SET_SERVO_POS = 12


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def crc16(data: Union[bytes, bytearray]) -> int:
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


class VescController:
    def __init__(self, settings: VescSettings) -> None:
        import serial

        self.settings = settings
        self.serial = serial.Serial(settings.port, settings.baudrate, timeout=0.1)

    def close(self) -> None:
        if self.serial.is_open:
            self.serial.close()

    def _send_packet(self, payload: bytearray) -> None:
        packet = bytearray()
        packet.append(2)
        packet.append(len(payload))
        packet.extend(payload)

        crc = crc16(payload)
        packet.append((crc >> 8) & 0xFF)
        packet.append(crc & 0xFF)
        packet.append(3)

        self.serial.write(packet)

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self.settings.deadzone else value

    def set_motor(self, throttle: float) -> float:
        throttle = self._apply_deadzone(throttle)
        throttle = clamp(throttle, -self.settings.max_duty, self.settings.max_duty)

        duty_value = int(throttle * 100000)
        payload = bytearray()
        payload.append(COMM_SET_DUTY)
        payload.extend(struct.pack(">i", duty_value))
        self._send_packet(payload)
        return throttle

    def set_steering(self, steering: float) -> float:
        steering = self._apply_deadzone(steering)
        steering = clamp(steering, -1.0, 1.0)

        servo_pos = self.settings.servo_center + steering * self.settings.servo_range
        servo_pos = clamp(servo_pos, self.settings.servo_min, self.settings.servo_max)

        servo_value = int(servo_pos * 1000)
        payload = bytearray()
        payload.append(COMM_SET_SERVO_POS)
        payload.extend(struct.pack(">h", servo_value))
        self._send_packet(payload)
        return servo_pos

    def stop(self) -> None:
        self.set_motor(0.0)
        self.set_steering(0.0)
