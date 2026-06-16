from __future__ import annotations

import json
from typing import Any


def stringify(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {stringify(k): stringify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [stringify(v) for v in value]
    return str(value)


def main() -> None:
    import depthai as dai

    payload: dict[str, Any] = {
        "depthai_version": dai.__version__,
        "available_devices": [],
    }

    for info in dai.Device.getAllAvailableDevices():
        payload["available_devices"].append(
            {
                "name": stringify(getattr(info, "name", None)),
                "mxid": stringify(getattr(info, "mxid", None)),
                "state": stringify(getattr(info, "state", None)),
            }
        )

    try:
        with dai.Device() as device:
            payload["device"] = {
                "name": stringify(device.getDeviceName()),
                "mxid": stringify(device.getMxId()),
                "usb_speed": stringify(device.getUsbSpeed()),
                "connected_cameras": stringify(device.getConnectedCameras()),
                "camera_sensor_names": stringify(device.getCameraSensorNames()),
                "connected_camera_features": stringify(device.getConnectedCameraFeatures()),
            }
    except Exception as exc:
        payload["device_error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
