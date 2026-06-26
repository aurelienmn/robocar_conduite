from live_settings import CameraSettings


def _socket(dai, modern_name: str, legacy_name: str):
    sockets = dai.CameraBoardSocket
    if hasattr(sockets, modern_name):
        return getattr(sockets, modern_name)
    return getattr(sockets, legacy_name)


class OakCamera:
    """OAK-D Lite camera wrapper returning OpenCV BGR frames."""

    def __init__(self, settings: CameraSettings) -> None:
        import depthai as dai

        self.dai = dai
        self.settings = settings
        self.pipeline = dai.Pipeline()

        xout = self.pipeline.create(dai.node.XLinkOut)
        xout.setStreamName("rgb")

        if settings.source.startswith("mono_"):
            cam = self.pipeline.create(dai.node.MonoCamera)
            cam.setBoardSocket(self._mono_socket(settings.source))
            cam.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
            cam.setFps(settings.fps)
            cam.out.link(xout.input)
        else:
            cam = self.pipeline.create(dai.node.ColorCamera)
            cam.setBoardSocket(_socket(dai, "CAM_A", "RGB"))
            cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
            cam.setFps(settings.fps)
            cam.setInterleaved(False)
            cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

            if settings.exposure_compensation != 0:
                cam.initialControl.setAutoExposureCompensation(settings.exposure_compensation)

            if settings.source == "rgb_preview":
                cam.setPreviewSize(settings.width, settings.height)
                cam.preview.link(xout.input)
            elif settings.source == "rgb_video":
                cam.setVideoSize(settings.width, settings.height)
                cam.video.link(xout.input)
            elif settings.source == "rgb_isp":
                cam.isp.link(xout.input)
            else:
                raise ValueError(f"Unknown camera source: {settings.source}")

        self.device = None
        self.queue = None
        self.frames_to_skip = max(0, settings.warmup_frames)

    def __enter__(self) -> "OakCamera":
        self.device = self.dai.Device(self.pipeline)
        self.queue = self.device.getOutputQueue(name="rgb", maxSize=1, blocking=False)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def read(self):
        if self.queue is None:
            raise RuntimeError("Camera not started. Use 'with OakCamera(...) as camera'.")

        packet = self.queue.get()
        while self.frames_to_skip > 0:
            self.frames_to_skip -= 1
            packet = self.queue.get()

        while True:
            next_packet = self.queue.tryGet()
            if next_packet is None:
                break
            packet = next_packet
        frame = packet.getCvFrame()
        if frame.ndim == 2:
            frame = self._cv2().cvtColor(frame, self._cv2().COLOR_GRAY2BGR)
        if frame.shape[1] != self.settings.width or frame.shape[0] != self.settings.height:
            frame = self._cv2().resize(frame, (self.settings.width, self.settings.height))
        return frame

    def _mono_socket(self, source: str):
        if source == "mono_left":
            return _socket(self.dai, "CAM_B", "LEFT")
        if source == "mono_right":
            return _socket(self.dai, "CAM_C", "RIGHT")
        raise ValueError(f"Unknown camera source: {source}")

    @staticmethod
    def _cv2():
        import cv2

        return cv2

    def close(self) -> None:
        if self.device is not None:
            self.device.close()
            self.device = None
            self.queue = None
