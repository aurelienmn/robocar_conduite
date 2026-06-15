import cv2
import depthai as dai

pipeline = dai.Pipeline()

cam = pipeline.create(dai.node.ColorCamera)
cam.setPreviewSize(640, 480)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("rgb")
cam.preview.link(xout.input)

with dai.Device(pipeline) as device:
    q = device.getOutputQueue("rgb")

    while True:
        frame = q.get().getCvFrame()

        cv2.imshow("OAK-D Lite", frame)

        if cv2.waitKey(1) == ord("q"):
            break