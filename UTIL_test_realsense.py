import pyrealsense2 as rs
import numpy as np
import cv2
import time

# realsense initialization
print("Realsense!")
frameWidth = 1280
frameHeight = 720
pipeline = rs.pipeline()
config = rs.config()

# Configure the color and depth streams with equal resolution and frame rate
# and the corresponding formats
# config.enable_stream(rs.stream.color, 256, 144, rs.format.bgr8, 90)
# config.enable_stream(rs.stream.depth, 256, 144, rs.format.z16, 90)

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
# self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 90)
# self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 90)
#
# self.config.enable_stream(rs.stream.color, frameWidth, frameHeight, rs.format.bgr8, 30)
# self.config.enable_stream(rs.stream.depth, frameWidth, frameHeight, rs.format.z16, 30)

# Start streaming and create the alignment object
profiles = pipeline.start()
align_to = rs.stream.color
aligner = rs.align(align_to=align_to)

# Get the camera parameters of the depth stream for complete computation of 3D location
depth_stream = profiles.get_stream(rs.stream.depth).as_video_stream_profile()
f = depth_stream.get_intrinsics().fx
cx = depth_stream.get_intrinsics().ppx
cy = depth_stream.get_intrinsics().ppy
intrinsics = depth_stream.get_intrinsics()
camera_mat = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])

while True:

    # REALSENSE issues : Can't seem to change the width and height of the glove

    # Wait for a coherent frame
    incoming_data = pipeline.wait_for_frames()
    aligned_frames = aligner.process(frames=incoming_data)
    color = aligned_frames.get_color_frame()
    depth = aligned_frames.get_depth_frame()

    if not depth or not color:
        continue

    # Convert images to numpy arrays
    color_img = np.asanyarray(color.get_data())
    depth_img = np.asanyarray(depth.get_data())

    # print(depth_img.shape)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_img, alpha=0.03), cv2.COLORMAP_JET)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2BGR)

    image = np.hstack((color_img, depth_colormap))
    image = cv2.resize(image, (int(image.shape[1] * 0.6), int(image.shape[0] * 0.6)))

    cv2.imshow('RealSense', image)
    cv2.waitKey(1)