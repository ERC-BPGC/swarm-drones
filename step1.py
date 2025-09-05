from pseyepy import Camera, Display
import cv2  # OpenCV for image handling
import time

cam_index = 1  # Camera index

# Initialize camera
# cams = Camera(fps=60, resolution=Camera.RES_LARGE)  # wrong camera mode
cams = Camera(fps=90, resolution=Camera.RES_SMALL)  # camera mode which is used in main mocap logic

for i in range(50):  # Capture 10 images
    frame, timestamp = cams.read()
    filename = f'cam_{cam_index}/image_{i}.jpg'
    print(f'Captured image {i} at {timestamp} with shape {frame.shape} \n')
    cv2.imshow('frame', frame)
    cv2.waitKey(0)
    cv2.imwrite(filename, frame)
    time.sleep(0.5)  # Wait a second between captures

# Clean up
cams.end()