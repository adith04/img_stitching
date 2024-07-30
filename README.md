# Stitching

The stitching ros package is designed to stitch together images captured by multiple cameras on a drone, creating a seamless panoramic view. This package includes two main programs:

# Programs

- panorama.py - This script is responsible for stitching the images. It takes the individual images from the drone's cameras and combines them into a single, continuous panoramic image using an iterative stitching process.

- yolo_detections.py - This script handles object detection within the stitched panoramic image. It uses the YOLOv8 model to detect and identify various objects, drawing masks on them for easy visualization.
