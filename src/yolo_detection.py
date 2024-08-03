#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
from ultralytics import YOLO

class YOLODetector:
    def __init__(self):
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')  # Use appropriate YOLOv8 model
        self.image_sub = rospy.Subscriber('/stitched_image', Image, self.callback)
        self.image_pub = rospy.Publisher('/yolo/detections', Image, queue_size=10)
        self.mask_pub = rospy.Publisher('/yolo/mask', Image, queue_size=10)
        
        # COCO class names
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", 
            "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", 
            "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", 
            "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", 
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", 
            "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", 
            "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", 
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", 
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", 
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", 
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", 
            "toothbrush"
        ]

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")
            return
        
        # Create a black mask
        mask = np.zeros(cv_image.shape[:2], dtype=np.uint8)
        
        # Perform object detection
        results = self.model(cv_image)

        # Draw bounding boxes and labels, and mask the largest contour within each bounding box
        for result in results:
            for box, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                class_name = self.class_names[int(label)]
                label_text = f"{class_name} {score:.2f}"
                cv2.putText(cv_image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Extract the region of interest (ROI) from the image
                roi = cv_image[y1:y2, x1:x2]

                # Convert the ROI to grayscale
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

                # Apply Gaussian blur
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)


                # Apply Canny edge detector with fine-tuned parameters
                edges = cv2.Canny(blurred, 100, 215) #50 250
                # Apply morphological operations to clean up the edges
                kernel = np.ones((5, 5), np.uint8)
                edges = cv2.dilate(edges, kernel, iterations=2)
                edges = cv2.erode(edges, kernel, iterations=2)

                # Find contours in the edge-detected image
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # If contours are found, draw the largest one on the mask
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    # Filter out small contours based on area
                    if cv2.contourArea(largest_contour) > 100:  # Adjust the threshold as needed
                        # Create a contour mask with the same size as the ROI
                        contour_mask = np.zeros_like(gray, dtype=np.uint8)
                        cv2.drawContours(contour_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
                        # Place the contour mask in the correct position on the overall mask
                        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], contour_mask)
        
        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
            self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "mono8"))
            rospy.loginfo("Published mask")
        except CvBridgeError as e:
            rospy.logerr(f"CvBridge Error: {e}")

def main():
    rospy.init_node('yolo_detector', anonymous=True)
    yolo_detector = YOLODetector()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")

if __name__ == '__main__':
    main()
