#! /usr/bin/env python
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import sys

class HSVDetector:
    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

    # TODO: Improvement: Base detection also on depth information to improve detection success rate
    def detect(self, hsv_image):
        """
        TASK:
            Fill in this function

            This function should make use of the min and max hsv values
            in self.hsv_min and self.hsv_max.

            It should perform HSV detection using these values,
            and return a rectangle bounding box for its detection.

        HINTS:
            Your detector may pick up some pixels that are not part of the toy blocks.
            The OpenCV functions findContours() and contourArea() may help you deal with this.

        Parameters:
            hsv_image: An OpenCV image in HSV color space.

        Returns:
            A rectangle bounding box as a tuple (x, y, width, height)
            representing the detection made by this detector.
            If detection fails, return None.
        """

        thresh = cv.inRange(hsv_image, self.hsv_min, self.hsv_max)

        contours, _ = cv.findContours(
            thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        if contours:
            contour_areas = [(contour, cv.contourArea(contour)) for contour in contours]
            tuple = max(contour_areas, key=lambda x: x[1])
            contour = tuple[0]
            x, y, w, h = cv.boundingRect(contour)

            return x, y, w, h

        return None


def detect_free_space(depth_image, blocks):
    
    # Convert scale of image to absolute values
    image_scaled = cv.convertScaleAbs(depth_image)

    # Dilate and erode to get rid of noise in depth image
    kernel = np.ones((20, 20), np.uint8)
    image_scaled = cv.dilate(image_scaled, kernel, iterations=1)
    image_scaled = cv.erode(image_scaled, kernel, iterations=1)
    
    # Set borders to zero
    image_scaled[0] = 0
    image_scaled[:,-1] = 0
    image_scaled[-1] = 0
    image_scaled[:,0] = 0

    # Set block coordinates to zero
    for _, (x, y, w, h) in blocks.items():
        cx = int(x + w/2)
        cy = int(y + h/2)
        image_scaled[cy,cx] = 0

    # Distance transformation
    image_distance = cv.distanceTransform(image_scaled, cv.DIST_L2, 0)
    # Normalize image in case we want to show it
    image_normalized = cv.normalize(image_distance, None, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)

    # Get coordinates of best spot to place the new brick
    index_free_space = np.argmax(image_normalized)
    y,x = np.unravel_index(index_free_space, image_normalized.shape)

    return (x,y)

class DepthListener:
    def __init__(self, topic='/realsense/aligned_depth_to_color/image_raw'):
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)

        # Since we check for None in the get function, we can set this to anything but None
        self.image = 0.0

        rospy.Subscriber(topic, Image, callback=self.callback)

    def callback(self, data):
        try:
            self.image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding="passthrough")
        except CvBridgeError as e:
            print(e)

        # FOR VIZ ONLY
        # depth_image_scaled = cv.convertScaleAbs(self.image, alpha=0.03)
        # depth_image_colormap = cv.applyColorMap(
        #     depth_image_scaled, cv.COLORMAP_JET)
        # cv.imshow('depth image', depth_image_colormap)
        # cv.waitKey(0)

        """
        TASK:
            The variable 'data' is a ROS message containing a depth image.
            Transform it into an OpenCV image using self.bridge.

            The result should be stored in self.image.

        HINTS:
            The resulting image should be a single-channel image of type uint16.
        """

    def get(self):
        self.image = None
        while self.image is None:
            self.rate.sleep()
        return self.image


class RGBListener:
    def __init__(self, topic='/realsense/rgb/image_raw'):
        self.bridge = CvBridge()
        self.rate = rospy.Rate(10)

        # Since we check for None in the get function, we can set this to anything but None
        self.image = 0.0

        rospy.Subscriber(topic, Image, callback=self.image_callback)

    def image_callback(self, data):
        try:
            # We select bgr8 because its the OpneCV encoding by default
            self.image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding="bgr8")
        except CvBridgeError as e:
            print(e)

        # cv.imshow("Image window", self.image)
        # cv.waitKey(1)

        """
        TASK:
            The variable 'data' is a ROS message containing a RGB image.
            Transform it into an OpenCV image using self.bridge.

            The result should be stored in self.image.

        HINTS:
            The resulting image should be an 8-bit image in BGR color space.
        """

    def get(self):
        self.image = None
        while self.image is None:
            self.rate.sleep()
        return self.image

# This is a convenience function for running multiple HSV detectors on a single image.


def detect_blocks(rgb_image, detectors):
    hsv = cv.cvtColor(rgb_image, cv.COLOR_BGR2HSV)

    out = {}
    for key, detector in detectors.items():
        det = detector.detect(hsv)
        if det is not None:
            out[key] = det

    return out

# TODO: Make node quitable
def main():
    rospy.init_node('perception_solution')

    depth_listener = DepthListener()
    rgb_listener = RGBListener()

    """
    TASK:
        Fill in appropriate min and max values for HSV detection.

    HINTS:
        Make a window with sliders like the one you saw in the ConstructSim course.
    """
    hsv_detectors = {
        'blue':   HSVDetector(hsv_min=(90, 200, 200),
                              hsv_max=(110, 255, 255)),

        'orange': HSVDetector(hsv_min=(15, 140, 140),
                              hsv_max=(25, 255, 255)),

        'yellow': HSVDetector(hsv_min=(25, 50, 140),
                              hsv_max=(35, 255, 255)),

        'green':  HSVDetector(hsv_min=(65, 120, 120),
                              hsv_max=(75, 176, 255)),
    }

    # TODO: @TA: Why does shutdown not work as expected when removing the print statements?
    while not rospy.is_shutdown():
        depth = depth_listener.get()
        rgb = rgb_listener.get()

        blocks = detect_blocks(rgb, hsv_detectors)
        free_space = detect_free_space(depth, blocks)

            # rospy.logwarn("0") 
            # TODO: (at some point) Enable detection of multiple objects of same color
            #       Maybe then introduce a threshold for the blob size to filter out noise
        
        for color, (x, y, w, h) in blocks.items():
            cx = int(x + w/2)
            cy = int(y + h/2)
            cv.circle(rgb, (cx, cy), 10, (0, 0, 255))
            cv.putText(rgb, color, (cx-25, cy-15),
                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        x, y = free_space
        x = int(x)
        y = int(y)
        cv.circle(rgb, (x, y), 10, (0, 255, 0))
        cv.putText(rgb, 'Free space', (x-25, y-15),
                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow('cv_window', rgb)
        key = cv.waitKey(3) & 0xff
        if key == ord('q'):
            break

    rospy.sleep(1)
    cv.destroyAllWindows()
    rospy.logwarn("Shutting down...") 

if __name__ == '__main__':
    main()
