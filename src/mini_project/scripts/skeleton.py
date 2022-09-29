#! /usr/bin/env python
import rospy
import cv2 as cv
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np


class HSVDetector:
    def __init__(self, hsv_min, hsv_max):
        self.hsv_min = hsv_min
        self.hsv_max = hsv_max

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
            contour = contours[0]

            x, y, w, h = cv.boundingRect(contour)

            return x, y, w, h

        return None


def detect_free_space(depth_image, blocks):
    """
    TASK:
        Fill in this function.

        This function takes a depth image and detections from one or more HSVDetector.

        It should return x,y coordinates of a point where a new block can be placed.

        The returned point must be on the table and away from the detected blocks.

        You may assume that the table is clear (except for the detected blocks),
        and that the camera is looking straight down on the table.

    HINTS:
        The floor has depth-value 0.-
        Look at the OpenCV function distanceTransform().

    Parameters:
        depth_image: An OpenCV depth image.
        blocks: Detected blocks in the form of a dictionary {color: bounding_box, ...}.

    Returns:
        A tuple (x, y) representing a free-space position, where a new block could be placed.
    """
    return (0, 0)


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
        'blue':   HSVDetector(hsv_min=(90, 120, 200),
                              hsv_max=(110, 150, 255)),

        'orange': HSVDetector(hsv_min=(10, 120, 200),
                              hsv_max=(30, 160, 255)),

        'yellow': HSVDetector(hsv_min=(31, 60, 200),
                              hsv_max=(45, 100, 255)),

        'green':  HSVDetector(hsv_min=(46, 60, 200),
                              hsv_max=(65, 100, 255)),
    }

    while True:
        depth = depth_listener.get()
        rgb = rgb_listener.get()

        blocks = detect_blocks(rgb, hsv_detectors)
        # free_space = detect_free_space(depth, blocks)

        for color, (x, y, w, h) in blocks.items():
            cx = int(x + w/2)
            cy = int(y + h/2)
            cv.circle(rgb, (cx, cy), 10, (0, 0, 255))
            cv.putText(rgb, color, (cx-25, cy-15),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

        # x, y = free_space
        # x = int(x)
        # y = int(y)
        # cv.circle(rgb, (x, y), 10, (0, 255, 0))
        # cv.putText(rgb, 'Free space', (x-25, y-15),
        #            cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

        cv.imshow('cv_window', rgb)
        key = cv.waitKey(3) & 0xff
        if key == ord('q'):
            break


if __name__ == '__main__':
    main()
