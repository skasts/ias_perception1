#! /usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

def callback(value):
    pass

def make_bars():
    cv2.namedWindow('HSV', 0)
    cv2.createTrackbar('hmin', 'HSV', 0, 255, callback)
    cv2.createTrackbar('hmax', 'HSV', 255, 255, callback)
    cv2.createTrackbar('smin', 'HSV', 0, 255, callback)
    cv2.createTrackbar('smax', 'HSV', 255, 255, callback)
    cv2.createTrackbar('vmin', 'HSV', 0, 255, callback)
    cv2.createTrackbar('vmax', 'HSV', 255, 255, callback)

def get_values():
    return {
        'min': (cv2.getTrackbarPos('hmin', 'HSV'), cv2.getTrackbarPos('smin', 'HSV'), cv2.getTrackbarPos('vmin', 'HSV')),
        'max': (cv2.getTrackbarPos('hmax', 'HSV'), cv2.getTrackbarPos('smax', 'HSV'), cv2.getTrackbarPos('vmax', 'HSV'))
    }

def image_callback(data):
    global bridge
    try:
        cv_image = bridge.imgmsg_to_cv2(data, 'bgr8')
    except CvBridgeError as e:
        print(e)

    vals = get_values()

    hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, vals['min'], vals['max'])

    cv_image = cv2.bitwise_and(cv_image, cv_image, mask=mask)

    cv2.imshow('test', cv_image)
    cv2.waitKey(3)

def main():
    global bridge
    make_bars()
    rospy.init_node('hsv_sampler')
    bridge = CvBridge()
    rospy.Subscriber('/realsense/rgb/image_raw', Image, callback=image_callback)
    rospy.spin()


if __name__ == '__main__':
    main()