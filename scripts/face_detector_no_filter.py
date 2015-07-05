#!/usr/bin/env python                                                          
import roslib;
roslib.load_manifest('face_detector')
import sys
import rospy
import cv2
import array
import numpy as np
from std_msgs.msg import String
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from compiler.ast import flatten
#import face_detector.msg 

#faceCascName = ["../config/haarcascade_frontalface_alt2.xml","../config/haarcascade_frontalface_alt.xml","../config/haarcascade_frontalface_default.xml",
#"../config/haarcascade_profileface.xml"]
faceCascName = ["../config/haarcascade_frontalface_alt.xml"]
faceCascade = []

class image_converter:
    def __init__(self):
        for i in range(len(faceCascName)):
            faceCascade.append(cv2.CascadeClassifier(faceCascName[i]))
        self.image_pub = rospy.Publisher("faces",Image)
        self.facenum_pub = rospy.Publisher("faces_number",Int32)
        cv2.namedWindow("Face Found",1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw",Image,self.callback)
    def callback(self,data):
        try:
            cv2.image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        except CvBridgeError,e:
            print e
        #convert img to grayscale
        gray = cv2.cvtColor(cv2.image,cv2.COLOR_BGR2GRAY)
        faces = []
        #cascading
        for i in range(len(faceCascade)):
            faces.append(faceCascade[i].detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.cv.CV_HAAR_SCALE_IMAGE
            ))
        #flatten face data
        flatten=lambda i,d=-1:[a for b in i for a in(flatten(b,d-(d>0))if hasattr(b,'__iter__')and d else(b,))]
        faces_new = flatten(faces,1)
        print "face number:", len(faces_new)
        #draw face detected rectangle
        #for face in faces:
        #    for (x, y, w, h) in face:
        #        cv2.rectangle(cv2.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        for i in range(len(faces_new)):
            faces_new[i] = cv2.image[faces_new[i][0]-20:(faces_new[i][0]+faces_new[i][2]+20),faces_new[i][1]-20:(faces_new[i][1]+faces_new[i][3]+20)]

        try:
            cv2.imshow("Faces found",faces_new[0])
            cv2.waitKey(3)
            for i in range(len(faces_new)):
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(faces_new[i],"bgr8"))
            #self.facenum_pub.publish(len(set(clusters))
        except CvBridgeError, e:
            print e
        facenum = Int32(len(faces_new))
        self.facenum_pub.publish(facenum)

def main(args):
    ic = image_converter()
    rospy.init_node("image_converter",anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print "Shutting down"
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main(sys.argv)
