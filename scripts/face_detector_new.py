#!/usr/bin/env python                                                          
import roslib;
roslib.load_manifest('face_detector')
import sys
import rospy
import cv2
import array
import numpy as np
import std_msgs
from std_msgs.msg import String
from std_msgs.msg import Int32
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from compiler.ast import flatten
#import face_detector.msg 

#faceCascName = ["../config/haarcascade_frontalface_alt2.xml","../config/haarcascade_frontalface_alt.xml","../config/haarcascade_frontalface_default.xml",
#"../config/haarcascade_profileface.xml"]
faceCascName = ["../config/haarcascade_frontalface_alt.xml"]
faceCascade = []
fov = [58,45]
class image_converter:
    def __init__(self):
        for i in range(len(faceCascName)):
            faceCascade.append(cv2.CascadeClassifier(faceCascName[i]))
        self.image_pub = rospy.Publisher("faces",Image)
        self.facenum_pub = rospy.Publisher("faces_number",Int32)
        self.facedir_hori_pub=rospy.Publisher("faces_horizontal_dir",Float64)
        self.facedir_ver_pub=rospy.Publisher("faces_vertical_dir",Float64)

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
        height,width,channels = cv2.image.shape
        faces_hori_dir = [0.0 for i in range(len(faces_new))]
        faces_ver_dir = [0.0 for i in range(len(faces_new))]
        for i in range(len(faces_new)):
            xmin = faces_new[i][0] - 20
            xmax = faces_new[i][0] + faces_new[i][2] + 20
            ymin = faces_new[i][1] - 20
            ymax = faces_new[i][1] + faces_new[i][3] + 20
            #print "xmin,xmax",xmin,xmax
            #boundary check
            if xmin < 0:
                xmin = 0
            if xmax > width:
                xmax = width
            if ymin < 0:
                ymin = 0
            if ymax > height:
                ymax = height
            faces_hori_dir[i] = float(((faces_new[i][0] + faces_new[i][2]) - width/2)*fov[0]/width)
            faces_ver_dir[i] = -float(((faces_new[i][1] + faces_new[i][3]) - height/2)*fov[1]/height)
            print faces_hori_dir[i]
            faces_new[i] = cv2.image[xmin:xmax,ymin:ymax]


        try:
            if len(faces_new) > 0:
                faces_view = faces_new[0]
                cv2.imshow("Faces found",faces_view)
            cv2.waitKey(3)
            for i in range(len(faces_new)):
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(faces_new[i],"bgr8"))
                facedir_ver= std_msgs.msg.Float64(faces_ver_dir[i])
                facedir_hori = std_msgs.msg.Float64(faces_hori_dir[i])
                self.facedir_ver_pub.publish(facedir_ver)
                self.facedir_hori_pub.publish(facedir_hori)

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
