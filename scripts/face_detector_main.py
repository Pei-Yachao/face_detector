#!/usr/bin/env python                                                          
import roslib
import sys
import rospy
import cv2
import array
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from compiler.ast import flatten
import matplotlib.pyplot as plt
import numpy
import scipy.cluster.hierarchy as hcluster

faceCascName = ["../config/haarcascade_frontalface_alt2.xml","../config/haarcascade_frontalface_alt.xml","../config/haarcascade_frontalface_default.xml",
"../config/haarcascade_profileface.xml"]
faceCascade = []
for i in range(len(faceCascName)):
    faceCascade.append(cv2.CascadeClassifier(faceCascName[i]))

class image_converter:
    def __init__(self):
        self.image_pub = rospy.Publisher("image_topic",Image)
        cv2.namedWindow("Image window",1)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/image_raw",Image,self.callback)

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
        #store x,y location 
        face_clusterdata = np.zeros((len(faces_new),2))
        for i in range(len(faces_new)):
            face_clusterdata[i] = np.array([faces_new[i][0], faces_new[i][1]])
        thresh = 20
        clusters = hcluster.fclusterdata(face_clusterdata, thresh, criterion = "distance")
        cln = np.zeros((len(clusters),1))
        for i in range(len(clusters)):
            for j in range(len(clusters)):
                if clusters[i] == clusters[j]:
                    cln[i] += 1
        loss = 0
        loss_index = []
        for i in range(len(cln)):
            if cln[i] == 1:
                loss += 1
                loss_index.append(i)
        print "cln:",cln
        print "face number:",len(set(clusters)) -loss
        print "cluster:",set(clusters)
        print "cluster2:",clusters
        #plt.scatter(*numpy.transpose(face_clusterdata), c=clusters)
        #plt.axis("equal")
        #title = "threshold: %f, number of clusters: %d" % (thresh, len(set(clusters)))
        #plt.title(title)
        #plt.show()
        #draw face detected rectangle
        #for face in faces:
        #    for (x, y, w, h) in face:
        #        cv2.rectangle(cv2.image, (x, y), (x+w, y+h), (0, 255, 0), 2)
       
        #show the detected faces
        cv2.imshow("Faces found",cv2.image)
        cv2.waitKey(3)
        #publish faces
        noise_flag = False
        cl_flag = [False for i in range(len(set(clusters)))]
        face_categ = [0 for i in range(len(set(clusters)))]
        for i in range(len(faces_new)):
            for j in range(len(loss_index)):
                if i == loss_index[j]:
                    noise_flag = True
            if noise_flag:
                continue
            elif cl_flag[clusters[i]-1]:
                continue
            else:
                face_categ[clusters[i]-1] = cv2.image[faces_new[i][0]:(faces_new[i][0]+faces_new[i][2]),faces_new[i][1]:(faces_new[i][1]+faces_new[i][3])]
                print face_categ[clusters[i]-1]
                cl_flag[clusters[i]-1] = True


        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(face_categ[0],"bgr8"))
        except CvBridgeError, e:
            print e

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
