# face_detector
  
## outline  
Ros node for detecting face robust.  
Also you can count the number of the faces robustly.  
This software uses the following pipeline to detect face robust:  
1.Detect faces with 4 different learning data of HaarCascade  
                             |  
                             V  
2.Cluster the face windows position(x,y) by Flat Clustering(Hierarchial clustering)  
                             |  
                             V  
3.Filter the noise which are detected as faces by looking the element of the cluster    

##software requirements  
ROS Indigo  
Ubuntu 14.04  
opencv2(tested in opencv 2.4.8 which the version is supported in ubuntu 14.04 LTS mainstream package)  
numpy  
python-opencv2  
scipy  
matplotlib  
cv_bridge  
image_transport  
sensor_msgs  
std_msgs  
rospy  
gfortran
liblapack-dev
libblas-dev

#installation    
1.cd /path/to/ros_ws/src && git clone https://github.com/demulab/face_detector.git  
2.cd /path/to/ros_ws/src/face_detector/scripts  
3.chmod +x dep.bash  
4.sudo ./dep.bash  
5.sed -i s//image_raw//your/image_topic/image_raw/g face_detector_main.py  
6.cd /path/to/ros_ws  
7.catkin_make  

#running sample    
in webcam  
1. roscore  
2. rosrun  uvc_camera uvc_camera_node  
3. rosrun face_detector face_detector.py  
in xtion  
1. roslaunch openni2_launch openni2.launch  
3. rosrun face_detector face_detector.py  

#topic information  
/faces : image of faces(sensor_msgs/Image type)  
/faces_number : number of faces  

  
  

  