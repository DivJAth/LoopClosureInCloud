#!/usr/bin/env python
import rospy
# from geometry_msgs.msg import Twist
# from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointCloud
# from nav_msgs.msg import Odometry

import pcl
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
# import ros_numpy
import numpy as np
def scan_callback(data):
    pc = ros_numpy.array(data)
    print(pc.shape)
    points=np.zeros((pc.shape[0],3))
    points[:,0]=pc['x']
    points[:,1]=pc['y']
    points[:,2]=pc['z']
    p = pcl.PointCloud(np.array(points, dtype=np.float32))
# 
# rospy.init_node('listener', anonymous=True)
# rospy.Subscriber("/velodyne_points", PointCloud2, callback)
# rospy.spin()

# def scan_callback(msg):
    # print(msg)
    

def listen():
    rospy.init_node ('Subcribe_KIITI_Velo', anonymous=True, log_level=rospy.DEBUG)
    scan_topic = '/kitti/velo/pointcloud'
    # scan_topic = 'PointCloud'
    scan_sub = rospy.Subscriber(scan_topic, PointCloud2, scan_callback, queue_size=1)
    rospy.loginfo("Subcribe to the topic %s", scan_topic)
    rospy.spin()
    # r = rospy.Rate(4)




if __name__ == '__main__':
    try:
        listen()
    except rospy.ROSInterruptException:
        pass



# import rospy
# import pcl
# from sensor_msgs.msg import PointCloud2
# import sensor_msgs.point_cloud2 as pc2
# import ros_numpy

# def callback(data):
#     pc = ros_numpy.numpify(data)
#     points=np.zeros((pc.shape[0],3))
#     points[:,0]=pc['x']
#     points[:,1]=pc['y']
#     points[:,2]=pc['z']
#     p = pcl.PointCloud(np.array(points, dtype=np.float32))

# rospy.init_node('listener', anonymous=True)
# rospy.Subscriber("/velodyne_points", PointCloud2, callback)
# rospy.spin()