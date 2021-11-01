# Loop Closure Using Cloud Technologies

## Robotic System & Cloud Infrastructure
* Tasks in robotics have ever changing resource requirements.
* Cloud computing allows for flexible allocation of compute and memory,  facilitate real-time deployment.

## Experiment
* Detect loop closure on Kitti Dataset.
* Loop Closure is the detection on when an object i.e. a car or robot has returned to a previously mapped location. 
* This has been implemented using  the algorithm from Learning Compact Geometric Features by Khoury et al.

## Overview: Learning Compact Geometric Features 
* Local geometry of the neighborhood is captured using spherical histogram centered around each point.
* Feature embedding is used to map the high dimensionality of the spherical histogram to a low dimension feature space.
* Correspondences: This reduces to performing nearest neighborhood query

## COMPONENTS
* Robot Operating System(ROS), A peer-to peer distributed system.
* Zookeeper, cluster management
* Kafka, messaging
* Spark Streaming, compute/process engine 
* Redis, database
* Kubernetes(pending)

## Architecture
![Integration of ROS with CLoud Technologies](https://github.com/arpg/cloud_closure/blob/master/IMG/LoopClosure.jpg)



