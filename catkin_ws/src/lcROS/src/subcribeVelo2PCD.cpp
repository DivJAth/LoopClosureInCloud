#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <vector>
#include <utility>

//#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/shot.h>
#include <pcl/features/shot_lrf.h>
#include <pcl/features/shot_omp.h>
#include <pcl/features/shot_lrf_omp.h>
#include <pcl/features/board.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/console/parse.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_types.h>


#include "liblzf-3.6/lzf_c.c"
#include "liblzf-3.6/lzf_d.c"

#include <ros/ros.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <boost/foreach.hpp>


#include <stdio.h>
#include <Python.h>
// #include <pyhelper.hpp>


using namespace std;

void usage(const char* program)
{
    cout << "Usage: " << program << " [options] <input.pcd>" << endl << endl;
    cout << "Options: " << endl;
    cout << "--relative If selected, scale is relative to the diameter of the model (-d). Otherwise scale is absolute." << endl;
    cout << "-r R Number of subdivisions in the radial direction. Default 17." << endl;
    cout << "-p P Number of subdivisions in the elevation direction. Default 11." << endl;
    cout << "-a A Number of subdivisions in the azimuth direction. Default 12." << endl;
    cout << "-s S Radius of sphere around each point. Default 1.18 (absolute) or 17\% of diameter (relative)." << endl;
    cout << "-d D Diameter of full model. Must be provided for relative scale." << endl;
    cout << "-m M Smallest radial subdivision. Default 0.1 (absolute) or 1.5\% of diameter (relative)." << endl;
    cout << "-l L Search radius for local reference frame. Default 0.25 (absolute) or 2\% of diameter (relative)." << endl;
    cout << "-t T Number of threads. Default 16." << endl;
    cout << "-o Output file name." << endl;
    cout << "-h Help menu." << endl;
}

vector<vector<double> > compute_intensities(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, 
                                            pcl::PointCloud<pcl::PointNormal>::Ptr normals,
                                            int num_bins_radius, 
                                            int num_bins_polar,
                                            int num_bins_azimuth,
                                            double search_radius,
                                            double lrf_radius, 
                                            double rmin,
                                            int num_threads)
{
    vector<vector<double> > intensities;
    intensities.resize(cloud->points.size());
    
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
    tree->setInputCloud(cloud);

    pcl::PointCloud<pcl::ReferenceFrame>::Ptr frames(new pcl::PointCloud<pcl::ReferenceFrame>());
    pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZ>::Ptr lrf_estimator(new pcl::SHOTLocalReferenceFrameEstimation<pcl::PointXYZ>());
    lrf_estimator->setRadiusSearch(lrf_radius);
    lrf_estimator->setInputCloud(cloud);
    
    lrf_estimator->compute(*frames);

    pcl::StopWatch watch_intensities;

    double ln_rmin = log(rmin);
    double ln_rmax_rmin = log(search_radius/rmin);
    
    double azimuth_interval = 360.0 / num_bins_azimuth;
    double polar_interval = 180.0 / num_bins_polar; 
    vector<double> radii_interval, azimuth_division, polar_division;
    for(int i = 0; i < num_bins_radius+1; i++) {
        radii_interval.push_back(exp(ln_rmin + ((double)i) / num_bins_radius * ln_rmax_rmin));
    }
    for(int i = 0; i < num_bins_azimuth + 1; i++) {
        azimuth_division.push_back(i * azimuth_interval);
    } 
    for(int i = 0; i < num_bins_polar + 1; i++) {
        polar_division.push_back(i * polar_interval);
    } 
    radii_interval[0] = 0;

    vector<double> integr_radius, integr_polar;
    double integr_azimuth;
    for(int i = 0; i < num_bins_radius; i++) {
        integr_radius.push_back((radii_interval[i+1]*radii_interval[i+1]*radii_interval[i+1])/3 - (radii_interval[i]*radii_interval[i]*radii_interval[i])/3 );
    }
    integr_azimuth = pcl::deg2rad(azimuth_division[1]) - pcl::deg2rad(azimuth_division[0]);
    for(int i = 0; i < num_bins_polar; i++) {
        integr_polar.push_back(cos(pcl::deg2rad(polar_division[i]))-cos(pcl::deg2rad(polar_division[i+1])));
    }  


    for(int i = 0; i < cloud->points.size(); i++) {
        vector<int> indices;
        vector<float> distances;
        vector<double> intensity;
        int sum = 0;
        intensity.resize(num_bins_radius * num_bins_polar * num_bins_azimuth);
 
        pcl::ReferenceFrame current_frame = (*frames)[i];
        Eigen::Vector4f current_frame_x (current_frame.x_axis[0], current_frame.x_axis[1], current_frame.x_axis[2], 0);
        Eigen::Vector4f current_frame_y (current_frame.y_axis[0], current_frame.y_axis[1], current_frame.y_axis[2], 0);
        Eigen::Vector4f current_frame_z (current_frame.z_axis[0], current_frame.z_axis[1], current_frame.z_axis[2], 0);

        if(isnan(current_frame_x[0]) || isnan(current_frame_x[1]) || isnan(current_frame_x[2]) ) {
            current_frame_x[0] = 1, current_frame_x[1] = 0, current_frame_x[2] = 0;  
            current_frame_y[0] = 0, current_frame_y[1] = 1, current_frame_y[2] = 0;  
            current_frame_z[0] = 0, current_frame_z[1] = 0, current_frame_z[2] = 1;  
        } else {
            float nx = normals->points[i].normal_x, ny = normals->points[i].normal_y, nz = normals->points[i].normal_z;
            Eigen::Vector4f n(nx, ny, nz, 0);
            if(current_frame_z.dot(n) < 0) {
                current_frame_x = -current_frame_x;
                current_frame_y = -current_frame_y;
                current_frame_z = -current_frame_z;
            }
        }
    
        fill(intensity.begin(), intensity.end(), 0);
        tree->radiusSearch(cloud->points[i], search_radius, indices, distances);
        for(int j = 1; j < indices.size(); j++) {
            if(distances[j] > 1E-15) {
                Eigen::Vector4f v = cloud->points[indices[j]].getVector4fMap() - cloud->points[i].getVector4fMap(); 
                double x_l = (double)v.dot(current_frame_x);
                double y_l = (double)v.dot(current_frame_y);
                double z_l = (double)v.dot(current_frame_z);
                
                double r = sqrt(x_l*x_l + y_l*y_l + z_l*z_l);
                double theta = pcl::rad2deg(acos(z_l / r));
                double phi = pcl::rad2deg(atan2(y_l, x_l));
                int bin_r = int((num_bins_radius - 1) * (log(r) - ln_rmin) / ln_rmax_rmin + 1);
                int bin_theta = int(num_bins_polar * theta / 180);
                int bin_phi = int(num_bins_azimuth * (phi + 180) / 360);

                bin_r = bin_r >= 0 ? bin_r : 0;
                bin_r = bin_r < num_bins_radius ? bin_r : num_bins_radius - 1;
                bin_theta = bin_theta < num_bins_polar ? bin_theta : num_bins_polar - 1;
                bin_phi = bin_phi < num_bins_azimuth ? bin_phi : num_bins_azimuth - 1;
                int idx = bin_r + bin_theta * num_bins_radius + bin_phi * num_bins_radius * num_bins_polar;
                intensity[idx] += 1;
                sum += 1;
            }
        }
        if(sum > 0) {
            for(int j = 0; j < intensity.size(); j++) {
                intensity[j] /= sum;
            }
        }
        intensities[i] = intensity;
    }
    pcl::console::print_highlight("Raw Spherical Histograms Time: %f (s)\n", watch_intensities.getTimeSeconds());
    return intensities;
}

typedef pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloud;

void callback(PointCloud point_cloud_wf)
{
    int num_bins_radius = 17, num_bins_polar = 11, num_bins_azimuth = 12;
    int num_threads = 16;
    double search_radius = 1.18, lrf_radius = 0.25;
    double diameter = 4*sqrt(3);
    double rmin = 0.1;
    //string output_file = "/home/user/Desktop/LoopClosure/SRC/histo_output.lzf";
 
    bool relative_scale = 0>= 0;    

    std::cout << relative_scale << std::endl;
    if(relative_scale) {
        search_radius = 0.17 * diameter;
        lrf_radius = 0.02 * diameter;
        rmin = 0.015 * diameter; 
    }

    // pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud_wf (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointNormal>);

    
    int success = 1;
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud (*point_cloud_wf, *point_cloud_wf,indices);
                
    const float VOXEL_GRID_SIZE = 0.9f;
    pcl::VoxelGrid<pcl::PointXYZ> vox_grid;
    vox_grid.setLeafSize( VOXEL_GRID_SIZE, VOXEL_GRID_SIZE, VOXEL_GRID_SIZE );
    vox_grid.setInputCloud(point_cloud_wf);
    vox_grid.filter(*cloud);


    if(success == -1) {
        PCL_ERROR("Could not read file.");
        return;
    }
    
    cout<<"Calculate Spherical Coordinate"<<endl;

    vector<vector<double> > intensities = compute_intensities(cloud, cloud_with_normals,
                                                              num_bins_radius, num_bins_polar, num_bins_azimuth, 
                                                              search_radius, lrf_radius, 
                                                              rmin, num_threads);
    
    // vector<double> intensities_flat;
	

    cout<<"Convert to C++ vector to PyObject list"<<endl;
    PyObject *listObj = PyList_New(intensities.size()*intensities.size());
    PyObject *pName, *pModule, *pDict, *pFunc;
    PyObject *pArgs, *pValue;
    Py_Initialize();
	if (!listObj) throw logic_error("Unable to allocate memory for Python list");
    // intensities_flat.resize(intensities.size()*intensities.size();
    int cnt = 0;
    
    for(int i = 0; i < intensities.size(); i++) {
        for(int j = 0; j < intensities[i].size(); j++) {
            PyObject *num = PyFloat_FromDouble( (double) intensities[i][j]);
		    if (!num) {
			Py_DECREF(listObj);
			throw logic_error("Unable to allocate memory for Python list");
		    }
            PyList_SET_ITEM(listObj, cnt, num);
            cnt++;
        }
    }
       
    PyRun_SimpleString("import sys\n" "import os\n" "import time\n"  "import lzf\n" "import struct\n"); 
    PyRun_SimpleString("sys.path.append( os.path.dirname(os.getcwd()) +'/catkin_ws/src/lcROS/src/')");

    cout<<"Call to Kafka from C++"<<endl;
    pName = PyUnicode_FromString("kakfa_send_test");
    /* Error checking of pName left out */

    pModule = PyImport_Import(pName);
    Py_DECREF(pName);

    if (pModule != NULL) {
        // here pass the function name
        pFunc = PyObject_GetAttrString(pModule, "mainfunc");
        /* pFunc is a new reference */

        if (pFunc && PyCallable_Check(pFunc)) {
            pArgs = PyTuple_New(1);
            // const char* intensity_char(reinterpret_cast<const char*>(&intensities_compressed[0]));
            PyTuple_SetItem(pArgs, 0, listObj);
            pValue = PyObject_CallObject(pFunc, pArgs);
            Py_DECREF(pArgs);
            if (pValue != NULL) {
                printf("Result of call: %ld\n", PyLong_AsLong(pValue));
                Py_DECREF(pValue);
            }
            else {
                Py_DECREF(pFunc);
                Py_DECREF(pModule);
                PyErr_Print();
                fprintf(stderr,"Call failed\n");
                return;
            }
        }
        else {
            if (PyErr_Occurred())
                PyErr_Print();
            fprintf(stderr, "Cannot find function \n");
        }
        Py_XDECREF(pFunc);
        Py_DECREF(pModule);
    }
    else {
        PyErr_Print();
        fprintf(stderr, "Failed to load \n");
        return;
    }
    Py_DECREF(listObj); 
    Py_Finalize();
   
    return;
}


int main(int argc, char** argv)
{
  cout<<"Subcribe to Kitti/velo/points"<<endl;  
  ros::init(argc, argv, "sub_pcl");
  ros::NodeHandle nh;
  ros::Subscriber sub = nh.subscribe<PointCloud>("/kitti/velo/pointcloud", 1, callback);
  ros::spin();
 }


