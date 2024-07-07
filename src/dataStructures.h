
#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <map>
#include <opencv2/core.hpp>

struct LidarPoint { 
    double x,y,z,r;
};

struct BoundingBox { 
    
    int boxID; 
    int trackID;
    
    cv::Rect roi; 
    int classID; 
    double confidence; 

    std::vector<LidarPoint> lidarPoints; 
    std::vector<cv::KeyPoint> keypoints; 
    std::vector<cv::DMatch> kptMatches; 
};

struct DataFrame { 
    
    cv::Mat cameraImg; 
    
    std::vector<cv::KeyPoint> keypoints; 
    cv::Mat descriptors; 
    std::vector<cv::DMatch> kptMatches;
    std::vector<LidarPoint> lidarPoints;

    std::vector<BoundingBox> boundingBoxes; 
    std::map<int,int> bbMatches; 
};

#endif /* dataStructures_h */
