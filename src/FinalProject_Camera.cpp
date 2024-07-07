/* INCLUDES FOR THIS PROJECT */
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <limits>
#include <opencv2/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

#include "dataStructures.h"
#include "matching2D.hpp"
#include "objectDetection2D.hpp"
#include "lidarData.hpp"
#include "camFusion.hpp"

using namespace std;

/* MAIN PROGRAM */
int main(int argc, const char *argv[]) {
   
    string dataPath = "../";

    // camera
    string imgBasePath = dataPath + "images/";
    string imgPrefix = "KITTI/2011_09_26/image_02/data/000000";
    string imgFileType = ".png";
    int imgStartIndex = 0; 
    int imgEndIndex = 77;  
    int imgStepWidth = 1;
    int imgFillWidth = 4; 

    // object detection
    string yoloBasePath = dataPath + "dat/yolo/";
    string yoloClassesFile = yoloBasePath + "coco.names";
    string yoloModelConfiguration = yoloBasePath + "yolov3.cfg";
    string yoloModelWeights = yoloBasePath + "yolov3.weights";

    // Lidar
    string lidarPrefix = "KITTI/2011_09_26/velodyne_points/data/000000";
    string lidarFileType = ".bin";

    // calibration data for camera and lidar
    cv::Mat P_rect_00(3, 4, cv::DataType<double>::type); 
    cv::Mat R_rect_00(4, 4, cv::DataType<double>::type); 
    cv::Mat RT(4, 4, cv::DataType<double>::type); 

    RT.at<double>(0, 0) = 7.533745e-03; RT.at<double>(0, 1) = -9.999714e-01; RT.at<double>(0, 2) = -6.166020e-04; RT.at<double>(0, 3) = -4.069766e-03;
    RT.at<double>(1, 0) = 1.480249e-02; RT.at<double>(1, 1) = 7.280733e-04; RT.at<double>(1, 2) = -9.998902e-01; RT.at<double>(1, 3) = -7.631618e-02;
    RT.at<double>(2, 0) = 9.998621e-01; RT.at<double>(2, 1) = 7.523790e-03; RT.at<double>(2, 2) = 1.480755e-02; RT.at<double>(2, 3) = -2.717806e-01;
    RT.at<double>(3, 0) = 0.0; RT.at<double>(3, 1) = 0.0; RT.at<double>(3, 2) = 0.0; RT.at<double>(3, 3) = 1.0;

    R_rect_00.at<double>(0, 0) = 9.999239e-01; R_rect_00.at<double>(0, 1) = 9.837760e-03; R_rect_00.at<double>(0, 2) = -7.445048e-03; R_rect_00.at<double>(0, 3) = 0.0;
    R_rect_00.at<double>(1, 0) = -9.869795e-03; R_rect_00.at<double>(1, 1) = 9.999421e-01; R_rect_00.at<double>(1, 2) = -4.278459e-03; R_rect_00.at<double>(1, 3) = 0.0;
    R_rect_00.at<double>(2, 0) = 7.402527e-03; R_rect_00.at<double>(2, 1) = 4.351614e-03; R_rect_00.at<double>(2, 2) = 9.999631e-01; R_rect_00.at<double>(2, 3) = 0.0;
    R_rect_00.at<double>(3, 0) = 0; R_rect_00.at<double>(3, 1) = 0; R_rect_00.at<double>(3, 2) = 0; R_rect_00.at<double>(3, 3) = 1; P_rect_00.at<double>(0, 0) = 7.215377e+02;

    P_rect_00.at<double>(0, 1) = 0.000000e+00; P_rect_00.at<double>(0, 2) = 6.095593e+02; P_rect_00.at<double>(0, 3) = 0.000000e+00;
    P_rect_00.at<double>(1, 0) = 0.000000e+00; P_rect_00.at<double>(1, 1) = 7.215377e+02; P_rect_00.at<double>(1, 2) = 1.728540e+02;
    P_rect_00.at<double>(1, 3) = 0.000000e+00; P_rect_00.at<double>(2, 0) = 0.000000e+00; P_rect_00.at<double>(2, 1) = 0.000000e+00;
    P_rect_00.at<double>(2, 2) = 1.000000e+00; P_rect_00.at<double>(2, 3) = 0.000000e+00;

    
    double sensorFrameRate = 10.0 / imgStepWidth; 
    int dataBufferSize = 2;      
    vector<DataFrame> dataBuffer; 
    bool bVis = false;            

    string detectorType = "AKAZE";
    string descriptorType = "AKAZE"; 
    cout << detectorType << "+" << descriptorType << endl;
    double diffCameraLiDAR = 0.0;
    int abnormalTTCCount = 0;
    
    for (size_t imgIndex = 0; imgIndex <= imgEndIndex - imgStartIndex; imgIndex += imgStepWidth) {

        ostringstream imgNumber;
        imgNumber << setfill('0') << setw(imgFillWidth) << imgStartIndex + imgIndex;
        string imgFullFilename = imgBasePath + imgPrefix + imgNumber.str() + imgFileType;

        // load image from file
        cv::Mat img = cv::imread(imgFullFilename);

       
        DataFrame frame;
        frame.cameraImg = img;
        dataBuffer.push_back(frame);

        /* DETECT & CLASSIFY OBJECTS */

        float confThreshold = 0.2;
        float nmsThreshold = 0.4;
        detectObjects((dataBuffer.end() - 1)->cameraImg, (dataBuffer.end() - 1)->boundingBoxes, confThreshold,
                      nmsThreshold,
                      yoloBasePath, yoloClassesFile, yoloModelConfiguration, yoloModelWeights, bVis);

        // load 3D Lidar points from file
        string lidarFullFilename = imgBasePath + lidarPrefix + imgNumber.str() + lidarFileType;
        std::vector<LidarPoint> lidarPoints;
        loadLidarFromFile(lidarPoints, lidarFullFilename);

      
        float minZ = -1.5, maxZ = -0.9, minX = 2.0, maxX = 20.0, maxY = 2.0, minR = 0.1; 
        cropLidarPoints(lidarPoints, minX, maxX, maxY, minZ, maxZ, minR);

        (dataBuffer.end() - 1)->lidarPoints = lidarPoints;

       
        float shrinkFactor = 0.10; 
        clusterLidarWithROI((dataBuffer.end() - 1)->boundingBoxes, (dataBuffer.end() - 1)->lidarPoints, shrinkFactor,
                            P_rect_00, R_rect_00, RT);

         bVis = true;
        if (bVis) {
            show3DObjects((dataBuffer.end() - 1)->boundingBoxes, cv::Size(4.0, 20.0), cv::Size(2000, 2000), true, {detectorType, descriptorType, imgNumber.str()});
        }
        bVis = false;

       
        // convert current image to grayscale
        cv::Mat imgGray;
        cv::cvtColor((dataBuffer.end() - 1)->cameraImg, imgGray, cv::COLOR_BGR2GRAY);

        // extract 2D keypoints from current image
        vector<cv::KeyPoint> keypoints;
        if (detectorType.compare("SHITOMASI") == 0) {
            detKeypointsShiTomasi(keypoints, imgGray, false);
        } else if (detectorType.compare("HARRIS") == 0) {
            detKeypointsHarris(keypoints, imgGray, bVis);
        } else {
            detKeypointsModern(keypoints, imgGray, detectorType, bVis);
        }

        
        bool bLimitKpts = false;
        if (bLimitKpts) {
            int maxKeypoints = 50;

            if (detectorType.compare("SHITOMASI") ==
                0) { 
                keypoints.erase(keypoints.begin() + maxKeypoints, keypoints.end());
            }
            cv::KeyPointsFilter::retainBest(keypoints, maxKeypoints);
            cout << " NOTE: Keypoints have been limited!" << endl;
        }

        (dataBuffer.end() - 1)->keypoints = keypoints;


        cv::Mat descriptors;

        descKeypoints((dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->cameraImg, descriptors,
                      descriptorType);

        (dataBuffer.end() - 1)->descriptors = descriptors;

        
        if (dataBuffer.size() > 1) 
        {
           
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";      
            string descriptorTypeCategory = "DES_HOG"; 
            string selectorType = "SEL_NN";     

            matchDescriptors((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints,
                             (dataBuffer.end() - 2)->descriptors, (dataBuffer.end() - 1)->descriptors,
                             matches, descriptorTypeCategory, matcherType, selectorType);

           
            (dataBuffer.end() - 1)->kptMatches = matches;

            /map<int, int> bbBestMatches;
            matchBoundingBoxes(matches, bbBestMatches, *(dataBuffer.end() - 2), *(dataBuffer.end() - 1)); 
          
            (dataBuffer.end() - 1)->bbMatches = bbBestMatches;

            
            for (auto it1 = (dataBuffer.end() - 1)->bbMatches.begin();
                 it1 != (dataBuffer.end() - 1)->bbMatches.end(); ++it1) {
                
                BoundingBox *prevBB, *currBB;
                for (auto it2 = (dataBuffer.end() - 1)->boundingBoxes.begin();
                     it2 != (dataBuffer.end() - 1)->boundingBoxes.end(); ++it2) {
                    if (it1->second == it2->boxID) 
                    {
                        currBB = &(*it2);
                    }
                }

                for (auto it2 = (dataBuffer.end() - 2)->boundingBoxes.begin();
                     it2 != (dataBuffer.end() - 2)->boundingBoxes.end(); ++it2) {
                    if (it1->first == it2->boxID)
                    {
                        prevBB = &(*it2);
                    }
                }

               
                if (currBB->lidarPoints.size() > 0 &&
                    prevBB->lidarPoints.size() > 0) 
                {
                   )
                    double TTCLiDAR;
                    computeTTCLidar(prevBB->lidarPoints, currBB->lidarPoints, sensorFrameRate, TTCLiDAR);
                    
                    double TTCCamera;
                    clusterKptMatchesWithROI(*currBB, (dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, (dataBuffer.end() - 1)->kptMatches);
                    computeTTCCamera((dataBuffer.end() - 2)->keypoints, (dataBuffer.end() - 1)->keypoints, currBB->kptMatches, sensorFrameRate, TTCCamera);
                   
                    bVis = true;
                    if (bVis) {
                        cv::Mat visImg = (dataBuffer.end() - 1)->cameraImg.clone();
                        showLidarImgOverlay(visImg, currBB->lidarPoints, P_rect_00, R_rect_00, RT, &visImg);
                        cv::rectangle(visImg, cv::Point(currBB->roi.x, currBB->roi.y),
                                      cv::Point(currBB->roi.x + currBB->roi.width, currBB->roi.y + currBB->roi.height),
                                      cv::Scalar(0, 255, 0), 2);

                        char str[200];
                        if (isnormal(fabs(TTCLiDAR - TTCCamera)))
                            diffCameraLiDAR += fabs(TTCLiDAR - TTCCamera);
                        else
                        {
                            cout << "#"<<imgNumber.str()<<" ";
                        }

                        sprintf(str, "TTC Lidar : %f s, TTC Camera : %f s", TTCLiDAR, TTCCamera);
                        putText(visImg, str, cv::Point2f(80, 50), cv::FONT_HERSHEY_PLAIN, 1.5, cv::Scalar(0, 0, 255));

                        string windowName = "Final Results : TTC";
                        cv::namedWindow(windowName, 4);
                        cv::imshow(windowName, visImg);
                        cv::imwrite(imgBasePath + "TTC/"+detectorType+"_"+descriptorType+"_"+imgNumber.str()+imgFileType, visImg);
                        cout << "Press key to continue to next frame" << endl;
                        cv::waitKey(0);
                    }
                    bVis = false;
                } 
            } 
        }
    } 
    cout << endl << diffCameraLiDAR << " average: " << diffCameraLiDAR/(imgEndIndex-abnormalTTCCount) << endl;

    return 0;
}
