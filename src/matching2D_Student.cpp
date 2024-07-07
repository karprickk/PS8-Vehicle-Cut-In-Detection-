
#include <numeric>
#include "matching2D.hpp"

using namespace std;


void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource,
                      cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType,
                      std::string selectorType) {
    
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int normType;
    if (descriptorType.compare("DES_BINARY")) normType = cv::NORM_HAMMING;
    else if (descriptorType.compare("DES_HOG")) normType = cv::NORM_L2;
    else throw invalid_argument("Unknown descriptorType " + descriptorType);

    if (matcherType.compare("MAT_BF") == 0) matcher = cv::BFMatcher::create(normType, crossCheck);
    else if (matcherType.compare("MAT_FLANN") == 0) {
        if (normType == cv::NORM_HAMMING) {
            const cv::Ptr<cv::flann::IndexParams> &indexParams = cv::makePtr<cv::flann::LshIndexParams>(12, 20, 2);
            matcher = cv::makePtr<cv::FlannBasedMatcher>(indexParams);
        } else matcher = cv::FlannBasedMatcher::create();
    } else throw invalid_argument("Unknown matcherType " + matcherType);

    
    if (selectorType.compare("SEL_NN") == 0) { 

        matcher->match(descSource, descRef, matches); 
    } else if (selectorType.compare("SEL_KNN") == 0) {
        int k = 2;
        vector<vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descSource, descRef, knn_matches, k);
        
        double minDistanceRatioDescriptor = 0.8;
        for (auto matchPair : knn_matches) {
            if (matchPair[0].distance < minDistanceRatioDescriptor * matchPair[1].distance) {
                matches.push_back(matchPair[0]);
            }
        }
    }
}


void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType) {
    
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0) {
        int threshold = 30;        
        int octaves = 3;           
        float patternScale = 1.0f; 

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    } else if (descriptorType.compare("BRIEF") == 0) extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    else if (descriptorType.compare("ORB") == 0) extractor = cv::ORB::create();
    else if (descriptorType.compare("FREAK") == 0) extractor = cv::xfeatures2d::FREAK::create();
    else if (descriptorType.compare("AKAZE") == 0) extractor = cv::AKAZE::create();
    else if (descriptorType.compare("SIFT") == 0) extractor = cv::xfeatures2d::SIFT::create();
    else throw invalid_argument("Unknown descriptorType" + descriptorType);

   
    double t = (double) cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    
}

void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
    
    int blockSize = 4;      
    double maxOverlap = 0.0; 
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); 

    double resolution = 0.01; 
    double k = 0.04;

   
    double t = (double) cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, resolution, minDistance, cv::Mat(), blockSize, false, k);

    
    for (auto it = corners.begin(); it != corners.end(); ++it) {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
    
    if (bVis) {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis) {
   
    int blockSize = 2;     
    int apertureSize = 3; 
    assert(1 == apertureSize % 2);  
    int minResponse = 100;
    double k = 0.04;       


    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(dst_norm, dst_norm_scaled);

    if (bVis) {
       
        string windowName = "Harris Corner Detector Response Matrix";
        cv::namedWindow(windowName, 4);
        cv::imshow(windowName, dst_norm_scaled);
        cv::waitKey(0);
    }
    
    double maxOverlap = 0.0; 
    for (size_t j = 0; j < dst_norm.rows; j++) {
        for (size_t i = 0; i < dst_norm.cols; i++) {
            int response = (int) dst_norm.at<float>(j, i);
            if (response > minResponse) { 

                cv::KeyPoint newKeyPoint;
                newKeyPoint.pt = cv::Point2f(i, j);
                newKeyPoint.size = 2 * apertureSize;
                newKeyPoint.response = response;

                
                bool boundaryOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it) {
                    double keypointOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                    if (keypointOverlap > maxOverlap) {
                        boundaryOverlap = true;
                        if (newKeyPoint.response >
                            (*it).response) {                      
                            *it = newKeyPoint; 
                            break;             
                        }
                    }
                }
                if (!boundaryOverlap) {                                     
                    keypoints.push_back(newKeyPoint); 
                }
            }
        } 

    if (bVis) {
        
        string windowName = "Harris Corner Detection Results";
        cv::namedWindow(windowName, 5);
        cv::Mat visImage = dst_norm_scaled.clone();
        cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}


void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis) {
    cv::Ptr<cv::Feature2D> detector;
    if (detectorType.compare("FAST") == 0) detector = cv::FastFeatureDetector::create();
    else if (detectorType.compare("BRISK") == 0) detector = cv::BRISK::create();
    else if (detectorType.compare("ORB") == 0) detector = cv::ORB::create();
    else if (detectorType.compare("AKAZE") == 0) detector = cv::AKAZE::create();
    else if (detectorType.compare("SIFT") == 0) detector = cv::xfeatures2d::SIFT::create();
    else throw invalid_argument(detectorType + " unsupported detectorType");
    detector->detect(img, keypoints);
    if (bVis) {
       
        string windowName = detectorType + " Keypoint Detection Results";
        cv::namedWindow(windowName);
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        cv::imshow(windowName, visImage);
        cv::waitKey(0);
    }
}
