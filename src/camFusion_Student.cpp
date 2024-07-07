
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


void
clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor,
                    cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT) {
    
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1) {
        
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; 
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2) {
           
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            
            if (smallerBox.contains(pt)) {
                enclosingBoxes.push_back(it2);
            }

        } 

       
        if (enclosingBoxes.size() == 1) {
            
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } 
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, std::vector<string> param) {
    
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for (auto it1 = boundingBoxes.begin(); it1 != boundingBoxes.end(); ++it1) {
        
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0, 150), rng.uniform(0, 150), rng.uniform(0, 150));

    
        int top = 1e8, left = 1e8, bottom = 0.0, right = 0.0;
        float xwmin = 1e8, ywmin = 1e8, ywmax = -1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2) {
           
            float xw = (*it2).x; 
            float yw = (*it2).y; 
            xwmin = xwmin < xw ? xwmin : xw;
            ywmin = ywmin < yw ? ywmin : yw;
            ywmax = ywmax > yw ? ywmax : yw;

            
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            
            top = top < y ? top : y;
            left = left < x ? left : x;
            bottom = bottom > y ? bottom : y;
            right = right > x ? right : x;

           
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(0, 0, 0), 2);

       
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int) it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left - 250, bottom + 50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax - ywmin);
        putText(topviewImg, str2, cv::Point2f(left - 250, bottom + 125), cv::FONT_ITALIC, 2, currColor);
    }

    float lineSpacing = 2.0; 
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i) {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    

    if (!bWait) {
        cv::waitKey(0); 
    }
}


double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    int ind = static_cast<int>(v.size() / 2);
    return v.size() % 2 != 0 ? v[ind] : (v[ind] + v[ind + 1]) / 2.0;
}

double mean(std::vector<double> v) {
    double sum = 0.0;
    for (auto x : v) sum += x;
    return sum / v.size();
}

void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches) {
    std::vector<double> distances;
    std::vector<cv::DMatch> keypointInRoI;
    for (auto kptMatch: kptMatches) {
        cv::Point pPrevious = kptsPrev[kptMatch.queryIdx].pt;
        cv::Point pCurrent = kptsCurr[kptMatch.trainIdx].pt;
        if (boundingBox.roi.contains(pCurrent)) {
            distances.push_back(cv::norm(pCurrent - pPrevious));
            keypointInRoI.push_back(kptMatch);
        }
    }
    double mu = median(distances);
    auto mnmx = std::minmax_element(distances.begin(), distances.end());
    double range = mnmx.second - mnmx.first;
    for (int i = 0; i < distances.size(); ++i)
        if (fabs(distances[i] - mu) < range * .8)
            boundingBox.kptMatches.push_back(keypointInRoI[i]);
}


// Compute time-to-collision (TTC) 
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg) {
    if(kptMatches.size()==0) return;
    vector<double> distanceRatios; 
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { 
      	
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);

        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { 
          	
            double minDistance = 100.0; 

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            double currDistance = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double prevDistance = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);

            if (prevDistance > std::numeric_limits<double>::epsilon() && currDistance >= minDistance)
            { 
              	
                double distanceRatio = currDistance / prevDistance;
                distanceRatios.push_back(distanceRatio);
            }
        } 
    }     

    if (distanceRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    
    double aveDistanceRatio = std::accumulate(distanceRatios.begin(), distanceRatios.end(), 0.0) / distanceRatios.size();
    double deltaPeriod = 1 / frameRate;
    TTC = -deltaPeriod / (1 - aveDistanceRatio);
}


void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC) {
    vector<double> xPrevious, xCurrent;
    for (auto p : lidarPointsPrev) xPrevious.push_back(p.x);
    for (auto p : lidarPointsCurr) xCurrent.push_back(p.x);
    TTC = (xPrevious.size() && xCurrent.size()) ? median(xCurrent) / (median(xPrevious) - median(xCurrent)) / frameRate : NAN;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame) {
    std::multimap<int, int> IDPrevCurr;
    for (auto match : matches) {
        cv::KeyPoint kpt = prevFrame.keypoints[match.queryIdx];
        int IDPrevious = -1;
        int IDCurrent = -1;
        for (auto &bb : prevFrame.boundingBoxes) if (bb.roi.contains(kpt.pt)) IDPrevious = bb.boxID;
        for (auto &bb : currFrame.boundingBoxes) if (bb.roi.contains(kpt.pt)) IDCurrent = bb.boxID;
        if (IDPrevious != -1 && IDCurrent != -1) IDPrevCurr.emplace(IDPrevious, IDCurrent);
    }

    for (auto const &bb : prevFrame.boundingBoxes) {
        auto range = IDPrevCurr.equal_range(bb.boxID);
        std::map<int, int> ReoccuringIDCurrent;
        for (auto it = range.first; it != range.second; it++) {
            auto subReoccuringID = ReoccuringIDCurrent.find(it->second); 
            if (subReoccuringID != ReoccuringIDCurrent.end()) subReoccuringID->second++;
            else ReoccuringIDCurrent.emplace(std::make_pair(it->second, 1));
        }
        int maxCurrentCount = 0;
        int peakIDCurrent = -1;
        for (auto it = ReoccuringIDCurrent.begin(); it != ReoccuringIDCurrent.end(); it++) {
            if (maxCurrentCount < it->second) {
                peakIDCurrent = it->first;
                maxCurrentCount = it->second;
            }
        }
        if (peakIDCurrent != -1) bbBestMatches.emplace(std::make_pair(bb.boxID, peakIDCurrent));
    }
}
