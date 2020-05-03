#include <numeric>
#include "matching2D.hpp"

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    std::vector<std::vector<cv::DMatch>> matches_vect;
    float distanceThresh = 0.8;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType;
        // SIFT is not a binary descriptor
        if (descriptorType.compare("DES_HOG") == 0)
            normType = cv::NORM_L2;
        else
            normType = cv::NORM_HAMMING;
             
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        matcher = cv::FlannBasedMatcher::create();

        descSource.convertTo(descSource, CV_32F);
        descRef.convertTo(descRef, CV_32F);
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        matcher->knnMatch(descSource, descRef, matches_vect, 2);

        // filter matches
        for (int i = 0; i < matches_vect.size(); i++)
        {
            if(matches_vect[i][0].distance < distanceThresh * matches_vect[i][1].distance)
                matches.push_back(matches_vect[i][0]);
        }
        //cout << "found " << matches.size() << " matches";
    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
double descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if(descriptorType.compare("BRIEF") == 0)
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
    else if(descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create();
    }
    else if(descriptorType.compare("FREAK") == 0)
    {
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if(descriptorType.compare("AKAZE") == 0)
    {
        extractor = cv::AKAZE::create();
    }
    else if(descriptorType.compare("SIFT") == 0)
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }

    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double exec_time = 1000 * t / 1.0 ;
    //cout << descriptorType << " descriptor extraction in " << exec_time << " ms" << endl;

    return exec_time;
}

 double detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
 {
     // create pointer to a generic feature detector
     cv::Ptr<cv::FeatureDetector> detector;

    double t = (double)cv::getTickCount();

     if (detectorType.compare("FAST")==0)
     {
         // detector = cv::FeatureDetector::create("FAST"); // apparently deprecated
         detector = cv::FastFeatureDetector::create();
     }
     else if (detectorType.compare("BRISK")==0)
     {
         detector = cv::BRISK::create();
     }
     else if (detectorType.compare("ORB")==0)
     {
         detector = cv::ORB::create();
     }
     else if (detectorType.compare("AKAZE")==0)
     {
         detector = cv::AKAZE::create();
     }
     else if (detectorType.compare("SIFT")==0)
     {
         detector = cv::xfeatures2d::SIFT::create();
     }

     detector->detect(img, keypoints);

     t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
     double exec_time = 1000 * t / 1.0 ;
    //cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << exec_time << " ms" << endl;

    return exec_time;
 }

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    int blockSize = 2;
    int apertureSize = 3;
    double k = 0.04;
    int threshold = 100;
    double maxOverlap = 0.0;

    double t = (double)cv::getTickCount();

    cv::Mat result = cv::Mat::zeros(img.size(), CV_32FC1);
    cv::Mat norm, norm_conv;
    cv::cornerHarris(img, result, blockSize, apertureSize, k);
    cv::normalize(result, norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    cv::convertScaleAbs(norm, norm_conv);

    //cv::namedWindow("norm_conv", 6);
    //cv::imshow("norm_conv", norm_conv);
    //cv::waitKey(0);

    //cout << "norm_conv.rows: " << norm_conv.rows << " norm_conv.cols: " << norm_conv.cols << " norm_conv content: " << (int)norm_conv.at<unsigned char>(5,5) << " type of mat: " << norm_conv.type() << endl; // typeid(norm_conv<unsigned int>.at(5,5)).name()

    for(int i = 0; i < norm_conv.rows; i++)
    {
        for(int j = 0; j < norm_conv.cols; j++)
        {
            if ((int)norm_conv.at<unsigned char>(i,j) > threshold)
            {
                cv::KeyPoint keypoint;
                keypoint.pt = cv::Point2f(j,i);
                keypoint.response = norm_conv.at<unsigned char>(i,j);
                keypoint.size = 2*apertureSize;
                //cout << "found keypoint at i=" << i << " j=" << j << endl;

                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); it++)
                {
                    if (cv::KeyPoint::overlap(*it, keypoint) > maxOverlap)
                    {
                        bOverlap = true;
                        if ( (*it).response < keypoint.response )
                            (*it) = keypoint;
                        //cout << "overlapped keypoint" << endl;
                        break;
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(keypoint);
                    //cout << "found keypoint at i=" << i << " j=" << j << endl;
                }
            }

        }
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double exec_time = 1000 * t / 1.0;
    //cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << exec_time << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return exec_time;
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
double detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    double exec_time = 1000 * t / 1.0;
    //cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << exec_time << " ms" << endl;

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

    return exec_time;
}