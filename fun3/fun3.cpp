#include <iostream>
#include <ctime>
#include <random>
#include <chrono>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace std::chrono;
using namespace cv;


// This demonstrated connectedComponents() on a hockey image with random colors
int main() {
    cout << "FUN3: OpenCV version = " << CV_VERSION << endl;
    Mat img = imread("../img/bw_170.png", IMREAD_GRAYSCALE);
    if (img.empty())
        exit(13);

    Mat labelImg(img.size(), CV_32S);

    // Parameters
    int AREA_THRESH = 100; // Area threshold, ignore too small components
    int NUM_RUNS = 10000; // How many times do we run things ?

    Mat stats, centroids;
    vector<Rect> bBoxes;
    bBoxes.reserve(100);

    using DSeconds = duration<double>;

    // Time connectedComponentsWithStats()
    {
        cout << "Starting connectedComponentsWithStats() ... " << endl;
        auto t1 = high_resolution_clock::now();
        for (int n = 0; n < NUM_RUNS; ++n) {
            int nL = connectedComponentsWithStats(img, labelImg, stats, centroids, 8);
            bBoxes.clear();
            for (int iL = 1; iL < nL; ++iL) {
                if (stats.at<int>(iL, CC_STAT_AREA) >= AREA_THRESH) {
                    bBoxes.emplace_back(stats.at<int>(iL, CC_STAT_LEFT), stats.at<int>(iL, CC_STAT_TOP),
                                        stats.at<int>(iL, CC_STAT_WIDTH), stats.at<int>(iL, CC_STAT_HEIGHT));
                }
            }
        }
        auto t2 = high_resolution_clock::now();
        cout << "Finished connectedComponentsWithStats() : " << DSeconds(t2 - t1).count() << " seconds" << endl;
    }

    vector<Rect> bBoxesCont;
    bBoxesCont.reserve(100);
    // Time findContours() and stuff
    {
        cout << "Starting findContours() ... " << endl;
        auto t1 = high_resolution_clock::now();
        for (int n = 0; n < NUM_RUNS; ++n) {
            vector<vector<Point>> contours;
            vector<Vec4i> hierarchy;
            findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            bBoxesCont.clear();
            for (const vector<Point> & cont : contours) {
                vector<Point> cPoly;
                approxPolyDP(cont, cPoly, 3, true);
                if (contourArea(cPoly) > AREA_THRESH)
                    bBoxesCont.push_back(boundingRect(Mat(cPoly)));
            }
        }
        auto t2 = high_resolution_clock::now();
        cout << "Finished findContours() : " << DSeconds(t2 - t1).count() << " seconds" << endl;
    }

    // -----Visualize-----
    mt19937 randomEngine = mt19937(time(nullptr));
    uniform_int_distribution<uchar> dist(0, 255);

    // Output image : Components
    // Note: label 0 is all black pixels, skip it, real labels start at 1
    Mat dstComp;
    cvtColor(img, dstComp, COLOR_GRAY2BGR);
    for (const Rect & bbox : bBoxes) {
        rectangle(dstComp, bbox, Scalar(dist(randomEngine), dist(randomEngine), dist(randomEngine)) );
    }

    // Contours
    Mat dstCont;
    cvtColor(img, dstCont, COLOR_GRAY2BGR);
    for (const Rect & bbox : bBoxesCont) {
        rectangle(dstCont, bbox, Scalar(dist(randomEngine), dist(randomEngine), dist(randomEngine)) );
    }


//    imshow("img", img);
    imshow("dstComp", dstComp);
    imshow("dstCont", dstCont);
    waitKey(0);
}
