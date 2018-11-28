#include <iostream>
#include <ctime>
#include <random>

#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;


// This demonstrated connectedComponents() on a hockey image with random colors
int main(){
    cout << "fun2 : OpenCV version = " << CV_VERSION << endl;
    Mat img = imread("../img/bw_170.png", IMREAD_GRAYSCALE);
    if (img.empty())
        exit(13);

    Mat labelImg(img.size(), CV_32S);

    Mat stats, centroids;
//    int nL = connectedComponents(img, labelImg, 8);
    int nL = connectedComponentsWithStats(img, labelImg, stats, centroids, 8);
    cout << "nL = " << nL << endl;

    // Find bounding boxes from components:
    int AREA_THRESH = 100; // Area threshold, ignore too small components
    vector<Rect> bBoxes;
    for (int iL = 1; iL < nL; ++iL) {
        if (stats.at<int>(iL, CC_STAT_AREA) >= AREA_THRESH ) {
            bBoxes.emplace_back(stats.at<int>(iL, CC_STAT_LEFT), stats.at<int>(iL, CC_STAT_TOP),
                                stats.at<int>(iL, CC_STAT_WIDTH), stats.at<int>(iL, CC_STAT_HEIGHT));
        }
    }

    // Random colors
    vector<Vec3b> colors(nL);
    vector<Scalar> colorsS(nL);
    mt19937 randomEngine = mt19937(time(nullptr));
    uniform_int_distribution<uchar> dist(0, 255);
    for (int i = 1; i < nL; ++i) {
        colors[i] = Vec3b(dist(randomEngine), dist(randomEngine), dist(randomEngine));
        colorsS[i] = Scalar(colors[i]);
    }
    colors[0] = Vec3b(0, 0, 0);

    // Output image: colored
    // Probably we don't need this for hockey !
    Mat dstColors(img.size(), CV_8UC3);
    for (int ir = 0; ir < dstColors.rows; ++ir) {
        for (int ic = 0; ic < dstColors.cols; ++ic) {
            int label = labelImg.at<int>(ir, ic);
            Vec3b &pixel = dstColors.at<Vec3b>(ir, ic);
            pixel = colors[label];
        }

    }

    // Output image : bounding boxes + centroids. We need this !
    // Note: label 0 is all black pixels, skip it, real labels start at 1
    Mat dstBB;
    cvtColor(img, dstBB, COLOR_GRAY2BGR);
    for (int iL = 1; iL < nL; ++iL) {
        if (stats.at<int>(iL, CC_STAT_AREA) >= AREA_THRESH ) {
            // Bounding Box
            Rect bbox(stats.at<int>(iL, CC_STAT_LEFT), stats.at<int>(iL, CC_STAT_TOP),
                    stats.at<int>(iL, CC_STAT_WIDTH), stats.at<int>(iL, CC_STAT_HEIGHT));
            rectangle(dstBB, bbox , colorsS[iL]);
            // Centroid
            int xC = (int)centroids.at<double>(iL, 0);
            int yC = (int)centroids.at<double>(iL, 1);
            circle(dstBB, Point(xC, yC), 5, colorsS[iL], -1);
        }
    }


//    cout << "Image size : " << dstColors.cols << " x " << dstColors.rows << " = " << dstColors.total() << endl;
//    for (int iL = 0; iL < 10; ++iL) {
//        cout << iL << " : ";
//        for (int iS = 0; iS < 5; ++iS) {
//            cout << stats.at<int>(iL, iS) << " ";
//        }
//        cout << endl;
//    }

    // Now let's find contours: This is a demo, I think we don't need it for the project!
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    int nC = contours.size();
    vector<vector<Point>> contoursPoly(nC);

    // Get bboxes from contours like in the hockey code
    vector<Rect> bBoxesCont;
    for (const vector<Point> & cont : contours) {
        vector<Point> cPoly;
        approxPolyDP(cont, cPoly, 3, true);
        if (contourArea(cPoly) > AREA_THRESH)
            bBoxesCont.push_back(boundingRect(Mat(cPoly)));
    }

    // Visualize contours
    Mat dstCont;
    cvtColor(img, dstCont, COLOR_GRAY2BGR);
//    for (int iC = 0; iC < nC; ++iC) {
//        drawContours(dstCont, contours, iC, Scalar(dist(randomEngine), dist(randomEngine), dist(randomEngine)));
//    }

    for (const Rect & bbox : bBoxesCont) {
        rectangle(dstCont, bbox, Scalar(dist(randomEngine), dist(randomEngine), dist(randomEngine)) );
    }


//    imshow("img", img);
    imshow("dstColors", dstColors);
    imshow("dstBB", dstBB);
    imshow("dstCont", dstCont);
    waitKey(0);
}
