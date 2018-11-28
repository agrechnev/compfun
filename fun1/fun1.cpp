#include <iostream>
#include <opencv2/opencv.hpp>


using namespace std;
using namespace cv;

int main(){
    cout << "OpenCV version = " << CV_VERSION << endl;
    Mat img = imread("../img/stuff.jpg", IMREAD_GRAYSCALE);
    if (img.empty())
        exit(13);

    int t = 70;
    Mat bw = img < t;
    Mat labelImg(img.size(), CV_32S);
    int nL = connectedComponents(bw, labelImg, 8);
    cout << "nL = " << nL << endl;

    // Random colors
    vector<Vec3b> colors(nL);
    for (int i = 1; i < nL; ++i) {
        colors[i] = Vec3b((rand()&255), (rand()&255), (rand()&255));
    }
    colors[0] = Vec3b(0, 0, 0);

    // Output image
    Mat dst(img.size(), CV_8UC3);
    for (int ir = 0; ir < dst.rows; ++ir) {
        for (int ic = 0; ic < dst.cols; ++ic) {
            int label = labelImg.at<int>(ir, ic);
            Vec3b &pixel = dst.at<Vec3b>(ir, ic);
            pixel = colors[label];
        }

    }

    imshow("img", img);
    imshow("bw", bw);
    imshow("dst", dst);
    waitKey(0);
}
