// mostra_cv.cpp
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat a = imread("../imgs/lenna.jpg", 1);
    imshow("window", a);
    waitKey(0);
}