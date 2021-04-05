// https://stackoverflow.com/questions/41614319/accessing-rgb-values-of-all-pixels-in-a-certain-image-in-opencv

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){

    Mat3b a = imread("../imgs/lenna.jpg");


    for(int l=0; l<a.rows; l++)
        for(int c=0; c<a.cols; c++){  //BGR
            a(l,c)[0] = 255 - a(l,c)[0]; // blue
            a(l,c)[1] = 255 - a(l,c)[1]; // green
            a(l,c)[2] = 255 - a(l,c)[2]; // red
        }
            
    imwrite("./inverted-lenna.jpg", a);
}