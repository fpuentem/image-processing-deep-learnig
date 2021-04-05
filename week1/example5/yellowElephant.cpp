
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){

    Mat3b a = imread("../imgs/elefante.jpg");
    Vec3b yellow(20, 200, 200);
    Mat3b b;

    b = a.clone();

    for(int l=0; l<a.rows; l++){
        for(int c=0; c<a.cols; c++){  //BGR

            double dist = norm(yellow, a(l, c), NORM_L2);
            
            if(dist < 70.0){
                b(l,c)[0] = 255;  // blue
                b(l,c)[1] = 255;  // green
                b(l,c)[2] = 255;  // red
            }
                
        }
    }
            
    imwrite("./white-elephant.jpg", b);
}