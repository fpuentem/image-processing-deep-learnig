#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat_<unsigned char> a;
    a = imread("../imgs/mickey.bmp", 0);

    Mat_<unsigned char> b(a.rows, a.cols);
    b = a.clone();

    for(int l=0; l<a.rows; l++){
        for(int c=0; c<a.cols; c++){
            if(a(l,c) == 0){
                // Check the next row and column (white)
                if((a(l, c+1) == 255) || (a(l+1, c) == 255)){
                    b(l, c +1) = 0;
                    b(l+1, c) = 0;
                }        
            }
        }
    }        


    imwrite("./mickey-without-noise.bmp", b);

    // Get size of img
    int R = a.rows;
    int C = a.cols;
    
    // Create mat for window
    Mat_<unsigned char> win_mat(R, 2*C);
    // Mat win_mat(Size(2*C, R), CV_8UC3);
    
    // Copy small images into big mat
    a.copyTo(win_mat(Rect(0, 0, C, R)));
    b.copyTo(win_mat(Rect(C, 0, C, R)));

    // Display big mat
    imshow("PSI5796 Lição de casa 1", win_mat);
    waitKey(0);
}

