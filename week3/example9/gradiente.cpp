//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

void grad(Mat_<float> ent, Mat_<float>& saix, Mat_<float>& saiy){
    Mat mx = (Mat_ <float>(3,3) << 
                                    0.0 , 0.0,  0.0,
                                    1.0, 0.0,  -1.0,
                                    0.0 , 0.0,  0.0);    
    
    // mx = mx/4.0;



    Mat my = (Mat_ <float>(3,3) << 
                                    -3.0 , -10.0,  -3.0,
                                     0.0 ,   0.0,   0.0,
                                    +3.0 , +10.0,  +3.0);
    my = my/16.0;

    filter2D(ent, saix, -1, mx, Point(-1, -1), 0);
    
    filter2D(ent, saiy, -1, my, Point(-1, -1), 0);
}



int main( void ){

    Mat_<float> ent;
    ent = imread( "../imgs/convolucao/fantom.pgm" , IMREAD_GRAYSCALE );
    
    Mat_<float> saix;
    Mat_<float> saiy;
    
    grad(ent, saix, saiy);

    Mat_<char> t;
    t = 0.5 + 5*saix;
    imwrite( "gardx_one_side_2.png", t );

    t = 0.5 + saiy;
    imwrite( "gardy.png", t );

    Mat_<float>saix_tmp;
    Mat_<float>saiy_tmp;

    pow(saix, 2, saix_tmp);
    pow(saiy, 2, saiy_tmp);

    Mat_<float> modgrad;
    sqrt(saix_tmp + saiy_tmp, modgrad);

    imwrite( "modgrad.png", modgrad );

}

