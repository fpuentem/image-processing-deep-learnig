#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(){
    Mat_<unsigned char> a;
    a = imread("../imgs/mickey_reduz.bmp", 0);

    Mat_<unsigned char> b(a.rows, a.cols);

    for(int l=0; l<a.rows; l++)
        for(int c=0; c<a.cols; c++)
            if(a(l,c) != a(l, c+1) || a(l,c) != a(l, c+1))
                b(l, c) = 0;
            else
                b(l,c) = 255;

    imwrite("./edge_img.bmp", b);
}