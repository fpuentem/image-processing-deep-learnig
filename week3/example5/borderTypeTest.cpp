//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(){
    Mat ent = (Mat_<float>(3,3) << 2, 2, 2, 3, 3, 3, 4, 4, 4);
    Mat ker = (Mat_<float>(3,3) << 1, 1, 1, 1, 1, 1, 1, 1, 1);
    Mat_<float> sai;
    // BORDER_CONSTANT 
    // BORDER_REPLICATE 
    // BORDER_REFLECT 
    // BORDER_WRAP 
    // BORDER_REFLECT_101 
    // BORDER_TRANSPARENT 
    // BORDER_REFLECT101 
    filter2D(ent,sai,-1,ker,Point(-1,-1),0,BORDER_CONSTANT);
    
    cout << "sai  = " << endl << format(sai, Formatter::FMT_PYTHON) << endl << endl;
}


