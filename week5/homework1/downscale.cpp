
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;


const float DOWNSCALE = 0.3;


int main(int argc, char** argv){

    Mat_<Vec3b> src; 
    src = imread( argv[1], IMREAD_COLOR );
    if( argc < 3){
        cerr << "Usage: " << argv[0] << " <path image file>" << " <downscale factor>" << endl;
        return EXIT_FAILURE;
    }

    float scale;
    if(sscanf(argv[2], "%f", &scale) != 1){
        cerr << "<downscale factor> must be float" << endl;
        return EXIT_FAILURE;
    }



    Mat_<Vec3b> dst;
    Mat_<Vec3b> tmp;

    float sigma1;
    float sigma2;
    sigma1 = (2.0*(1/scale))/6.0;
    sigma2 = (4.0*(1/scale))/6.0;

    // Part a: without filter
    resize(src, tmp, Size(0,0), scale, scale, INTER_NEAREST);
    imwrite("part_a_without_filter.png", tmp);

    // Part b: with gaussian filter sigma1 = 2*downsacle/6 (1.1 pixels)
    GaussianBlur(src, tmp, Size(5, 5), sigma1, 0);
    resize(tmp, dst, Size(0,0), scale, scale, INTER_NEAREST);
    imwrite("part_b_with_filter_sigma1.png", dst);

    // Part c: with gaussian filter sigma1 = 2*downsacle/6 (2.2 pixels)
    GaussianBlur(src, tmp, Size(5, 5), sigma2, 0);
    resize(tmp, dst, Size(0,0), scale, scale, INTER_NEAREST);
    imwrite("part_c_with_filter_sigma2.png", dst);

    // Part d: interpolation INTER_AREA
    resize(src, dst, Size(0,0), scale, scale, INTER_AREA);
    imwrite("part_d_inter_area.png", dst);

    return 0;
}