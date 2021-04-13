//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;


int main( int argc, char ** argv ){

    if(argc =! 3){
        printf(" Error in number of arguments\n");
        printf(" Usage:\n %s [input_image_name] [output_image_name] \n", argv[0]);
        return EXIT_FAILURE;
    }
    
    Mat entg;
    Mat entf;
    Mat saig;
    Mat saif;

    entg = imread( argv[1], IMREAD_GRAYSCALE );
    entg.convertTo(entf, CV_32FC1, 1.0/255.0);

    Mat ker = (Mat_ <float>(3,3) << 
                                    +1, +1, +1,
                                    +1, +1, +1,
                                    +1, +1, +1);   

    ker = (1.0/9.0)*ker;

    filter2D(entf, saif, -1, ker, Point(-1, -1), 0);

    saif.convertTo(saig, CV_8UC1, 255.0);

    imwrite(argv[2], saig);
    
    // const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    // src = imread( samples::findFile( filename ), IMREAD_GRAYSCALE );
    // if (src.empty())
    // {
    //     printf(" Error opening image\n");
    //     printf(" Usage:\n %s [image_name-- default lena.jpg] \n", argv[0]);
    //     return EXIT_FAILURE;
    // }

    // dst = src.clone();

    // noiseAdd(src, dst);

    // imwrite("./lenna-with-noise.jpg", dst);
    // return 0;
}

