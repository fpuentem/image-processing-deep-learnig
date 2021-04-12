//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;



void noiseAdd(Mat_<unsigned char>, Mat_<unsigned char>);

int main( int argc, char ** argv ){
    Mat_<unsigned char> src;
    Mat_<unsigned char> dst;

    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    src = imread( samples::findFile( filename ), IMREAD_GRAYSCALE );
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- default lena.jpg] \n", argv[0]);
        return EXIT_FAILURE;
    }

    dst = src.clone();

    noiseAdd(src, dst);

    imwrite("./lenna-with-noise.jpg", dst);
    return 0;
}

void noiseAdd(Mat_<unsigned char> src, Mat_<unsigned char> dst){
    srand(7);
    for(int l=0; l<src.rows; l++){
        for(int c=0; c<src.cols; c++){
            if(rand()%20 == 0){
                dst(l,c) = rand()%256;
            }
        }
    }

}
