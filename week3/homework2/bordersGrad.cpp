//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

void ShowManyImages(string title, int nArgs, ...);

int main( void ){
    
    Mat src;
    Mat grad;
    const String window_name = "Sobel Demo - Simple Edge Detector";
    int ksize = 3;
    int scale = 1;
    int delta = 0;
    int ddepth = CV_16S;

    src = imread( "../imgs/convolucao/quadrado.png" , IMREAD_GRAYSCALE );
    // x direction
    Mat grad_x_a;
    Mat grad_x_b;
    Mat abs_grad_x;
    Sobel(src, grad_x_a, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
    
    grad_x_b = - grad_x_a;
    
    // converting back to CV_8U
    convertScaleAbs(grad_x_a, abs_grad_x);
    
    imwrite( "grad_x_a.png", grad_x_a );
    imwrite( "grad_x_b.png", grad_x_b );
    imwrite( "abs_grad_x.png", abs_grad_x );

    // y direction
    Mat grad_y_a;
    Mat grad_y_b;
    Mat abs_grad_y;
    Sobel(src, grad_y_a, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    
    grad_y_b = - grad_y_a;
    // converting back to CV_8U
    convertScaleAbs(grad_y_a, abs_grad_y);
    
    imwrite( "grad_y_a.png", grad_y_a );
    imwrite( "grad_y_b.png", grad_y_b );
    imwrite( "abs_grad_y.png", abs_grad_y );

    ShowManyImages("PSI5796 Lição de casa 2", 6, abs_grad_x, grad_x_a, grad_x_b, abs_grad_y, grad_y_a, grad_y_b);
}


void ShowManyImages(string title, int nArgs, ...) {
    int size;
    int i;
    int m, n;
    int x, y;

    // w - Maximum number of images in a row
    // h - Maximum number of images in a column
    int w, h;

    // scale - How much we have to resize the image
    float scale;
    int max;

    // If the number of arguments is lesser than 0 or greater than 12
    // return without displaying
    if(nArgs <= 0) {
        printf("Number of arguments too small....\n");
        return;
    }
    else if(nArgs > 14) {
        printf("Number of arguments too large, can only handle maximally 12 images at a time ...\n");
        return;
    }
    // Determine the size of the image,
    // and the number of rows/cols
    // from number of arguments
    else if (nArgs == 1) {
        w = h = 1;
        size = 300;
    }
    else if (nArgs == 2) {
        w = 2; h = 1;
        size = 300;
    }
    else if (nArgs == 3 || nArgs == 4) {
        w = 2; h = 2;
        size = 300;
    }
    else if (nArgs == 5 || nArgs == 6) {
        w = 3; h = 2;
        size = 200;
    }
    else if (nArgs == 7 || nArgs == 8) {
        w = 4; h = 2;
        size = 200;
    }
    else {
        w = 4; h = 3;
        size = 150;
    }

    // Create a new 3 channel image
    Mat DispImage = Mat::zeros(Size(100 + size*w, 60 + size*h), CV_8U);

    // Used to get the arguments passed
    va_list args;
    va_start(args, nArgs);

    // Loop for nArgs number of arguments
    for (i = 0, m = 20, n = 20; i < nArgs; i++, m += (20 + size)) {
        // Get the Pointer to the IplImage
        Mat img = va_arg(args, Mat);

        // Check whether it is NULL or not
        // If it is NULL, release the image, and return
        if(img.empty()) {
            printf("Invalid arguments");
            return;
        }

        // Find the width and height of the image
        x = img.cols;
        y = img.rows;

        // Find whether height or width is greater in order to resize the image
        max = (x > y)? x: y;

        // Find the scaling factor to resize the image
        scale = (float) ( (float) max / size );

        // Used to Align the images
        if( i % w == 0 && m!= 20) {
            m = 20;
            n+= 20 + size;
        }

        // Set the image ROI to display the current image
        // Resize the input image and copy the it to the Single Big Image
        Rect ROI(m, n, (int)( x/scale ), (int)( y/scale ));
        Mat temp; resize(img,temp, Size(ROI.width, ROI.height));
        temp.copyTo(DispImage(ROI));
    }

    // Create a new window, and show the Single Big Image
    namedWindow( title, 1 );
    imshow( title, DispImage);
    waitKey();

    // End the number of arguments
    va_end(args);
}