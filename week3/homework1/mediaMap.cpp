//
#include <opencv2/opencv.hpp> 
#include <cmath>
#include <string>
using namespace std;
using namespace cv;

int DELAY_CAPTION = 2000;
int DELAY_BLUR = 2000;
int MAX_KERNEL_LENGTH = 21;
Mat src; Mat dst;
char window_name[] = "Median Blur - OpenCV";
int display_caption( const char* caption );
int display_dst( int delay );
char kernel_str[20];

int main( int argc, char ** argv ){
    namedWindow( window_name, WINDOW_AUTOSIZE );
    const char* filename = argc >=2 ? argv[1] : "lena.jpg";
    src = imread( samples::findFile( filename ), IMREAD_COLOR );
    if (src.empty())
    {
        printf(" Error opening image\n");
        printf(" Usage:\n %s [image_name-- default lena.jpg] \n", argv[0]);
        return EXIT_FAILURE;
    }
    if( display_caption( "Original Image" ) != 0 )
    {
        return 0;
    }
    dst = src.clone();
    if( display_dst( DELAY_CAPTION ) != 0 )
    {
        return 0;
    }
    if( display_caption( "Median Blur" ) != 0 )
    {
        return 0;
    }
    for ( int i = 1; i < MAX_KERNEL_LENGTH; i = i + 2 )
    {
        int r = sprintf(kernel_str, "Kernel Size: %d", i); 
        display_caption(kernel_str);

        medianBlur( src, dst, i );
        
        if(argc >= 3 && stoi(argv[2]) == i){
            imwrite("./mediana_map.png", dst);
        }
        
        if( display_dst( DELAY_BLUR ) != 0 )
        {
            return 0;
        }

    }
    display_caption( "Done!" );
    return 0;
}

int display_caption( const char* caption )
{
    dst = Mat::zeros( src.size(), src.type() );
    putText( dst, caption,
             Point(20, src.rows/2),
             FONT_HERSHEY_COMPLEX, 1, Scalar(255, 255, 255) );
    return display_dst(DELAY_CAPTION);
}

int display_dst( int delay )
{    
    imshow( window_name, dst );
    int c = waitKey ( delay );
    if( c >= 0 ) { return -1; }
    return 0;
}