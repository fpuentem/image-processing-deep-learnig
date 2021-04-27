#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    Mat_<float> src = ( Mat_<float>(3, 2) << 
                                            0, 0,
                                            0, 512,
                                            511, 511);

    cout << "src = " << endl << format(src, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> dst = ( Mat_<float>(3, 2) << 
                                            200, 100,
                                            100, 400,
                                            400, 400);

    cout << "dst = " << endl << format(dst, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> m =  getAffineTransform(src, dst);
    cout << "m = " << endl << format(m, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<unsigned char> a;
    a = imread( "../transformacao/lennag.jpg", IMREAD_GRAYSCALE ); 

    Mat_<unsigned char> b;
    warpAffine(a, b, m, a.size(), INTER_LINEAR, BORDER_WRAP);

    imwrite("afim.jpg", b);


    return 0;
}
