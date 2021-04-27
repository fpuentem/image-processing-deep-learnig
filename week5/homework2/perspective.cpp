#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;


int main() {

    // Point tp point perspective correction
    Mat_<float> src = (Mat_<float>(4,2) <<	235,156,
                                            639,204,
                                            451,323,
                                            0  ,259);

    Mat_<float> dst = (Mat_<float>(4,2) <<	10 ,10,
                                            628,10,
                                            628,467,
                                            10 ,467);


    // Calculation of transformatin matrix
    Mat_<double> m = getPerspectiveTransform(src, dst);
    cout << "m = " << endl << format(m, Formatter::FMT_PYTHON)
     << endl << endl;


    // Verification
    Mat_<double> v = (Mat_<double>(3,1) << 451, 323, 1 );
    Mat_<double> w = m*v;

    cout << "x = " << endl << format(w(0)/w(2), Formatter::FMT_PYTHON) << endl << endl;
    cout << "y = " << endl << format(w(1)/w(2), Formatter::FMT_PYTHON) << endl << endl;

    // Perspective correction
    Mat_<Vec3b> s; 
    Mat_<Vec3b> d;
    s = imread("../transformacao/quadrado2.png", IMREAD_COLOR);
    
    warpPerspective(s, d, m, s.size(), INTER_LINEAR, 
    BORDER_CONSTANT, Scalar(255, 255, 255));
    imwrite("quadrado2b.png", d);

    return 0;
}
