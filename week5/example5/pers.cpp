#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    Mat_<float> src = ( Mat_<float>(3, 2) << 
                                            73, 0,
                                            533,  0,
                                            -22, 479,
                                            629, 479);

    cout << "src = " << endl << format(src, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> dst = ( Mat_<float>(3, 2) << 
                                            16, 0,
                                            630, 0,
                                            14, 479,
                                            630, 479);

    // Verifica se a transformaçao esta fazendo o que queremos
    Mat_<double> v = ( Mat_<double>(3,1) << -22, 479, 1);
    Mat_<double> w = m * v;
    cout << "w = " << endl << format(w, Formatter::FMT_PYTHON) << endl << endl;


    cout << w(0)/w(2) << " " << w(1)/w(2) << endl;






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
