//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;


Mat_ <float> dcReject(Mat_<float> a){
    // Elimina nivel DC (substrai media)
    a = a - mean(a)[0];
    return a;
}

Mat_<float> dcReject(Mat_<float> a, float dontcare){
    // Elimina nivel DC (substrai media) com dontcare
    Mat_<uchar> naodontcare = (a != dontcare);
    Scalar media = mean(a, naodontcare);
    subtract(a, media[0], a, naodontcare);
    Mat_<uchar> simdontcare = (a == dontcare);
    subtract(a, dontcare, a, simdontcare);
    return a;
}

Mat_<float> somaAbsDois(Mat_<float> a){
    // Faz somatoria absoluta da imagen dar dois
    double soma = sum(abs(a))[0];
    a /= (soma/2.0);
    return a;
}

int main(){
    Mat_<float> a = ( Mat_<float>(1, 13) << 
                                0, 1, 5, 3, 1, -1, 3, 1, 1, -2, 6, 2, 0);
    cout << "a = " << endl << format(a, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> q = ( Mat_<float>(1,3) << 0, 1, 0.5 );
    cout << "q = " << endl << format(q, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> p;
    filter2D(a, p, -1, q, Point(-1, -1), 0, BORDER_REPLICATE);
    cout << "p = " << endl << format(p, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> q2 = somaAbsDois(dcReject(q));
    cout << "q2 = " << endl << format(q2, Formatter::FMT_PYTHON) << endl << endl;
    Mat_<float> p2;
    filter2D(a, p2, -1, q2, Point(-1, -1), 0, BORDER_REPLICATE);
    cout << "p2 = " << endl << format(p2, Formatter::FMT_PYTHON) << endl << endl;

    Mat_<float> p3;
    Mat_<float> q3 = ( Mat_<float>(1,3) << 0.2, 0.8, 0.5 );
    Mat_<float> q4 = somaAbsDois(dcReject(q3));
    cout << "q4 = " << endl << format(q4, Formatter::FMT_PYTHON) << endl << endl;
    filter2D(a, p3, -1, q3, Point(-1, -1), 0, BORDER_REPLICATE);
    cout << "p3 = " << endl << format(p3, Formatter::FMT_PYTHON) << endl << endl;
}
