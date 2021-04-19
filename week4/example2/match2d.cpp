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
    Mat_<float> a;
    Mat_<float> b;
    a = imread( "../tmatch/bbox2.pgm", IMREAD_GRAYSCALE );
    b = a/255.0;


    Mat_<float> q;
    q = imread( "../tmatch/letramore.bmp", IMREAD_GRAYSCALE );
    q = somaAbsDois( dcReject(q));

    Mat_<float> p;
    Mat_<float> p_;
    filter2D(b, p, -1, q, Point(-1, -1), 0, BORDER_REPLICATE);
    filter2D(a, p_, -1, q, Point(-1, -1), 0, BORDER_REPLICATE);
    imwrite("correlacao.png", p_);

    Mat d;
    cvtColor(a, d, COLOR_GRAY2RGB);
    for(int l = 0; l < a.rows; l++){
        for(int c = 0; c < a.cols; c++){
            if( p(l, c) >= 0.8){
                rectangle( d, Point(c-109,l-38), Point(c+109,l+38), Scalar(0,0,255), 6 );   
            }
        }
    }

    imwrite("ocorrencia.png", d);

}


