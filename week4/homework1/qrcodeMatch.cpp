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

    Mat_<uchar> temp;
    
    Mat_<float> a;
    temp = imread( "../tmatch-simp/op00.jpg", IMREAD_GRAYSCALE );
    temp.convertTo(a, CV_32F, 1.0/255.0, 0.0);

    Mat_<float> q;
    q = imread( "../tmatch-simp/padrao_reduz.png", IMREAD_GRAYSCALE );
    q = somaAbsDois( dcReject(q) );
    
    Mat_<float> p;
    filter2D(a, p, -1, q, Point(-1, -1), 0, BORDER_REPLICATE);

    p.convertTo(temp, CV_8U, 255.0, 0.0);
    imwrite("correlacao3.png", temp);

    a.convertTo(temp, CV_8U, 255.0, 0.0);
    Mat_<Vec3b> d;
    
    cvtColor(temp, d, COLOR_GRAY2RGB);

    for(int l = 0; l < a.rows; l++){
        for(int c = 0; c < a.cols; c++){
            if( p(l, c) >= 0.40 ){
                rectangle( d, Point(c-35,l-35), Point(c+35,l+35), Scalar(0,0,255), 6 );   
            }
        }
    }

    imwrite("ocorrencia3.png", d);

}




