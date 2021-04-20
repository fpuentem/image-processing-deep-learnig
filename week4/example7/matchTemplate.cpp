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

Mat_<float> matchTemplateSame(Mat_<float> a, Mat_<float> q,int method,float backg=0.0){  
    Mat_<float> p{ a.size(), backg };  
    Rect rect{(q.cols-1)/2,(q.rows-1)/2, a.cols-q.cols+1, a.rows-q.rows+1};  
    Mat_<float> roi{ p, rect };  
    matchTemplate(a, q, roi, method);
    return p;
}

int main(){

    Mat_<uchar> temp;
    
    Mat_<float> a;
    // "../tmatch-simp/op00.jpg"
    temp = imread( "../tmatch/a.png", IMREAD_GRAYSCALE );
    temp.convertTo(a, CV_32F, 1.0/255.0, 0.0);

    Mat_<float> q;
    // "../tmatch-simp/padrao_reduz.png"
    temp = imread( "../tmatch/q.png", IMREAD_GRAYSCALE );    
    temp.convertTo(q, CV_32F, 1.0/255.0, 0.0);
    
    Mat_<float> q1 = somaAbsDois( dcReject(q) );
    Mat_<float> p1;
    filter2D(a, p1, -1, q1, Point(-1, -1), 0, BORDER_REPLICATE);
    p1.convertTo(temp, CV_8U, 255.0, 0.0);
    imwrite("qr-p1.png", temp);

    Mat_<float> p2 = matchTemplateSame(a, q1, TM_CCORR);  
    p2.convertTo(temp, CV_8U, 255.0, 0.0); 
    imwrite("qr-p2.png", temp);

    Mat_<float> p3 = matchTemplateSame(a, q, TM_CCOEFF_NORMED);  
    p3 = abs(p3);
    p3.convertTo(temp, CV_8U, 255.0, 0.0); 
    
    p3 = abs(p3);
    imwrite("qr-p3.png", temp);

    return 0;
}


   