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
    Mat_<uchar> img;

    Mat_<float> a;
    temp = imread( "../tmatch/a.png", IMREAD_GRAYSCALE );
    img = temp.clone();
    temp.convertTo(a, CV_32F, 1.0/255.0, 0.0);

    Mat_<float> q;
    temp = imread( "../tmatch/q.png", IMREAD_GRAYSCALE );    
    temp.convertTo(q, CV_32F, 1.0/255.0, 0.0);
    

    Mat_<float> p = matchTemplateSame(a, q, TM_CCOEFF_NORMED);  
    p = abs(p);
    p.convertTo(temp, CV_8U, 255.0, 0.0); 
    
    imwrite("ursosNCC.png", temp);

    Mat_<Vec3b> d;
    cvtColor(img, d, COLOR_GRAY2RGB);

    for(int l = 0; l < a.rows; l++){
        for(int c = 0; c < a.cols; c++){
            if( p(l, c) > 0.9 ){
                rectangle(d, Point(c,l), Point(c,l), Scalar(0,255,0), 4);   
            }
        }
    }

    imwrite("ocorrenciaUrso.png", d);

    return 0;
}


   
   