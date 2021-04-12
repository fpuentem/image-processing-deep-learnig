//
#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

Mat_<unsigned char> mediana(Mat_<unsigned char> a){
    Mat_<unsigned char> b(a.rows, a.cols);
    for(int l=0; l<b.rows; l++){
        for(int c=0; c<b.cols; c++){
            vector<int> v;
    
            // Size of kernel is hardcode
            for(int l2=-1; l2<1; l2++){
                for(int c2=-1; c2<1; c2++){
                    if(0<=l+l2 && l+l2<a.rows && c+c2>=0 && c+c2 <a.cols){
                        v.push_back(a(l+l2,c+c2));
                    }
                }
            }
            auto meio =  v.begin() + v.size()/2; 
            nth_element(v.begin(), meio, v.end());
            b(l,c) = *meio;   
        }
    }
    return b;
}
int main(){
    Mat_<unsigned char> a;
    Mat_<unsigned char> b;

    a = imread("../imgs/filtros/ruido.png", 0);
    printf("**Original size**\n");
    printf("rows: %d\n", a.rows);
    printf("cols: %d\n", a.cols);
    b = mediana(a);

    printf("**Size after filter mediana**\n");
    printf("rows: %d\n", b.rows);
    printf("cols: %d\n", b.cols);
    imwrite("./mediana_lenna.jpg", b);
}