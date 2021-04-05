//
#include <opencv2/opencv.hpp> 
#include <queue>

using namespace std;
using namespace cv;

Mat3b drawBlue(Mat3b a, int ls, int cs){
    Mat3b b = a.clone();

    Vec3b white(255, 255, 255);

    Vec3b blue(255, 0, 0);
    queue<int> q;
    // (1)
    q.push(ls);
    q.push(cs);

    while(!q.empty()){ //(2)
        //(3)
        int l = q.front(); q.pop();
        int c = q.front(); q.pop();
        if (b(l, c) == white){
            b(l, c) = blue;
            if((l-1) > -1){
                q.push(l-1); q.push(c);
            }

            if((l+1) < b.rows){
                q.push(l+1); q.push(c);
            }

            if((c+1) < b.cols){
                q.push(l); q.push(c+1);
            }

            if((c-1) > -1){
                q.push(l); q.push(c-1);
            }
        }
    }
    return b;
}

int main(){

    Mat3b a = imread("../imgs/mickey_reduz.bmp");
    Mat3b b;

    b = drawBlue(a, 5, 4);
 
    imwrite("./fila.jpg", b);

    // Get size of img
    int R = a.rows;
    int C = b.cols;

    // printf("Rows: %d\n", a.rows);
    // printf("Colss: %d\n", a.cols);
    
    // Create mat for window
    Mat win_mat(Size(2*C, R), CV_8UC3);
    
    // Copy small images into big mat
    a.copyTo(win_mat(Rect(0, 0, C, R)));
    b.copyTo(win_mat(Rect(C, 0, C, R)));

    // Display big mat
    imshow("PSI5796 Exercício 1 - Parte 2", win_mat);
    waitKey(0);

}
