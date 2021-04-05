#include <opencv2/opencv.hpp> 
// #include <queue>
#include <stack>

using namespace std;
using namespace cv;


Mat3b drawBlue(Mat3b a, int ls, int cs){
    Mat3b b = a.clone();

    Vec3b white(255, 255, 255);
    Vec3b blue(255, 0, 0);

    stack<Point> stk;
    // (1)
    stk.push(Point(ls, cs));
    

    while(!stk.empty()){ //(2)
        //(3)
        Point p = stk.top(); stk.pop();


        int l = p.x;
        int c = p.y;

        if (b(l, c) == white){
            b(l, c) = blue;
            stk.push(Point(l-1, c));
            stk.push(Point(l+1, c));
            stk.push(Point(l, c+1));
            stk.push(Point(l, c-1));
        }
    }
    return b;
}

int main(){

    Mat3b a = imread("../imgs/mickey_reduz.bmp");
    Mat3b b;

    b = drawBlue(a, 159, 165);
 
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
    imshow("PSI5796 Exercício 2 - Parte 2", win_mat);
    waitKey(0);
}