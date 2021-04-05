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
            q.push(l-1); q.push(c);
            q.push(l+1); q.push(c);
            q.push(l); q.push(c+1);
            q.push(l); q.push(c-1);
        }
    }
    return b;
}

int main(){

    Mat3b a = imread("../imgs/mickey_reduz.bmp");
    Mat3b b;

    b = drawBlue(a, 159, 165);
 
    imwrite("./fila.jpg", b);
}