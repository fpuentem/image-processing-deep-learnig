//
#include <opencv2/opencv.hpp> 
#include <queue>

using namespace std;
using namespace cv;

Mat3b drawRed(Mat3b a, int ls, int cs, float threshold){
    Mat3b b = a.clone();

    Vec3b red(0, 0, 255);

    Vec3b yellow(20, 200, 200);
    
    queue<int> q;
    // (1)
    q.push(ls);
    q.push(cs);

    while(!q.empty()){ //(2)
        //(3)
        int l = q.front(); q.pop();
        int c = q.front(); q.pop();
        
        double dist = norm(yellow, b(l, c), NORM_L2);
        
        if (dist < threshold){
            b(l, c) = red;
            q.push(l-1); q.push(c);
            q.push(l+1); q.push(c);
            q.push(l); q.push(c+1);
            q.push(l); q.push(c-1);
        }
    }
    return b;
}

int main(){

    Mat3b a = imread("../imgs/elefante.jpg");
    Mat3b b;

    b = drawRed(a, 125, 125, 70.0);
 
    imwrite("./red-elefante.jpg", b);

    // Create 706x310 mat for window
    Mat win_mat(cv::Size(706, 310), CV_8UC3);
    
    // Copy small images into big mat
    a.copyTo(win_mat(Rect(  0, 0, 353, 310)));
    b.copyTo(win_mat(Rect(353, 0, 353, 310)));

    // Display big mat
    imshow("PSI5796 Lição de casa 2", win_mat);
    waitKey(0);
}

