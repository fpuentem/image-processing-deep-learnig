#include <opencv2/opencv.hpp> 
// #include <queue>
#include <stack>

using namespace std;
using namespace cv;


int sizeOfConnectedComponents(Mat3b *a, int ls, int cs){

    // Mat3b b = a.clone();
    int sz = 0;
    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
    Vec3b blue(255, 0, 0);

    stack<Point> stk;
    // (1)
    stk.push(Point(ls, cs));
    

    while(!stk.empty()){ //(2)
        //(3)
        Point p = stk.top(); stk.pop();


        int l = p.x;
        int c = p.y;

        if ((*a)(l, c) == black){
            (*a)(l, c) = blue;
            sz = sz + 1;
            stk.push(Point(l-1, c));
            stk.push(Point(l+1, c));
            stk.push(Point(l, c+1));
            stk.push(Point(l, c-1));
        }
    }
    return sz;
}

int main(){

    Mat3b a = imread("../imgs/letras.bmp");
    int s;
    int count = 0;

    for(int l=0; l<a.rows; l++){
        for(int c=0; c<a.cols; c++){
            s = sizeOfConnectedComponents(&a, l, c);
            
            if(s > 1)
                count = count + 1;
        }
    }
    
    printf("Number of connected components: %d\n", count);

    imwrite("./fila.jpg", a);


    // // Display big mat
    imshow("PSI5796 Exercício 3 - Parte 2", a);
    waitKey(0);
}