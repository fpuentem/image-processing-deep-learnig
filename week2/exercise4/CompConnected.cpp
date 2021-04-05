#include <opencv2/opencv.hpp> 
// #include <queue>
#include <stack>

using namespace std;
using namespace cv;


int sizeOfConnectedComponents(Mat3b *a, int ls, int cs, Point *topCorner, Point *botCorner){

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
            // the lowest a.rows and a.cols
            if((*topCorner).x > l)    
                (*topCorner).x = l;

            if((*topCorner).y >  c)     
                (*topCorner).y = c;

            // the highest 0 and 0 
            if((*botCorner).x < l)    
                (*botCorner).x = l;

            if((*botCorner).y < c)    
                (*botCorner).y = c;

            (*a)(l, c) = blue;
            sz = sz + 1;

            // in the borders of image?
            stk.push(Point(l-1, c));
            stk.push(Point(l+1, c));
            stk.push(Point(l, c+1));
            stk.push(Point(l, c-1));
        }
    }
    return sz;
}

int main(){

    Mat3b a = imread("../imgs/c3.bmp");
    int s;
    int count = 0;
    
    Point b(0,0);
    Point t(a.rows, a.cols);

    for(int l=0; l<a.rows; l++){
        for(int c=0; c<a.cols; c++){
        
            Point b(0,0);
            Point t(a.rows, a.cols);
            
            
            s = sizeOfConnectedComponents(&a, l, c, &t, &b);
            
            if(s > 1){
                count = count + 1;
                printf("********\n");
                printf("# pix of connected components: %d\n", s);
                printf("Top point: (%d, %d)\n", t.x, t.y);
                printf("Botton point: (%d, %d)\n", b.x, b.y);
            
            }
        }
    }
    
    printf("Number of connected components: %d\n", count);

    imwrite("./fila.jpg", a);


    // // Display big mat
    imshow("PSI5796 Exercício 4 - Parte 2", a);
    waitKey(0);
}