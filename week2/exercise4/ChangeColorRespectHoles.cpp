#include <opencv2/opencv.hpp> 
// #include <queue>
#include <stack>

using namespace std;
using namespace cv;

void drawGreen(Mat3b *src,  Point s){
    // (*dst) = (*src).clone();

    Vec3b white(255, 255, 255);

    Vec3b green(0, 255, 0);
    
    stack<Point> stk;
    // (1)
    stk.push(s);
    

    while(!stk.empty()){ //(2)
        //(3)
        Point p = stk.top(); stk.pop();

        int l = p.x;
        int c = p.y;

        if ((*src)(l, c) == white){
            (*src)(l, c) = green;
            if((l-1) > -1){
                stk.push(Point(l-1, c));
            }

            if((l+1) < (*src).rows){
                stk.push(Point(l+1, c));
            }

            if((c+1) < (*src).cols){
                stk.push(Point(l, c+1));
            }

            if((c-1) > -1){
                stk.push(Point(l, c-1));
            }
        }
    }
}


void findNumberOfHoles(Mat3b *src, Point s){

    // Mat3b b = a.clone();
    int sz = 0;
    int szW = 0;

    Vec3b white(255, 255, 255);
    Vec3b black(0, 0, 0);
    Vec3b green(0, 255, 0);
    Vec3b blue(255, 0, 0);
    Vec3b red(0, 0, 255);

    int holes = 0;

    stack<Point> stk;
    stack<Point> stkW;
    
    if((*src)(s.x, s.y) == black){
        // (1)
        stk.push(s);
        
        while(!stk.empty()){ //(2)
            //(3)
            Point p = stk.top(); stk.pop();


            int l = p.x;
            int c = p.y;

            if ((*src)(l, c) == black){
                (*src)(l, c) = blue;
                sz = sz + 1;

                // in the borders of image?
                stk.push(Point(l-1, c));
                stk.push(Point(l+1, c));
                stk.push(Point(l, c+1));
                stk.push(Point(l, c-1));
            }else{
                if ((*src)(l, c) == white){
                    
                    stkW.push(Point(l, c));

                    while(!stkW.empty()){ //(2)
                        //(3)
                        Point pW = stkW.top(); stkW.pop();
                        int lW = pW.x;
                        int cW = pW.y;
                        if ((*src)(lW, cW) == white){
                            szW = szW + 1;
                            (*src)(lW, cW) = red;
                            
                            stkW.push(Point(lW-1, cW));
                            stkW.push(Point(lW+1, cW));
                            stkW.push(Point(lW, cW+1));
                            stkW.push(Point(lW, cW-1));                /* code */
                            }            
                    }
                    holes = holes + 1;
                    // printf(">>>szW: %d\n", szW);
                    szW = 0;
                }    
            }
        }

        printf("Number of holes: %d\n", holes);
        printf("Seed pixel l = %d and c = %d\n", s.x, s.y);
    }
   
}

// void changeColor(Mat3b *src,){

// }


int main(){
    Mat3b a = imread("../imgs/c3.bmp");
    Mat3b b = a.clone();

    drawGreen(&a, Point(0,0));
    
    int s;
    int count = 0;
    




    // sizeOfConnectedComponents(&a, Point(191, 286));
    for(int l=0; l<a.rows; l++){
        for(int c=0; c<a.cols; c++){
            findNumberOfHoles(&a, Point(l,c));
        }
    }

    imwrite("./fila.jpg", a);


    // // Display big mat
    imshow("PSI5796 Exercício 4 - Parte 2", a);
    waitKey(0);
}
/*
https://www.delftstack.com/howto/cpp/cpp-create-a-dictionary/
*/