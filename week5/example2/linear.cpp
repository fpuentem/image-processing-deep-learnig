#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    if( argc != 5){
        printf("linear: Muda resolucao de imagem usando interpolacao bilinear.\n");
        printf("linear ent.pgm sai.pgm nl nc\n");
        printf("Erro: Numero de argumentos invalido");]
        return EXIT_FAILURE;
    }
    Mat_<unsigned char> a;
    a = imread( argv[1], IMREAD_GRAYSCALE );
    int nl;
    int nc;
    if(sscanf(argv[3], "%d", &nl) != 1)
        printf("Erro: leitura nl");
    if(sscanf(argv[4], "%d", &nc) != 1)
        printf("Erro: leitura nc");
    
    Mat_<unsigned char> b(nl, nc);

    for(int l=0; l<b.rows; l++){
        for(int c=0; c<b.cols; c++){
            double ald = l * ( (a.rows - 1.0) / (b.rows - 1.0) );
            double acd = c * ( (a.cols - 1.0) / (b.cols - 1.0) );
            int fal = int(ald);
            int fac = int(acd);

            double dl = ald - fal;
            double dc = acd - fac;

            double p1 = (1 - dl) * (1 - dc);
            double p2 = (1 - dl) * dc;
            double p3 = dl*(1 - dc);
            double p4 = dl * dc;

            b(l, c) =  cvRound(
                            p1*a(fal, fac) + p2*a(fal, fac+1) + 
                            p3*a(fal+1, fac) + p4*a(fal+1, fac+1)
                            );
        }
    }
        
    imwrite(argv[2], b);
}