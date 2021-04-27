#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv){

    if( argc != 4){
        printf("cvvizinho ent.pgm sai.pgm fator\n");
        printf("Erro: Numero de argumentos invalido");
        return EXIT_FAILURE;    
    }
    Mat_<unsigned char> a;
    a = imread( argv[1], IMREAD_GRAYSCALE );
    
    double fator;
    sscanf(argv[3], "%lf", &fator);

    Mat_<unsigned char> b;
    resize(a, b, Size(0,0), fator, fator, INTER_NEAREST);

    imwrite(argv[2], b);




}
