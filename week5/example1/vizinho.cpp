#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;

int main(int argc, char** argv){
    if( argc != 5){
        printf("vizinho: Muda resolucao de imagem usando interpolacao vizinho+px.\n");
        printf("vizinho ent.pgm sai.pgm fatorl fatorc\n");
        printf("Erro: Numero de argumentos invalido");
    }
    Mat_<unsigned char> a;
    a = imread( argv[1], IMREAD_GRAYSCALE );
    float fatorl, fatorc;

    if(sscanf(argv[3], "%f", &fatorl) != 1)
        printf("Erro, leitura fatorl");

    if(sscanf(argv[4], "%f", &fatorc) != 1)
        printf("Erro, leitura fatorc");

    int nl = cvRound(a.rows*fatorl);
    int nc = cvRound(a.cols*fatorc);

    Mat_<unsigned char> b(nl, nc);

    for(int l=0; l<b.rows; l++)
        for(int c=0; c<b.cols; c++)
            b(l, c) = a(cvRound(l/fatorl), cvRound(c/fatorc));    

    imwrite(argv[2], b);
}