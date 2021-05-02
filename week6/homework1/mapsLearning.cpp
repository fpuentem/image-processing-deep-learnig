#include <opencv2/opencv.hpp> 
#include <cmath>

using namespace std;
using namespace cv;


Mat_<unsigned char> features33(Mat_<unsigned char> a, int lc, int cc) {
    Mat_<unsigned char> d(1,9);
    int i=0;

    for (int l=-1; l<=1; l++){
        for (int c=-1; c<=1; c++) {
            d(i)=a(lc+l,cc+c);
            i++;
        }
    }
    return d;
}


int main() {

    Mat_<unsigned char> ax; 
    ax = imread( "../aprendizagem/janei.pgm", IMREAD_GRAYSCALE );
    
    Mat_<unsigned char> ay; 
    ay = imread( "../aprendizagem/janei-1.pgm", IMREAD_GRAYSCALE );

    Mat_<unsigned char> qx; 
    qx = imread( "../aprendizagem/julho.pgm", IMREAD_GRAYSCALE );


    Mat_<Vec3b> colorQp; 
    cvtColor(qx, colorQp, COLOR_GRAY2BGR);

    Mat_<unsigned char> qp(qx.rows,qx.cols);
    Mat_<unsigned char> tmp(qx.rows,qx.cols);
    Mat_<unsigned char> tmp1(qx.rows,qx.cols);


    Mat_<float> features((ax.rows-1)*(ax.cols-1), 9);	
    Mat_<float> saidas((ax.rows-1)*(ax.cols-1), 1);


    int i = 0;
    for (int l=1; l<ax.rows-1; l++){				
        for (int c=1; c<ax.cols-1; c++){
            Mat_<unsigned char> f(1,9);
            f = features33(ax,l,c);				
            
            for(int k = 0; k < 9; k++){
                features(i,k) = f(k)/255.0;		
            }
            
            saidas(i) = ay(l,c)/255.0;			
            i = i + 1;
        }
    }

    flann::Index ind(features,flann::KDTreeIndexParams(4));	// Aqui, as 4 arvores estao criadas e procura na matriz features

    Mat_<float> query(1,9);								//matriz de busca, mesma dimensao features (9)

    vector<int> indices(1);								//retorna valor do vizinho próximo
    vector<float> dists(1);								//distância do pixel ao vizinho mais próximo




    for (int l=0; l<qp.rows; l++){
        for (int c=0; c<qp.cols; c++) {
            query = features33(qx, l, c)/255.0;
            ind.knnSearch(query, indices, dists, 1, flann::SearchParams(0));
            qp(l,c)=255*saidas(indices[0]);			// Saida e' um numero entre 0 e 255
        }
    }

    //Filtros medianas para eliminar o ruido
    medianBlur(qp, tmp1, 5);
    medianBlur(tmp1, tmp, 7);

    //pintado de vermelho e superposicao imagem
    for (int l=0; l<qp.rows; l++){
        for (int c=0; c<qp.cols; c++) {
            if(tmp(l, c) == 0)
                colorQp(l,c) = Vec3b(0, 0, 200);
        }
    }

    imwrite("julho-flann_filter_5_7.png", colorQp);

}