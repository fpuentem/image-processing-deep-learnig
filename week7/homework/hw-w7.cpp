#include <opencv2/opencv.hpp> 
#include <cmath>
#include <chrono>

using namespace std;
using namespace cv;

void erro(string st1, string st2) { 
    string st = st1 + st2; 
    // colorPrint(st,0xc);
    printf("%s\n", st.c_str()); 
    exit(1); 
}

void xerro1(string st1) { 
    string st = "File="+string(__FILE__)+" line="+to_string(__LINE__)+" "+st1+"\n"; 
    // colorPrint(st,0xc); 
    printf("%s\n", st.c_str());
    exit(1); 
}

template<class T>
void copia(Mat_<T> ent, Mat_<T>& sai, int li, int ci) {
    // Copia ent para dentro de sai a partir de sai(li,ci).
    // sai deve vir alocado
    // ent deve ser normalmente menor que sai
    // Ha protecao para o caso de ent nao caber dentro de sai
    int lisai=max(0,li);
    int cisai=max(0,ci);
    int lfsai=min(sai.rows-1,li+ent.rows-1);
    int cfsai=min(sai.cols-1,ci+ent.cols-1);
    Rect rectsai=Rect(cisai,lisai,cfsai-cisai+1,lfsai-lisai+1);

    int lient=max(0,-li);
    int cient=max(0,-ci);
    int lfent=min(ent.rows-1, sai.rows-1-li);
    int cfent=min(ent.cols-1, sai.cols-1-ci);
    Rect rectent=Rect(cient,lient,cfent-cient+1,lfent-lient+1);

    if (rectsai.width>0 && rectsai.height>0 && rectent.width>0 && rectent.height>0) {
        Mat_<T> sairoi=sai(rectsai);
        Mat_<T> entroi=ent(rectent);
        entroi.copyTo(sairoi);
    }
}


class MNIST {
    public:
        bool localizou;
        int nlado; bool inverte; bool ajustaBbox; string metodo;
        int na; vector< Mat_<unsigned char> > AX; vector<int> AY; Mat_<float> ax; Mat_<float> ay;
        int nq; vector< Mat_<unsigned char> > QX; vector<int> QY; Mat_<float> qx; Mat_<float> qy;
        Mat_<float> qp;

        MNIST(int _nlado=28, bool _inverte=true, bool _ajustaBbox=true, string _metodo="flann") {
            nlado=_nlado; inverte=_inverte;
            ajustaBbox=_ajustaBbox; metodo=_metodo;
        }
        Mat_<unsigned char> bbox(Mat_<unsigned char> a); // Ajusta para bbox. Se nao consegue, faz localizou=false
        Mat_<float> bbox(Mat_<float> a); // Ajusta para bbox. Se nao consegue, faz localizou=false
        void leX(string nomeArq, int n, vector< Mat_<unsigned char> >& X, Mat_<float>& x); // funcao interna
        void leY(string nomeArq, int n, vector<int>& Y, Mat_<float>& y); // f. interna
        void le(string caminho="", int _na=60000, int _nq=10000);
        // Le banco de dados MNIST que fica no path caminho
        // ex: mnist.le("."); mnist.le("c:/diretorio");
        // Se _na ou _nq for zero, nao le o respectivo
        // ex: mnist.le(".",60000,0);
        int contaErros();
        Mat_<unsigned char> geraSaida(Mat_<unsigned char> q, int qy, int qp); // f. interna
        Mat_<unsigned char> geraSaidaErros(int maxErr=0);
        // Conta erros e gera imagem com maxErr primeiros erros
        Mat_<unsigned char> geraSaidaErros(int nl, int nc);
        // Gera uma imagem com os primeiros nl*nc digitos classificados erradamente
};




int main(){
    MNIST mnist(14,true,true);  
    mnist.le("/home/fabricio/projects/usp-projects/image-processing-deep-learnig/week7/mnist");  
    
    // TimePoint t1 = timePoint();  
    auto tic = chrono::steady_clock::now();

    flann::Index ind(mnist.ax,flann::KDTreeIndexParams(4));  
    
    // TimePoint t2 = timePoint();  
    auto toc0 = chrono::steady_clock::now();

    vector<int> indices(1); vector<float> dists(1);
    for(int l=0; l<mnist.qx.rows; l++){    
        ind.knnSearch(mnist.qx.row(l),indices,dists,1,flann::SearchParams(0));    
        mnist.qp(l)=mnist.ay(indices[0]);
    }  
    
    // TimePoint t3 = timePoint();  
    auto toc1 = chrono::steady_clock::now();
    
    printf("Erros=%10.2f%%\n", 100.0*mnist.contaErros()/mnist.nq);  
    // printf("Tempo de treinamento: %f\n",timeSpan(t1,t2));  
    // printf("Tempo de predicao: %f\n",timeSpan(t2,t3));
    

    cout << "Tempo de treinamento: " << chrono::duration_cast<chrono::milliseconds>(toc0 - tic).count() << " sec\n";
    cout << "Tempo de predicao: " << chrono::duration_cast<chrono::milliseconds>(toc1 - toc0).count() << " sec\n";
    
    return 0;

}

//<<<<<<<<<<<<<<<<< MNIST <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
Mat_<unsigned char> MNIST::bbox(Mat_<unsigned char> a) {
  // Ajusta para bbox. Se nao consegue, faz localizou=false
  int esq=a.cols, dir=0, cima=a.rows, baixo=0; // primeiro pixel diferente de 255.
  for (int l=0; l<a.rows; l++)
    for (int c=0; c<a.cols; c++) {
      if (a(l,c)!=255) {
        if (c<esq) esq=c;
        if (dir<c) dir=c;
        if (l<cima) cima=l;
        if (baixo<l) baixo=l;
      }
    }
  Mat_<unsigned char> d;
  if (!(esq<dir && cima<baixo)) { // erro na localizacao
    localizou=false;
    d.create(nlado,nlado);
    d.setTo(128);
  } else {
    localizou=true;
    Mat_<unsigned char> roi(a, Rect(esq,cima,dir-esq+1,baixo-cima+1));
    resize(roi,d,Size(nlado,nlado),0, 0, INTER_AREA);
    }
  return d;
}

Mat_<float> MNIST::bbox(Mat_<float> a) {
  // Ajusta para bbox. Se nao consegue, faz localizou=false
  int esq=a.cols, dir=0, cima=a.rows, baixo=0; // primeiro pixel menor que 0.5.
  for (int l=0; l<a.rows; l++)
    for (int c=0; c<a.cols; c++) {
      if (a(l,c)<=0.5) {
        if (c<esq) esq=c;
        if (dir<c) dir=c;
        if (l<cima) cima=l;
        if (baixo<l) baixo=l;
      }
    }
  Mat_<float> d;
  if (!(esq<dir && cima<baixo)) { // erro na localizacao
    localizou=false;
    d.create(nlado,nlado);
    d.setTo(0.5);
  } else {
    localizou=true;
    //Mat_<unsigned char> roi(a, Rect(esq,cima,dir-esq+1,baixo-cima+1));
    Mat_<float> roi(a, Rect(esq,cima,dir-esq+1,baixo-cima+1)); //Consertei 5/11/2019
    resize(roi,d,Size(nlado,nlado),0, 0, INTER_AREA);
  }
  return d;
}

void MNIST::leX(string nomeArq, int n, vector< Mat_<unsigned char> >& X, Mat_<float>& x) {
  X.resize(n);
  for (unsigned i=0; i<X.size(); i++) X[i].create(nlado,nlado);

  FILE* arq=fopen(nomeArq.c_str(),"rb");
  if (arq==NULL) erro("Erro: Arquivo inexistente ",nomeArq);
  uint8_t b;
  Mat_<unsigned char> t(28,28),d;
  fseek(arq,16,SEEK_SET);
  for (unsigned i=0; i<X.size(); i++) {
    if (fread(t.data,28*28,1,arq)!=1) xerro1("Erro leitura "+nomeArq);
    if (inverte) t=255-t;
    if (ajustaBbox) {
      d=bbox(t);
    } else{
      if (nlado!=28) resize(t,d,Size(nlado,nlado),0, 0, INTER_AREA);
      else t.copyTo(d);
    }
    X[i]=d.clone();
  }
  fclose(arq);

  x.create(X.size(),X[0].total());
  for (int i=0; i<x.rows; i++)
    for (int j=0; j<x.cols; j++)
      x(i,j)=X[i](j)/255.0;
}

void MNIST::leY(string nomeArq, int n, vector<int>& Y, Mat_<float>& y) {
  Y.resize(n);
  y.create(n,1);
  FILE* arq=fopen(nomeArq.c_str(),"rb");
  if (arq==NULL) erro("Erro: Arquivo inexistente ",nomeArq);
  uint8_t b;
  fseek(arq,8,SEEK_SET);
  for (unsigned i=0; i<y.total(); i++) {
    if (fread(&b,1,1,arq)!=1) xerro1("Erro leitura "+nomeArq);
    Y[i]=b;
    y(i)=b;
  }
  fclose(arq);
}

void MNIST::le(string caminho, int _na, int _nq) {
  na=_na; nq=_nq;
  if (na>60000) xerro1("na>60000");
  if (nq>10000) xerro1("nq>10000");

  if (na>0) {
    leX(caminho+"/train-images-idx3-ubyte",na,AX,ax);
    leY(caminho+"/train-labels-idx1-ubyte",na,AY,ay);
  }
  if (nq>0) {
    leX(caminho+"/t10k-images-idx3-ubyte",nq,QX,qx);
    leY(caminho+"/t10k-labels-idx1-ubyte",nq,QY,qy);
    qp.create(nq,1);
  }
}

int MNIST::contaErros() {
  // conta numero de erros
  int erros=0;
  for (int l=0; l<qp.rows; l++)
    if (qp(l)!=qy(l)) erros++;
  return erros;
}

Mat_<unsigned char> MNIST::geraSaida(Mat_<unsigned char> q, int qy, int qp) {
  Mat_<unsigned char> d(28,38,192);
//   putTxt(d,0,28,to_string(qy));
//   putTxt(d,14,28,to_string(qp));
  int delta=(28-q.rows)/2;
  copia(q,d,delta,delta);
  return d;
}

Mat_<unsigned char> MNIST::geraSaidaErros(int maxErr) {
  // Gera imagem 23x38, colocando qy e qp a direita.
  int erros=contaErros();
  Mat_<unsigned char> e(28,40*min(erros,maxErr),192);
  for (int j=0, i=0; j<qp.rows; j++) {
    if (qp(j)!=qy(j)) {
      Mat_<unsigned char> t=geraSaida(QX[j],qy(j),qp(j));
      copia(t,e,0,40*i);
      i++;
      if (i>=min(erros,maxErr)) break;
    }
  }
  return e;
}

Mat_<unsigned char> MNIST::geraSaidaErros(int nl, int nc) {
  // Gera uma imagem com os primeiros nl*nc digitos classificados erradamente
  Mat_<unsigned char> e(28*nl,40*nc,192);
  int j=0;
  for (int l=0; l<nl; l++)
    for (int c=0; c<nc; c++) {
      //acha o proximo erro
      while (qp(j)==qy(j) && j<qp.rows) j++;
      if (j==qp.rows) goto saida;
      Mat_<unsigned char> t=geraSaida(QX[j],qy(j),qp(j));
      copia(t,e,28*l,40*c);
      j++;
    }
  saida:
  return e;
}
