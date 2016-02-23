#include "dual_perception.h"
#include "utils.h"

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 3, 2016 4:05:45 PM
 * perception
 */

/**
 * constructor of  class DualPerception
 * @param  unsigned int size, the size of omega or input data
 * @param  double b, initial b of the model 
 * @param  double learnRate, learning rate
 * @param  int iterations, max iteration
 */
DualPerception::DualPerception(double b, double learnRate, int iterations) {
    this->b = b;
    eta = learnRate;
    max_iterations = iterations;
}
DualPerception::~DualPerception() {
    a.clear();
    w.clear();
}
    
/**
 * @param  std::vector<std::vector<double> >& train_x, the vector of supervised data inputs, each of which is one input
 * @param  std::vector<int>& train_y, the vector of supervised data outputs, each of which is one output of input related by subscript
 * @return bool if error occurs, return false
 */
bool DualPerception::train(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y) {
    unsigned int N = train_x.size();   // the size of input data vector
    if(train_x.size() != train_y.size() || 0 == N) {
        return false;
    }
    initWeight(N);
    std::vector<std::vector<double> > gram = calulateGram(train_x);

    unsigned int size = train_x.size();
    for(int iter = 0 ; iter  < max_iterations; ++ iter) {     // round
        bool flag = true;
        for(int i = 0; i < N; i++) {
            if (updateCondition(train_x, train_y, i)) {
                flag = false;
                update(i, train_y);
            }
        }
        if(flag) {
            generateModel(train_x, train_y);
            return true;
        }
    }
    return false;
}

bool DualPerception::updateCondition(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y, unsigned int index) {
    double y_tmp = b;
    unsigned int size_alpha = a.size();
    for (int j=0; j<size_alpha; j++) {
        y_tmp += a.at(j) * train_y.at(j) * gram_matrics[j][index];
    }
    
    if ( (y_tmp * train_y.at(index)) <= 0 ) {
        return true;
    }
    return false;
}

/**
 * @param  std::vector<std::vector<double> >& data_x, the input data to predict, each row is one input
 * @return std::vector<int> each of which is the predict output after training 
 */
std::vector<int> DualPerception::predict(std::vector<std::vector<double> >& data_x) {
    std::vector<int> ret;
    for(int i = 0 ; i < data_x.size(); ++ i){
        ret.push_back(predict(data_x[i]));
    }
    return ret;
}

/**
 * @param  std::vector<double>& x, the input data to predict, each row is one input
 * @return int the predict output after training 
 */
int DualPerception::predict(std::vector<double>& x) {
    return sign(dot_product(x, w) + b);
}

void DualPerception::generateModel(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y) {
    unsigned int alpha_size = train_x.size();
    unsigned int omega_size = train_x[0].size();
    w.clear();
    w.resize(omega_size);
    // b = 0;
    for (int i = 0; i < alpha_size; ++i) {
        // alphai * yi * xi
        for(int j=0; j<omega_size; ++j) {
            w[j] += a[i] * train_y[i] * train_x[i][j];
        }
        // b += a[i] * train_y[i];
    }
}

/**
 * print the perception model
 */
void DualPerception::printPerceptronModel() {
    std::cout<<"The primitive form of the perception model is :"<<std::endl;
    std::cout<<"    f(x)=sign(";
    for(int i = 0 ; i < w.size(); ++ i){
        if( i ) std::cout<<"+";
        if(1!=w[i] && 0!=w[i]) std::cout<<w[i];
        std::cout<<"x"<<i+1;
    }
    if(b > 0) std::cout<<"+";
    std::cout<<b<<")"<<std::endl;
}

/**
 * init the size of omega, which is equal to the input data as well
 * @param  unsigned int size
 */
bool DualPerception::initWeight(unsigned int size) {
    a.resize(size);
}

/**
 * init the omega and b
 * @param  std::vector<double> Omega
 * @param  double b
 */
bool DualPerception::initWeight(std::vector<double> alpha, double b) {
    unsigned size = alpha.size();
    if (size != a.size()) {
        return false;
    }
    for(int i = 1; i < size; ++ i){
        a[i] = alpha.at(i);
    }
    this->b = b;
    return true;
}

/**
 * calculate the gram matrics of input datas
 * @param  std::vector<std::vector<double> >& train_x, the input train data vector
 * @return std::vector<std::vector<double> >, the gram matrics of input data
 */
std::vector<std::vector<double> > DualPerception::calulateGram(std::vector<std::vector<double> >& train_x) {
    int size = train_x.size();
    std::vector<std::vector<double> > gram(size, std::vector<double>(size, 0));
    for(int i = 0 ; i < size ; ++ i){
        for(int j = 0 ; j  < size; ++ j){
            gram[i][j] = dot_product(train_x[i], train_x[j]);
        }
    }
    this->gram_matrics.clear();
    this->gram_matrics = gram;
    return this->gram_matrics;
}

/**
 * update omega and b
 * @param  std::vector<double>& x, input data
 * @param  double y, the output of the input
 */
bool DualPerception::update(unsigned int index, std::vector<int>& train_y) {
    if (index < 0 || index >= a.size()) {
        return false;
    }
    a[index] += eta;
    b += eta * train_y.at(index);
    return true;
/*
    for(int i = 0 ; i < a.size(); ++ i)
        std::cout<<a[i]<<",";
    std::cout<<std::endl;
    std::cout<<b<<std::endl;
*/
}

/** // test of primitive model
    std::vector<std::vector<double> >test_x(3);
    test_x[0].push_back(3);test_x[0].push_back(3);
    test_x[1].push_back(4);test_x[1].push_back(3);
    test_x[2].push_back(1);test_x[2].push_back(1);
    std::vector<int> test_y(3);
    test_y[0] = 1;
    test_y[1] = 1;
    test_y[2] = -1;
   
    DualPerception *model = new DualPerception(1, 1.0, 100);
    model->train(test_x,test_y);
    model->printPerceptronModel();
*/
