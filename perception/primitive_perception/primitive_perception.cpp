#include "primitive_perception.h"
#include "utils.h"

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 3, 2016 4:05:45 PM
 * perception
 */

/**
 * constructor of  class PrimitivePerception
 * @param  unsigned int size, the size of omega or input data
 * @param  double b, initial b of the model 
 * @param  double learnRate, learning rate
 * @param  int iterations, max iteration
 */
PrimitivePerception::PrimitivePerception(unsigned int size, double b, double learnRate, int iterations) {
    w.resize(size);
    this->b = b;
    eta = learnRate;
    max_iterations = iterations;
}
PrimitivePerception::~PrimitivePerception() {
    w.clear();
}
    
/**
 * @param  std::vector<std::vector<double> >& train_x, the vector of supervised data inputs, each of which is one input
 * @param  std::vector<int>& train_y, the vector of supervised data outputs, each of which is one output of input related by subscript
 * @return bool if error occurs, return false
 */
bool PrimitivePerception::train(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y) {
    if(train_x.size() != train_y.size() || 0 == train_x.size()) {
        return false;
    }
    initWeight(train_x[0].size());

    unsigned int size = train_x.size();
    for(int iter = 0 ; iter  < max_iterations; ++ iter){     // round
        bool flag = true;
        for(int i = 0; i < size;){
            if( (dot_product(w,train_x[i]) + b) * train_y[i] <= 0) {
                update(train_x[i],train_y[i]);
                flag = false;
            }else{
                ++i;     // why ?
            }
        }
        if(flag) return true;
    }
    return false;
}

/**
 * @param  std::vector<std::vector<double> >& data_x, the input data to predict, each row is one input
 * @return std::vector<int> each of which is the predict output after training 
 */
std::vector<int> PrimitivePerception::predict(std::vector<std::vector<double> >& data_x) {
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
int PrimitivePerception::predict(std::vector<double>& x){
    return sign(dot_product(x, w) + b);
}

/**
 * print the perception model
 */
void PrimitivePerception::printPerceptronModel(){
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
bool PrimitivePerception::initWeight(unsigned int size) {
    w.resize(size);
}

/**
 * init the omega and b
 * @param  std::vector<double> Omega
 * @param  double b
 */
bool PrimitivePerception::initWeight(std::vector<double> Omega, double b) {
    unsigned size = Omega.size();
    if (size != w.size()) {
        return false;
    }
    for(int i = 1; i < size; ++ i) {
        w[i] = Omega.at(i);
    }
    this->b = b;
    return true;
}

/**
 * update omega and b
 * @param  std::vector<double>& x, input data
 * @param  double y, the output of the input
 */
bool PrimitivePerception::update(std::vector<double>& x, double y) {
    unsigned size = x.size();
    if (size != w.size()) {
        return false;
    }
    for(int i = 0 ; i < size; ++ i){
        w[i] += eta*y*x[i];
    }
    b += eta*y;
    return true;
/*
    for(int i = 0 ; i < w.size(); ++ i)
        std::cout<<w[i]<<",";
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
   
    PrimitivePerception *model = new PrimitivePerception(2, 1, 1.0, 100);
    model->train(test_x,test_y);
    model->printPerceptronModel();
*/