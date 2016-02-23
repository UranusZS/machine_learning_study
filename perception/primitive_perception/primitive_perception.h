#ifndef PRIMITIVEPERCEPTION_H
#define PRIMITIVEPERCEPTION_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 3, 2016 4:05:45 PM
 * perception
 */
 
#include <iostream>
#include <vector>
#include <algorithm>
#include "utils.h"

/**
 * This class discrips the primitive form of the model of perception.
 * You can use is by steps of :
 * init and init weight
 * train with given supervised data
 * predict and output model
 */
class PrimitivePerception {
public:
    PrimitivePerception(unsigned int size=2, double b=1, double learnRate=1.0, int iterations=100);
    ~PrimitivePerception();
    bool train(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y);
    int predict(std::vector<double>& x);
    std::vector<int> predict(std::vector<std::vector<double> >& data_x);
    void printPerceptronModel();
    bool initWeight(std::vector<double> Omega, double b);
protected:
    bool update(std::vector<double>& x, double y);
    bool initWeight(unsigned int size);
private:
    // Maximum Iteration
    int max_iterations;         
    // model f(x) = sign(wx+b), Omega(w) is a vector
    std::vector<double> w;
    double b;
    // learning rate
    double eta;
};

#endif

