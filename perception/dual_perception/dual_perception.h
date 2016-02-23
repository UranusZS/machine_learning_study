#ifndef DUALPERCEPTION_H
#define DUALPERCEPTION_H

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
 * This class discrips the dual form of the model of perception.
 * You can use is by steps of :
 * init and init weight
 * train with given supervised data
 * predict and output model
 */
class DualPerception {
public:
    DualPerception(double b=1, double learnRate=1.0, int iterations=100);
    ~DualPerception();
    bool train(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y);
    int predict(std::vector<double>& x);
    std::vector<int> predict(std::vector<std::vector<double> >& data_x);
    void printPerceptronModel();
    bool initWeight(std::vector<double> alpha, double b);
protected:
    bool update(unsigned int index, std::vector<int>& train_y);
    bool updateCondition(std::vector<std::vector<double> >& train_x,std::vector<int>& train_y, unsigned int index);
    std::vector<std::vector<double> > calulateGram(std::vector<std::vector<double> >& train_x);
    bool initWeight(unsigned int size);
    void generateModel(std::vector<std::vector<double> >& train_x, std::vector<int>& train_y);
private:
    // Maximum Iteration
    int max_iterations;         
    // model f(x) = sign(wx+b) = sign(v(alphaj*yj*xj...) dot x) + b), a -> alpha, b
    std::vector<double> a;

    // omaga = sum(alphai * yi * xi) | i <- 1..N
    std::vector<double> w;
    // b = sum(alphai * yi) | i <- 1..N
    double b;
    
    // learning rate
    double eta;
    // the gram matrics of input data
    std::vector<std::vector<double> > gram_matrics;
};

#endif

