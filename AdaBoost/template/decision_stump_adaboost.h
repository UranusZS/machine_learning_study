/*************************************************************************
    > File Name: decision_stump_adaboost.h
    > Author: ZS
    > Mail: dragon_201209@126.com
    > Created Time: 2016年04月09日 星期六 10时26分47秒
    @date April 9, 2016 10:26:47 AM 
 ************************************************************************/

#ifndef DECISION_STUMP_ADABOOST_H
#define DECISION_STUMP_ADABOOST_H

#include "decision_stump.h"
#include "adaboost.h"
#include "constant.h"
#include <vector>
#include <cmath>
#include <iostream>

class DecisionStumpAdaBoost : public AdaBoost {
public:

    DecisionStumpAdaBoost();
    ~DecisionStumpAdaBoost();

    // init the weak classifier vector
    void init_weak_classifier();
    // learn the weak classifier
    void weak_classifier_learn(DecisionStump & stump);
    // calculate the sumation of the weights
    void calculate_weight_sum(double &weight_sum, double &weight_label_sum, double &positive_weight_sum, double &negative_weight_sum);


private:
    std::vector<DecisionStump> _weak_classifier_vec;
};

DecisionStumpAdaBoost::DecisionStumpAdaBoost() {
   // 
} 

DecisionStumpAdaBoost::~DecisionStumpAdaBoost() {
    //
}

void DecisionStumpAdaBoost::init_weak_classifier() {
    _weak_classifier_vec.clear();
}

void DecisionStumpAdaBoost::weak_classifier_learn(DecisionStump &stump) {
    //
}

 
void DesisionStumpAdaboost::calculate_weight_sum(double &weight_sum, double &weight_label_sum, double &positive_weight_sum, double &negative_weight_sum) {
    weight_sum          = 0.0;
    weight_label_sum    = 0.0;
    positive_weight_sum = 0.0;
    negative_weight_sum = 0.0;

    size_t total_sample = this->get_total_sample();
    for (size_t i = 0; i < total_sample; ++i) {
        double w = this->get_weight_by_index(i);
        weight_sum += w;
        Record record = this->get_sorted_record_by_index(i);
        if (POSITIVE == record.label) {
            weight_label_sum    += w;
            positive_weight_sum += w
        } else {
            weight_label_sum    -= w;
            negative_weight_sum += w;
        }
    }
}




#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
