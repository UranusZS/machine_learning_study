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

    // compute the outputs of the weak classifier with precalculated weights
    bool calculate_classifier_outputs(const double weight_label_sum_right, 
                                      double &output_right,
                                      double &output_left,
                                      const double weight_sum       = 0.0,
                                      const double weight_label_sum = 0.0,
                                      const double weight_sum_right = 0.0,
                                      const double positive_weight_sum_right = 0.0,
                                      const double negative_weight_sum_right = 0.0,
                                      const double positive_weight_sum       = 0.0,
                                      const double negative_weight_sum       = 0.0
                                );

    // compute the error with precalculated weights, and the outputs of the weak classifier
    double compute_error(const double positive_weight_sum_right,
                         const double negative_weight_sum_right,
                         const double positive_weight_sum_left,
                         const double negative_weight_sum_left,
                         const double output_right,
                         const double output_left
                    ) const;

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

bool DecisionStumpAdaBoost::calculate_classifier_outputs(const double weight_label_sum_right, 
                                                         double &output_right,
                                                         double &output_left,
                                                         const double weight_sum,
                                                         const double weight_label_sum,
                                                         const double weight_sum_right,
                                                         const double positive_weight_sum_right,
                                                         const double negative_weight_sum_right,
                                                         const double positive_weight_sum,
                                                         const double negative_weight_sum
                                                        ) {
    // different boosting types, require different parameters and return different values
    switch (this->get_boosting_type()) {
        case DISCRETE_TYPE:
            if (weight_label_sum_right > 0) {
                output_right = 1.0;
                output_left  = -1.0;
            } else {
                output_left  = 1.0;
                output_right = -1.0;
            }
            break;
        case REAL_TYPE:
            output_right = log((positive_weight_sum_right + epsilon) / (negative_weight_sum_right + epsilon)) / 2.0;
            output_left  = log((positive_weight_sum - positive_weight_sum_right + epsilon) / (negative_weight_sum - negative_weight_sum_right + epsilon)) / 2.0;
            break;
        case GENTLE_TYPE:
            output_right = weight_label_sum_right / weight_sum_right;
            output_left  = (weight_label_sum - weight_label_sum_right) / (weight_sum - weight_sum_right);
            break;
        default:
            return false;
    }
    return true;
}

double DecisionStumpAdaBoost::compute_error(const double positive_weight_sum_right,
                                            const double negative_weight_sum_right,
                                            const double positive_weight_sum_left,
                                            const double negative_weight_sum_left,
                                            const double output_right,
                                            const double output_left
                                        ) const {
    // the error 
    double error = 0.0;
    switch (this->get_boosting_type()) {
        case DISCRETE_TYPE:
            error  = positive_weight_sum_right * (1.0 - output_right) / 2.0;
            error += positive_weight_sum_left * (1.0 - output_left) / 2.0;
            error += negative_weight_sum_right * (1.0 + output_right) / 2.0;
            error += negative_weight_sum_left * (1.0 + output_left) / 2.0;
            break;
        case REAL_TYPE:
            error  = positive_weight_sum_right * exp(-output_right);
            error += positive_weight_sum_left * exp(-output_left);
            error += negative_weight_sum_right * exp(output_right);
            error += negative_weight_sum_left * exp(output_left);
            break;
        case GENTLE_TYPE:
            error  = positive_weight_sum_right * (1.0 - output_right) * (1.0 - output_right);
            error += positive_weight_sum_left * (1.0 - output_left) * (1.0 - output_left);
            error += negative_weight_sum_right * (1.0 + output_right) * (1.0 + output_right);
            error += negative_weight_sum_left * (1.0 + output_left) * (1.0 + output_left);
            break;
        default:
            return error;
    }
    return error;
}
 


#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
