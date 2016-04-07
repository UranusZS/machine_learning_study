/*************************************************************************
	> File Name: adaboost.h
	> Author: ZS
	> Mail: dragon_201209@126.com
    @date April 7, 2016 10:45:49 PM
 ************************************************************************/

#ifndef ADABOOST_H
#define ADABOOST_H

#include "element.h"
#include <vector>


class AdaBoost {
public:

private:
    int                        _boosting_type;
    size_t                     _total_feature;
    std::vector<DecisionStump> _weak_classifiers;

    // training samples
    size_t              _total_sample;
    std::vector<Record> _sample_vec;
       
    // formated training data 
    std::vector<Record> _sorted_sample_vec;

    double _weight_sum;
    double _weight_label_sum;
    double _positive_weight_sum;
    double _negative_weight_sum;
};

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
