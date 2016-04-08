/*************************************************************************
	> File Name: decision_stump.h
	> Author: ZS
	> Mail: dragon_201209@126.com
	> @date April 7, 2016 9:25:45 PM
 ************************************************************************/

#ifndef DECISION_DUMP_H
#define DECISION_DUMP_H

#include <vector>

class DecisionStump {
public:
    // constuct and destruct function
    DecisionStump();
    ~DecisionStump();

    // set the members' value
    void set(int index, double threshold, double output_right, double output_left, double error = 0);

    double evaluate(const double feature_value) const;
    double evaluate(const std::vector<FeatureElement> &ele_vec) const;

    // return the member's value of the class
    int    index() const;
    double threshold() const;
    double output_right() const;
    double output_left() const;
    double error() const ;
private:
    int    _index;
    double _threshold;
    double _output_right;
    double _output_left;
    double _error;
};

DecisionStump::DecisionStump() : _index(-1), _error(-1.0) {
}

DecisionStump::~DecisionStump() {
}

void DecisionStump::set(int index, double threshold, double output_right, double output_left, double error) {
    _index        = index;
    _threshold    = threshold;
    _output_right = output_right;
    _output_left  = output_left;
    _error        = error;
}

double DecisionStump::evaluate(const double feature_value) const {
    if (feature_value > _threshold) {
        return _output_right;
    } 
    return _output_left;
}

double DecisionStump::evaluate(const std::vector<FeatureElement> &ele_vec) const {
    return evaluate(ele_vec.at(_index)); 
}

int DecisionStump::index() const {
    return _index;
}

double DecisionStump::threshold() const {
    return _threshold;
}

double DecisionStump::output_right() const {
    return _output_right;
}

double DecisionStump::output_left() const {
    return _output_left;
}

double DecisionStump::error() const {
    return _error;
}

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
