/*************************************************************************
	> File Name: adaboost.h
	> Author: ZS
	> Mail: dragon_201209@126.com
    @date April 7, 2016 10:45:49 PM
 ************************************************************************/

#ifndef ADABOOST_H
#define ADABOOST_H

#include "element.h"
#include "iostream"
#include <vector>
#include <algorithm>

class AdaBoost {
public:
    AdaBoost();
    virtual ~AdaBoost();

    void set_boosting_type(int boosting_type);
    void init();

    void add_record(Record &record);
    void sorted_sample(bool verbose = false);
    void set_record(std::vector<Record> &record_vec);
private:
    // the boosting type: 1 for Discrete AdaBoost, 2 for Real AdaBoost, and 4 for Gentle AdaBoost perhaps
    int                 _boosting_type;

    // training samples
    size_t              _total_sample;
    std::vector<Record> _sample_vec;
       
    // formated training data 
    std::vector<Record> _sorted_sample_vec;
};

AdaBoost::AdaBoost() {
    //
}

AdaBoost::~AdaBoost() {
    //_sorted_sample_vec.clear();
    //_sample_vec.clear();
}

void AdaBoost::set_boosting_type(int boosting_type) {
    _boosting_type = boosting_type;
}

void AdaBoost::init() {
    _sample_vec.clear();
    _sorted_sample_vec.clear();
    _total_sample = 0;
}

void AdaBoost::add_record(Record &record) {
    _sample_vec.push_back(record);
    ++ _total_sample;
}

void AdaBoost::sorted_sample(bool verbose) {
    _sorted_sample_vec.resize(_total_sample);

    for (size_t i = 0; i < _total_sample; ++i) {    // iterate all samples
        _sorted_sample_vec[i] = _sample_vec[i];     // copied one by one, in order to avoid the memory leak
        std::sort(_sorted_sample_vec[i].input.begin(), _sorted_sample_vec[i].input.end());
    }

    if (verbose) {
        for (size_t i = 0; i < _total_sample; ++i) {
            std::cout<<i<<std::endl;
            std::cout<<_sorted_sample_vec[i].label;
            for (size_t j = 0; j < _sorted_sample_vec[i].input.size(); ++j) {
                std::cout<<"    "<<_sorted_sample_vec[i].input[j].index<<":"<<_sorted_sample_vec[i].input[j].value;
            }
        }
    }
}

void AdaBoost::set_record(std::vector<Record> &record_vec) {
    _sample_vec.clear();

    _total_sample      = record_vec.size();
    _sample_vec        = record_vec;         
}

/** // test
    #include "adaboost.h"
    #include "read_data.h"
    #include "element.h"
    #include <iostream>
    #include <string>
    #include <vector>

    using namespace std;

    int main() {

        string filename = "./data/train.data";
        vector< Record > record_vec;
        read_problem(filename, record_vec);

        AdaBoost ada;

        ada.set_record(record_vec);
        ada.sorted_sample(true);

        size_t size = record_vec.size();
        for(size_t i = 0; i < size; ++i) {
            cout<<"The "<<(i+1)<<" line:"<<endl;
            cout<<"    The label is "<<record_vec.at(i).label<<endl;
            cout<<"    The input is ";
            for(size_t j = 0; j < record_vec.at(i).input.size(); ++j) {
                cout<<(record_vec[i]).input[j].index<<":"<<(record_vec[i]).input[j].value<<"    ";
            }
            cout<<endl;
        }

        return 0;
    }
*/

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
