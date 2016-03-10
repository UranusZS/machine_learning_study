#include <iostream>
#include <vector>
#include "Utils.h"
#include "C45DecisionTree.h"

using namespace std;

int main() {
    // age: 1-> Young 2 -> middle-aged 3-> elderly
    // having job: 1 -> yes 2 no
    // having house: 1 yes 2 no
    // credit conditions: 1 -> normal 2 -> good 3 very good
    // class: 1 -> yes 2 no

    vector<Feature> age_vec;
    age_vec.push_back(1); age_vec.push_back(2); age_vec.push_back(3);
    vector<Feature> job_vec;
    job_vec.push_back(1); job_vec.push_back(2);
    vector<Feature> house_vec;
    house_vec.push_back(1); house_vec.push_back(2);
    vector<Feature> credit_vec;
    credit_vec.push_back(1); credit_vec.push_back(2); credit_vec.push_back(3);

    vector< vector<Feature> > feature_table;
    feature_table.push_back(age_vec);
    feature_table.push_back(job_vec);
    feature_table.push_back(house_vec);
    feature_table.push_back(credit_vec);

    const unsigned att_num = 4;
    const unsigned rule_num = 15;
    // the last colomn is the label
    int train_data[rule_num][att_num + 1] = {    
                        {1, 2, 2, 1, 2},
                        {1, 2, 2, 2, 2},
                        {1, 1, 2, 2, 1},
                        {1, 1, 1, 1, 1},
                        {1, 2, 2, 1, 2},

                        {2, 1, 2, 1, 2},
                        {2, 1, 2, 2, 2},
                        {2, 2, 1, 2, 1},
                        {2, 1, 1, 3, 1},
                        {2, 1, 1, 3, 1},

                        {3, 1, 1, 3, 1},
                        {3, 1, 1, 2, 1},
                        {3, 2, 2, 2, 1},
                        {3, 2, 2, 3, 1},
                        {3, 1, 2, 1, 2}
                    };

    // get init vector
    vector< vector< Feature > > input_vec;
    vector<Label> label_vec;
    for(unsigned int i=0; i<rule_num; ++i) {
        vector< Feature > input_tmp;
        for(unsigned int j=0; j<att_num; ++j) {
            Feature f_tmp;
            f_tmp.set(train_data[i][j]);
            input_tmp.push_back(f_tmp);
        }
        input_vec.push_back(input_tmp);
        label_vec.push_back(Label(train_data[i][att_num]));
    }

    // test print
    auto size = input_vec.size();
    for(auto i=0; i<size; ++i) {
        cout<<i+1<<endl;
        for(auto j=0; j<input_vec.at(i).size(); ++j) {
            cout<<"    "<<input_vec.at(i).at(j)<<endl;
        }
        cout<<"    "<<label_vec.at(i)<<endl;
    }

    // init data
    C45DecisionTree c45tree;

    for(auto i=0; i<feature_table.size(); ++i) {
        c45tree.addFeature(feature_table.at(i));
    }

    for(auto i=0; i<input_vec.size(); ++i) {
        c45tree.addData(input_vec.at(i), label_vec.at(i));
    }

    cout<<endl;
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
