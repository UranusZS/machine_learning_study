#include <iostream>
#include <vector>
#include "Utils.h"

using namespace std;

int main() {
    // age: 1-> Young 2 -> middle-aged 3-> elderly
    // having job: 1 -> yes 2 no
    // having house: 1 yes 2 no
    // credit conditions: 1 -> normal 2 -> good 3 very good
    // class: 1 -> yes 2 no

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

    cout<<endl;
}


/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
