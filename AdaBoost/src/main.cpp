/**
 * main
 * @author ZS (dragon_201209@126.com)
 * @date April 6, 2016 15:25:45 PM
 * AdaBoost
 */

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

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */