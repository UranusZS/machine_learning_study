#ifndef READ_DATA_H
#define READ_DATA_H


/**
 * to read formatted data
 * @author ZS (dragon_201209@126.com)
 * @date April 6, 2016 12:25:45 AM
 * AdaBoost
 */

#include "element.h"
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>

/**
The format of training and test data file is the same as SVM-Light(http://svmlight.joachims.org/) and libsvm(http://www.csie.ntu.edu.tw/~cjlin/libsvm). That is:

    <label> <feature>:<value> <feature>:<value> ... <feature>:<value>  
       .  
       .  
       .  

    <label> = {+1, -1}  
    <feature>: feature index (integer value starting from 1)  
    <value>: feature value (double)'

There are sample data sets at http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/.
*/


/**
 * 模板函数：将string类型变量转换为常用的数值类型
 */
template <typename Type>
Type stringToNum(const std::string& str, Type &res) {
    std::istringstream iss(str);
    iss >> res;
    return res;    
}

bool get_record(std::string &line, Record &record, char delimitter = ':');
bool read_problem(std::string &filename, std::vector< Record > &record_vec);

/**
 * read from input file
 */
bool read_problem(std::string &filename, std::vector< Record > &record_vec) {
    std::ifstream fin(filename.c_str());
    if (!fin) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line_str;
    while (getline(fin, line_str)) {
    	Record record;
    	get_record(line_str, record);
    	record_vec.push_back(record);
    }

    fin.close();
	return true;
}

/**
 * read one line, and put into record struct
 */
bool get_record(std::string &line, Record &record, char delimitter) {
	std::istringstream is(line);
	is >> record.label;

	record.input.clear();

	std::string element_str;
	while (is >> element_str) {
		int pos = element_str.find(delimitter);

		FeatureElement ele;
		stringToNum(element_str.substr(0, pos), ele.index);
		stringToNum(element_str.substr(pos + 1), ele.value);

		record.input.push_back(ele);
	}
	return true;
}

/** // test
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
*/

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */