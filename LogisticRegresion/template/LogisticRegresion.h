#ifndef LOGISTICREGRESSION_H
#define LOGISTICREGRESSION_H


/**
 * one realization of LR algorithm, classifier of two classes
 * @author ZS (dragon_201209@126.com)
 * @date Mar 28, 2016 4:25:45 PM
 * lr
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>    // std::shuffle
#include <random>       // std::default_random_engine

//模板函数：将string类型变量转换为常用的数值类型
template <typename Type>
Type stringToNum(const std::string& str, Type &res) {
    std::istringstream iss(str);
    iss >> res;
    return res;    
}

/**
 * lr_node for sparse feature
 **/
typedef struct _lr_node {
public:
    size_t index;
    double value;
    _lr_node() {}
    _lr_node(std::string str);
    bool set_from_str(std::string str, const char* delimiter = ":");
} lr_node;

lr_node::_lr_node(std::string str) {
    set_from_str(str);
}

bool lr_node::set_from_str(std::string str, const char* delimiter) {
    size_t d_pos = str.find(delimiter);
    if (d_pos == std::string::npos) {
        return false;
    }
    index = stringToNum(str.substr(0, d_pos), index);
    value = stringToNum(str.substr(d_pos + 1), value);
    return true;
}

class LogisticRegression {
public:
    LogisticRegression(double lam = 0.1);
    ~LogisticRegression();

    void save_model(std::string model_file);            // save model to file
    bool load_model(std::string model_file);            // load model from file
    bool load_training_file(std::string labeled_file);  // load labeled training data from file

    bool sgd_train();
    void predict();
    void test();

    void print_omega();

protected:
    void init_omega(double init_value = 0.0);
    void update_omega(double rate, std::vector<lr_node> &x);
    static double inner_dot(std::vector<lr_node> &x, std::vector<lr_node> &y);
    static double inner_dot(std::vector<lr_node> &x, std::vector<double> &o);
    static bool read_formated_file(std::string formated_file, std::vector< std::vector<lr_node> > &data_input, std::vector<size_t> &given_label, size_t &input_size, bool load_size = true);
    static double sigmoid(double x);
    static void string_split(std::string terms_str, std::string spliting_tag, std::vector<std::string> &token_vec);
private:
    // the sample input and label
    std::vector< std::vector<lr_node> >   input_vec;     // labeled input data
    std::vector<size_t>                   label_vec;     // label of input data
    size_t                                input_size;    // the max id of input data, or the dimension of the input
    size_t                                label_size;    // the number of classes, of the maximum class label
    // the parameters of the model
    std::vector<double> omega;                           // the LR parameters of omega
    double lamda;

    // const parameters 
    static const long double LOG_LIM;
};
const long double LogisticRegression::LOG_LIM = 0.00000000000000000001;

LogisticRegression::LogisticRegression(double lam) : lamda(lam) {
    //
}

LogisticRegression::~LogisticRegression() {
    //
}

/**
 * Stochastic Gradient Descent (SGD) Optimization
 */
bool LogisticRegression::sgd_train() {
    if (input_vec.size() != label_vec.size()) {
        return false;
    }
    int loop = 100;
    size_t labeled_size = input_vec.size();

    std::vector<size_t> index_vec(labeled_size);
    for (size_t i = 0; i < labeled_size; ++i) {
        index_vec[i] = i;
    }

    // Stochastic Gradient Descent (SGD) Optimization
    for (int i = 0; i < loop; ++i) {       // the loop
        auto engine = std::default_random_engine{};
        std::shuffle(std::begin(index_vec), std::end(index_vec), engine);
        for (size_t j = 0; j < labeled_size; ++j) {  // stochastic gradient ascent
            size_t index = index_vec.at(j);

            size_t label = label_vec.at(index);
            size_t predict = sigmoid(inner_dot(input_vec.at(index), omega));
            double rate = lamda * (label - predict);
            update_omega(rate, input_vec.at(index));
            // tmp += alpha * data[j].x[i] * (data[j].y - Sigmoid(data[j], w))
        } // end of inner for
    } // end of outer for

    return true;
}

/**
 * update omega
 */
void LogisticRegression::update_omega(double rate, std::vector<lr_node> &x) {
    size_t x_size = x.size();
    size_t o_size = omega.size();

    size_t x_index = 0;
    size_t o_index = 0;

    while(x_index < x_size && o_index < o_size) {
        if(x[x_index].index == o_index) {
            omega[o_index] += x[x_index].value * rate;
            ++x_index;
            ++o_index;
            continue;
        }
        if(x[x_index].index < o_index) {
            ++x_index;
        } else {
            ++o_index;
        }
    }

}

double LogisticRegression::sigmoid(double x) {
    double sgmd = 1.0 / (1.0 + exp(-x));
    return sgmd;
}

double LogisticRegression::inner_dot(std::vector<lr_node> &x, std::vector<double> &o) {
    double sum = 0;

    size_t x_size = x.size();
    size_t o_size = o.size();

    size_t x_index = 0;
    size_t o_index = 0;
    while(x_index < x_size && o_index < o_size) {
        if(x[x_index].index == o_index) {
            sum += x[x_index].value * o.at(o_index);
            ++x_index;
            ++o_index;
            continue;
        }
        if(x[x_index].index < o_index) {
            ++x_index;
        } else {
            ++o_index;
        }
    }

    return sum;
}

/**
 * calculate the inner dot of the two input vector
 */
double LogisticRegression::inner_dot(std::vector<lr_node> &x, std::vector<lr_node> &y) {
    double sum = 0;

    size_t x_size = x.size();
    size_t y_size = y.size();

    size_t x_index = 0;
    size_t y_index = 0;
    while(x_index < x_size && y_index < y_size) {
        if(x[x_index].index == y[y_index].index) {
            sum += x[x_index].value * y[y_index].value;
            ++x_index;
            ++y_index;
            continue;
        }
        if(x[x_index].index < y[y_index].index) {
            ++x_index;
        } else {
            ++y_index;
        }
    }

    return sum;
}

/**
 * save model from file of model_file
 */
void LogisticRegression::save_model(std::string model_file) { 
    //std::cout << "Saving model..." << std::endl;
    std::ofstream fout(model_file.c_str());
    for (size_t k = 0; k < input_size; k++) {
        fout << omega[k] << " ";
    }
    fout << std::endl;
    fout.close();
}

/**
 * load model from file of model_file
 */
bool LogisticRegression::load_model(std::string model_file) {
    //std::cout << "Loading model..." << std::endl;
    omega.clear();
    std::ifstream fin(model_file.c_str());
    if(!fin) {
        std::cerr << "Error opening file: " << model_file << std::endl;
       return false;
    }

    std::string line_str;
    while (getline(fin, line_str)) {
        std::vector<std::string> line_vec;
        string_split(line_str, " ", line_vec);
        for (std::vector<std::string>::iterator it = line_vec.begin(); it != line_vec.end(); it++) {
            omega.push_back(atof(it->c_str()));
        }
    }

    input_size = omega.size();
    label_size = 2;

    fin.close();
    return true;
}

/**
 * the format of the formated file is: 
 *     label, eg 0
 *     input, eg 1:4 5:9
 * 
 */
bool LogisticRegression::read_formated_file(std::string formated_file, std::vector< std::vector<lr_node> > &data_input, std::vector<size_t> &given_label, size_t &input_size, bool load_size) {
    std::ifstream fin(formated_file.c_str());
    if(!fin) {
        std::cerr << "Error opening file: " << formated_file << std::endl;
        return false;
    }

    size_t max_label_size = 0;
    size_t max_input_size = 0;
    std::string label_line_str;
    std::string input_line_str;

    while (getline(fin, label_line_str)) {
        // get label
        size_t label = 0;
        stringToNum(label_line_str, label);
        given_label.push_back(label);
        if(label > max_label_size) {
            max_label_size = label;
        }
        // get input
        getline(fin, input_line_str);
        std::vector<std::string> input_str_vec;
        string_split(input_line_str, " \t", input_str_vec);

        std::vector<lr_node> input;
        // for the bias of b (of w*x + b)
        lr_node bias_node; bias_node.index = 0; bias_node.value = 1;
        input.push_back(bias_node);
        size_t vec_size = input_str_vec.size();
        for(size_t i = 0; i < vec_size; ++i) {
            lr_node lr_tmp(input_str_vec.at(i));
            input.push_back(lr_tmp);
            if(lr_tmp.index > max_input_size) {
                max_input_size = lr_tmp.index;
            }
        } // end of for
        data_input.push_back(input);
    } // end of while 

    if(load_size) {
        input_size = max_input_size + 1;
    }

    fin.close();
    return true;
}

/**
 * the format of the labeled file is: 
 *     label, eg 0
 *     input, eg 1:4 5:9
 * 
 */
bool LogisticRegression::load_training_file(std::string labeled_file) {
    return read_formated_file(labeled_file, input_vec, label_vec, input_size, true);
}

/**
 * init omega with the value init_value
 */
void LogisticRegression::init_omega(double init_value) {
    omega.clear();
    for (int i = 0; i < input_size; ++i) {
        omega.push_back(init_value);
    }
}

void LogisticRegression::print_omega() {
    for (int i = 0; i < omega.size(); ++i) {
        std::cout<< omega.at(i) <<std::endl;
    }   
}

/**
 * split terms_str by spliting_tag, and put result into token_vec
 */
void LogisticRegression::string_split(std::string terms_str, std::string spliting_tag, std::vector<std::string> &token_vec) {
    token_vec.clear();
    const char* delimiters = spliting_tag.c_str();
    char* term_chs = new char[terms_str.size()];
    strcpy(term_chs, terms_str.c_str());
    char* pch = strtok(term_chs, delimiters);
    while (pch != NULL) {
        token_vec.push_back(std::string(pch));
        pch = strtok (NULL, delimiters);
    }
}

#endif
 /* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
