#include "utils.h"

//向量的点积
double dot_product(std::vector<double>& a, std::vector<double>& b) {
    if(a.size() != b.size()) return 0;
    double res = 0;
    for(int i = 0 ; i < a.size(); ++ i){
        res +=a[i]*b[i];
    }
    return res;
}

