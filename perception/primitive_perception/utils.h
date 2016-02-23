#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <algorithm>

#define random(x) (rand()%(x))
#define sign(x) ((x > 0) ? 1: -1)

//向量的点积
double dot_product(std::vector<double>& a, std::vector<double>& b); 

#endif
