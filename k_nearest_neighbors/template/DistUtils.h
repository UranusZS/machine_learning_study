#ifndef DISTUTILS_H
#define DISTUTILS_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 25, 2016 4:05:45 PM
 * knn-kdtree
 */

/**
 * struct(class) of distance operator
 * The KDNode will call certain function of dist of the relative class 
 * Certain class must contain public member of boolean err, and calculation function of dist
 */

#include <cmath>
#include <vector>

template<typename T>
class VectorEuclideanDist {
public:
	bool err;
	VectorEuclideanDist() : err(false) {}
	T dist (const std::vector<T> &a, const std::vector<T> &b);
	virtual ~VectorEuclideanDist() {};
};

template<typename T>
T VectorEuclideanDist<T>::dist (const std::vector<T> &a, const std::vector<T> &b) {
	if (a.size() != b.size()) {
		err = true;
		return 0;
	}
	T d = 0;
	// typename 
	typename std::vector<T>::size_type size = a.size();
	for(typename std::vector<T>::size_type i=0; i<size; ++i) {
		d += (a[i] - b[i]) * (a[i] - b[i]);
	}
	return sqrt(d);
}

/**  // test 
	#include "DistUtils.h"
	#include <iostream>
	#include <vector>

	using namespace std;

	int main() {
	    VectorEuclideanDist<double> distance;
	    vector<double> a(2,1);
	    vector<double> b(2,3);
	    cout<<"dist->"<<distance.dist(a, b)<<endl;
	}
*/
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
