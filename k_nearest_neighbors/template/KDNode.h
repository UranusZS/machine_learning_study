#ifndef KDNODE_H
#define KDNODE_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 25, 2016 4:05:45 PM
 * knn-kdtree
 */

#include <memory>
#include <limits>
#include <iostream>

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, int K_DIM = 2>
class kdnode;

/**
 * FEATURE_TYPE    type of the feature
 * K_DIM           dimention of kdtree 
 * CATEGORY_TYPE   category of the kdnode, or value of the kdnode
 */
template <typename FEATURE_TYPE, typename CATEGORY_TYPE, int K_DIM>
class kdnode {
public:
	// data struct
	typedef std::shared_ptr< kdnode<FEATURE_TYPE, CATEGORY_TYPE, K_DIM> > ptr;
	ptr left;                /* left child */
    ptr right;               /* right child */
    int axis;                /* for which axis to divide, or ki */


    FEATURE_TYPE  feature;   /* feature of node, eg -> vector<double> */
    CATEGORY_TYPE category;  /* category of the node by the feature, eg -> int, double... */

    void print();

    // functions
    kdnode(FEATURE_TYPE f, CATEGORY_TYPE c) : feature(f), category(c), axis(0) {}
    virtual ~kdnode() {}
};

template<>
void kdnode<typename std::vector<double>, int, 2>::print() {
    std::vector<double>::size_type size = feature.size();
    std::vector<double>::size_type i = 0;
    for(; i<size; ++i) {
        std::cout<<feature.at(i)<<" ";
    }
    std::cout<<category;
    std::cout<<std::endl;
    return;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, int K_DIM>
void kdnode<FEATURE_TYPE, CATEGORY_TYPE, K_DIM>::print() {
    return;
}

 #endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
