#ifndef KDTREE_H
#define KDTREE_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 25, 2016 4:05:45 PM
 * knn-kdtree
 */

#include <memory>
#include <limits>
#include <queue>
#include <vector>

#include "KDNode.h"
#include "DistUtils.h"


template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM = 2>
class kdtree;

/**
 * FEATURE_TYPE    type of the feature
 * K_DIM           dimention of kdtree 
 * CATEGORY_TYPE   category of the kdnode, or value of the kdnode
 *
 * Dist            the class to calculate distance, eg -> VectorEuclideanDist<double> to calculate euclidean distance of vector
 */
template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
class kdtree {
public:
	kdtree() {}
	virtual ~kdtree() {}

	// get rid of annoying typename
	typedef kdnode<FEATURE_TYPE, CATEGORY_TYPE, K_DIM> KDNode;   /* node of kdtree */
	typedef typename KDNode::ptr kdnode_ptr;                               /* ptr of the node of kdtree */
	typedef std::vector<kdnode_ptr> KDNodes;                               /* vector list of the ptrs of the kdtree node */

	void add(kdnode_ptr node_ptr) {     /* add node to tree data vector */
		kd_nodes.push_back(node_ptr);
	}

	void clear() {                      /* reset data vector and clear tree */
		root.reset();
		kd_nodes.clear();
	}

	void print_nodes();
	// public function
	bool build_tree();                                                       /* build kd-tree */
	bool search_knn(KDNode &node, unsigned int k, KDNodes k_nearst_result);  /* search k nearest nodes, and store to k_nearst_result */

private:

	bool make_kdtree(kdnode_ptr node);   /* make tree  with the root node */

	kdnode_ptr root;     /* root of the kdtree */
	KDNodes kd_nodes;    /* nodes data of the kdtree */
	DISTANCE dist;       /* distance operator of the kdtree */
};

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
bool kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::make_kdtree(kdnode_ptr node) {
	return true;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
bool kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::build_tree() {
	return true;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
bool kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::search_knn(KDNode &node, unsigned int k, KDNodes k_nearst_result) {
	return true;
}

template <> 
void kdtree<typename std::vector<double>, int, VectorEuclideanDist<double>, 2>::print_nodes() {
	typename std::vector<double>::size_type size = kd_nodes.size();
	typename std::vector<double>::size_type i = 0;
	for(; i<size; ++i) {
		kd_nodes.at(i)->print();
	}
}

/**  // test 
    #include <iostream>
    #include <vector>

    #include "DistUtils.h"
    #include "KDNode.h"
    #include "KDTree.h"

    using namespace std;

    int main() {
        typedef kdnode< vector<double>, int, 2 > node_2d;
        typedef kdtree< vector<double>, int, VectorEuclideanDist<double>, 2 > tree_2d;
        tree_2d tree;

        node_2d::ptr nodeptr_2d;
        
        vector< vector<double> > data; vector<int> target;
        vector<double> a; a.push_back(2); a.push_back(3);
        data.push_back(a); target.push_back(2);
        vector<double> b; b.push_back(5); b.push_back(4); 
        data.push_back(b); target.push_back(5);
        vector<double> c; c.push_back(9); c.push_back(6); 
        data.push_back(c); target.push_back(9);
        vector<double> d; d.push_back(4); d.push_back(7); 
        data.push_back(d); target.push_back(4);
        vector<double> e; e.push_back(8); e.push_back(1); 
        data.push_back(e); target.push_back(8);
        vector<double> f; f.push_back(7); f.push_back(2); 
        data.push_back(f); target.push_back(7);

        for(vector<double>::size_type i=0; i<data.size(); i++) {
            node_2d::ptr n_i(new node_2d(data.at(i), target.at(i)));
            tree.add(n_i);
        }

        tree.print_nodes();

    }
*/
 #endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
