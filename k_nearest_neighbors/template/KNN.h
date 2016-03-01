#ifndef KNN_H
#define KNN_H

#include "KDTree.h"
// #include <stddef.h> size_t
#include <map>

/**
 * @author ZS (dragon_201209@126.com)
 * @date Feb 29, 2016 4:05:45 PM
 * knn-knn
 */

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
class knn {
public:
	// get rid of annoying typename
	typedef kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM> KDTree;
	typedef typename KDTree::KDNode_ptr NodePtr;
	typedef typename KDTree::KDNode Node;
	typedef typename KDTree::KDNodes NodeVec;

	void add(NodePtr node_ptr) {
		tree.add(node_ptr);
	}

	void add(FEATURE_TYPE &feature, CATEGORY_TYPE &category) {
        NodePtr n_p(new Node(feature, category));
        tree.add(n_p);
	}

	void clear() {
		tree.clear();
	}

	CATEGORY_TYPE predict(FEATURE_TYPE feature, size_t k) {
		tree.build_tree();
		NodeVec k_nearst_result;
		search_knn(feature, k, k_nearst_result);

		CATEGORY_TYPE max_cat;
		size_t max_cat_count = 0;
		std::map<CATEGORY_TYPE, size_t> m;
		typename std::map<CATEGORY_TYPE, size_t>::iterator m_it;

		typename NodeVec::size_type size = k_nearst_result.size();
		for(typename NodeVec::size_type i=0; i<size; ++i) {

			CATEGORY_TYPE cat = k_nearst_result.at(i)->category;
			m_it = m.find(cat);
			
			// update map
			if(m_it == m.end()) { // first added
				m.insert(std::pair<CATEGORY_TYPE, size_t>(cat, 1));
				if(1 > max_cat_count) {
					max_cat_count = 1;
					max_cat = cat;
				}
			} else {              // already exist
				size_t t_count = m_it->second + 1;
				m[cat] = t_count;
				if(t_count >max_cat_count) {
					max_cat_count = t_count;
					max_cat = cat;
				}
			}
		}
		// return the category which most neighbors belong to
		return max_cat;
	}

	bool search_knn(FEATURE_TYPE feature, size_t k, NodeVec &k_nearst_result) {
		tree.build_tree();
		return tree.search_knn(feature, k, k_nearst_result);
	}

private:
	KDTree tree;
};

/**  // test
	#include <iostream>
	#include <vector>

	#include "DistUtils.h"
	#include "KDNode.h"
	#include "KDTree.h"
	#include "KNN.h"

	using namespace std;

	int main() {
	    typedef kdnode< vector<double>, int, 2 > node_2d;
	    typedef kdtree< vector<double>, int, VectorEuclideanDist<double>, 2 > tree_2d;
	    typedef knn< vector<double>, int, VectorEuclideanDist<double>, 2 > knn_2d;
	    
	    vector< vector<double> > data; vector<int> target;
	    vector<double> a; a.push_back(2); a.push_back(3);
	    data.push_back(a); target.push_back(1);
	    vector<double> b; b.push_back(5); b.push_back(4); 
	    data.push_back(b); target.push_back(1);
	    vector<double> c; c.push_back(9); c.push_back(6); 
	    data.push_back(c); target.push_back(9);
	    vector<double> d; d.push_back(4); d.push_back(7); 
	    data.push_back(d); target.push_back(4);
	    vector<double> e; e.push_back(8); e.push_back(1); 
	    data.push_back(e); target.push_back(8);
	    vector<double> f; f.push_back(7); f.push_back(2); 
	    data.push_back(f); target.push_back(7);

	    knn_2d k_n_n;

	    for(vector<double>::size_type i=0; i<data.size(); i++) {
	        //node_2d::ptr n_i(new node_2d(data.at(i), target.at(i)));
	        //k_n_n.add(n_i);
	        k_n_n.add(data.at(i), target.at(i));
	    }
	 
	    vector<double> point;
	    point.push_back(3); point.push_back(3);

	    knn_2d::NodeVec k_nearest_result;    // std::vector<KDNode_ptr>, std::vector<node_2d::ptr>
	    unsigned int K = 2;
	    bool find = k_n_n.search_knn(point, K, k_nearest_result);
	    cout<<"k_nearest_result.size->"<<k_nearest_result.size()<<endl;
	    for(knn_2d::NodeVec::size_type i =0; i<k_nearest_result.size(); ++i) {
	        // cout<<k_nearest_result.at(i).use_count()<<endl;
	        k_nearest_result.at(i)->print();
	    }

	    cout<<"the predict is: "<<k_n_n.predict(point, K)<<endl;
	}
*/
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */