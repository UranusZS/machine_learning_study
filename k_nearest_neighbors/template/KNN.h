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
	typedef kdtree<typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM> KDTree;

	void add(KDNode_ptr node_ptr) {
		tree.add(node_ptr);
	}

	void clear() {
		tree.clear();
	}

	CATEGORY_TYPE predict(FEATURE_TYPE feature, size_t k) {
		KDTree::KDNodes k_nearst_result;
		search_knn(feature, k, k_nearst_result);

		CATEGORY_TYPE max_cat;
		size_t max_cat_count = 0;
		std::map<CATEGORY_TYPE, size_t> m;
		std::map<CATEGORY_TYPE, size_t>::iterator m_it;
		for(KDNodes::size_type i=k_nearst_result.size(); i>-1; --i) {

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

	bool search_knn(FEATURE_TYPE feature, size_t k, KDTree::KDNodes &k_nearst_result) {
		return tree.search_knn(feature, k, k_nearst_result);
	}
private:
	KDTree tree;
};

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */