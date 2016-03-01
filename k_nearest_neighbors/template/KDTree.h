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
#include <functional>
#include <algorithm>

#include "KDNode.h"
#include "DistUtils.h"


namespace util {       /* namespace util */

/* compare by the axis, FEATURE_TYPE must has the function of at() */
template <typename FEATURE_TYPE>
bool axis_compare(const FEATURE_TYPE &fl, const FEATURE_TYPE &fr, std::size_t dim) {
    return fl.at(dim) - fr.at(dim) < 0;
}

}                      /* namespace util */

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
    typedef kdnode<FEATURE_TYPE, CATEGORY_TYPE, K_DIM> KDNode;             /* node of kdtree */
    typedef typename KDNode::ptr KDNode_ptr;                               /* ptr of the node of kdtree */
    typedef std::vector<KDNode_ptr> KDNodes;                               /* vector list of the ptrs of the kdtree node */

    // utilities
    typedef std::pair<double, KDNode_ptr> DistanceTuple;                   /* distance tuple */

    struct SmallestOnTop {
        bool operator()(const DistanceTuple &a, const DistanceTuple &b) const {
            return a.first > b.first;
        }
    };
    struct LargestOnTop {
        bool operator()(const DistanceTuple &a, const DistanceTuple &b) const {
            return a.first < b.first;
        }
    };

    typedef std::priority_queue<DistanceTuple, std::vector<DistanceTuple>, SmallestOnTop> MinPriorityQueue;
    typedef std::priority_queue<DistanceTuple, std::vector<DistanceTuple>, LargestOnTop>  MaxPriorityQueue;
    
    template<typename NODE_TYPE>
    struct Sort : public std::binary_function<NODE_TYPE, NODE_TYPE, bool> {
        Sort(std::size_t dim) : m_dimension(dim) {}
        bool operator()(const NODE_TYPE &lhs, const NODE_TYPE &rhs) const {
            return util::axis_compare(lhs->feature, rhs->feature, m_dimension);
        }
        std::size_t m_dimension;
    };

    // public functions
    void add(KDNode_ptr node_ptr) {     /* add node to tree data vector */
        kd_nodes.push_back(node_ptr);
    }

    void clear() {                      /* reset data vector and clear tree */
        root.reset();
        kd_nodes.clear();
    }

    void print_nodes();                 /* to see the data of the kd-tree */
    
    void print_tree();                  /* to see the node relation of the kd-tree */

    bool build_tree();                                                       /* build kd-tree */

    bool search_knn(FEATURE_TYPE feature, size_t k, KDNodes &k_nearest_result);  /* search k nearest nodes, and store to k_nearst_result */

private:

    KDNode_ptr make_kdtree(KDNodes &nodes, int depth);   /* make tree  with the root node */

    void print_tree_relation(KDNode_ptr root);

    template <typename PriorityQueue>
    static void knearest(const FEATURE_TYPE &feature, const KDNode_ptr &currentNode, size_t k, DISTANCE &dist, PriorityQueue &result);

    KDNode_ptr root;     /* root of the kdtree */
    KDNodes kd_nodes;    /* nodes data of the kdtree */
    DISTANCE dist;       /* distance operator of the kdtree */

};

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
typename kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::KDNode_ptr kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::make_kdtree(KDNodes &nodes, int depth) {
    if (nodes.empty()) {
        return KDNode_ptr();
    }
    int axis = depth % K_DIM;                      /* choose the axis to split */
    typename KDNodes::size_type median = nodes.size() / 2;  /* median of the nodes */
    /* split into 2 parts */
    std::nth_element(nodes.begin(), nodes.begin() + median, nodes.end(), Sort<KDNode_ptr>(axis));
    KDNodes left_nodes(nodes.begin(), nodes.begin() + median);
    KDNodes right_nodes(nodes.begin() + median + 1, nodes.end());

/*
    // to test
    std::cout<<"left_nodes.size->"<<left_nodes.size()<<std::endl;
    std::cout<<"right_nodes.size->"<<right_nodes.size()<<std::endl;
*/

    KDNode_ptr root = nodes.at(median);
    root->axis = axis;

    root->left  = make_kdtree(left_nodes,  depth + 1);
    root->right = make_kdtree(right_nodes, depth + 1);
/*
    // to test
    if (root->left && root->right)
        std::cout<<"-root->category:"<<root->category<<"-left->category:"<<root->left->category<<"-right->category:"<<root->right->category<<std::endl;
    else if (root->left)
        std::cout<<"-root->category:"<<root->category<<"-left->category:"<<root->left->category<<std::endl;
    else if (root->right)
        std::cout<<"-root->category:"<<root->category<<"-right->category:"<<root->right->category<<std::endl;
*/

    return root;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
bool kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::build_tree() {
    if (kd_nodes.empty()) {
        return false;
    }
    root = make_kdtree(kd_nodes, 0);
    if(root) {
        return true;
    }
    return false;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
bool kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::search_knn(FEATURE_TYPE feature, size_t k, KDNodes &k_nearest_result) {
    if (!root || k <1) {
        return false;
    }
    DISTANCE distance;
    MaxPriorityQueue queue_tmp;
    knearest(feature, root, k, distance, queue_tmp);
    typename MaxPriorityQueue::size_type size = queue_tmp.size();
    k_nearest_result.resize(size);
    for(typename KDNodes::size_type i=0; i<size; ++i) {    // be careful of the unsigned type
        // reverse order, typedef std::pair<double, KDNode_ptr> DistanceTuple;
        k_nearest_result[size - i -1] = (queue_tmp.top().second);
/*        
        // to test
        std::cout<<"search_knn.[size - i -1]->"<<i<<std::endl;
        std::cout<<"result_tmp.top.second->"<<(queue_tmp.top()).second->category<<std::endl;
        std::cout<<"k_nearest_result[size - i -1].use_count->"<<k_nearest_result[size - i -1].use_count()<<std::endl;
*/
        queue_tmp.pop();
    }
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

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
template <typename PriorityQueue>
void kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::knearest(const FEATURE_TYPE &feature, const KDNode_ptr &currentNode, size_t k, DISTANCE &distance, PriorityQueue &result) {
    if (!currentNode) {
        return;
    }

    double d = distance.dist(feature, currentNode->feature);
    double d_axis = util::axis_compare(feature, currentNode->feature, currentNode->axis);

    if (result.size() < k or d <= result.top().first) {
        result.push(DistanceTuple(d, currentNode));
        if (result.size() > k) {
            result.pop();
        }
    }

/*
    // to test
    std::cout<<"currentNode->category"<<currentNode->category<<std::endl;
    std::cout<<"result.size->"<<result.size()<<"-result.top.first->"<<result.top().first<<"-result.top.second->"<<(result.top()).second->category<<std::endl;
*/

    KDNode_ptr near = (false == d_axis ? currentNode->left : currentNode->right);
    KDNode_ptr far  = (false == d_axis ? currentNode->right : currentNode->left);

    knearest(feature, near, k, distance, result);
    // why ?
    if (d_axis >= result.top().first) {
        return;
    }
    knearest(feature, far, k, distance, result);
    return;
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
void kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::print_tree() {
    print_tree_relation(root);
}

template <typename FEATURE_TYPE, typename CATEGORY_TYPE, typename DISTANCE, int K_DIM>
void kdtree<FEATURE_TYPE, CATEGORY_TYPE, DISTANCE, K_DIM>::print_tree_relation(KDNode_ptr root) {
    if (!root) 
        return;
    if (root->left && root->right)
        std::cout<<"root->axis:"<<root->axis<<" | "<<"-root->category:"<<root->category<<"-left->category:"<<root->left->category<<"-right->category:"<<root->right->category<<std::endl;
    else if (root->left)
        std::cout<<"root->axis:"<<root->axis<<" | "<<"-root->category:"<<root->category<<"-left->category:"<<root->left->category<<std::endl;
    else if (root->right)
        std::cout<<"root->axis:"<<root->axis<<" | "<<"-root->category:"<<root->category<<"-right->category:"<<root->right->category<<std::endl;

    print_tree_relation(root->left);
    print_tree_relation(root->right);
}

/** reference: 
#include <algorithm>
    #include <functional>
    #include <iostream>
    #include <vector>
     
    struct same : std::binary_function<int, int, bool>
    {
        bool operator()(int a, int b) const { return a == b; }
    };
     
    int main()
    {
        std::vector<int> v1{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        std::vector<int> v2{10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
        std::vector<bool> v3(v1.size());
     
        std::transform(v1.begin(), v1.end(), v2.begin(), v3.begin(), std::not2(same()));
     
        std::cout << std::boolalpha;
        for (std::size_t i = 0; i < v1.size(); ++i)
            std::cout << v1[i] << ' ' << v2[i] << ' ' << v3[i] << '\n';
    }
*/

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
     
        // tree.print_nodes(); // to test

        tree.build_tree();
        // tree.print_tree();  // to test, different category needed, just see the tree structrue

        vector<double> point;
        point.push_back(3); point.push_back(3);
        tree_2d::KDNodes k_nearest_result;    // std::vector<KDNode_ptr>, std::vector<node_2d::ptr>
        unsigned int K = 2;
        bool find = tree.search_knn(point, K, k_nearest_result);
        cout<<"find->"<<find<<endl;
        cout<<"k_nearest_result.size->"<<k_nearest_result.size()<<endl;
        for(tree_2d::KDNodes::size_type i =0; i<k_nearest_result.size(); ++i) {
            // cout<<k_nearest_result.at(i).use_count()<<endl;
            k_nearest_result.at(i)->print();
        }

    }
*/
 #endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */

