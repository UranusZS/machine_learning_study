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

/*    
    vector<double> a; a.push_back(2); a.push_back(2);
    //node_2d(a, 2);
    node_2d::ptr n_a(new node_2d(a, 2));
    tree.add(n_a);
*/

}
