#include "NaiveBayes.h"
#include <iostream>

using namespace std;

int main() {
	cout<<"NaiveBayes"<<endl;

	// set X2 : S -> 0, M -> 1, L -> 2, label : -1 -> 0
	vector< vector<int> > data;
	vector<int> label;

	vector<int> a; a.push_back(1); a.push_back(0);
	data.push_back(a); label.push_back(0);
	vector<int> b; b.push_back(1); b.push_back(1);
	data.push_back(b); label.push_back(0);
	vector<int> c; c.push_back(1); c.push_back(1);
	data.push_back(c); label.push_back(1);
	vector<int> d; d.push_back(1); d.push_back(0);
	data.push_back(d); label.push_back(1);
	vector<int> e; e.push_back(1); e.push_back(0);
	data.push_back(e); label.push_back(0);
	vector<int> f; f.push_back(2); f.push_back(0);
	data.push_back(f); label.push_back(0);
	vector<int> g; g.push_back(2); g.push_back(1);
	data.push_back(g); label.push_back(0);
	vector<int> h; h.push_back(2); h.push_back(1);
	data.push_back(h); label.push_back(1);
	vector<int> i; i.push_back(2); i.push_back(2);
	data.push_back(i); label.push_back(1);
	vector<int> j; j.push_back(2); j.push_back(2);
	data.push_back(j); label.push_back(1);
	vector<int> k; k.push_back(3); k.push_back(2);
	data.push_back(k); label.push_back(1);
	vector<int> l; l.push_back(3); l.push_back(1);
	data.push_back(l); label.push_back(1);
	vector<int> m; m.push_back(3); m.push_back(1);
	data.push_back(m); label.push_back(1);
	vector<int> n; n.push_back(3); n.push_back(2);
	data.push_back(n); label.push_back(1);
	vector<int> o; o.push_back(3); o.push_back(2);
	data.push_back(o); label.push_back(0);

	NaiveBayes<int> nb(2);
	for(unsigned int i=0; i<label.size(); ++i) {
		nb.add(data.at(i), label.at(i));
	}

	nb.print_nodes();

	nb.train();

	nb.print();

	vector<int> point;
	point.push_back(2); point.push_back(0);

	int ck = nb.classify(point);
	cout<<"The classify of given data->("<<point.at(0)<<", "<<point.at(1)<<") is class "<<ck<<endl;
}

/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */