#ifndef NAIVEBAYES_H
#define NAIVEBAYES_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Mar 3, 2016 11:05:45 AM
 * Naive Bayes
 */

#include <vector>
#include <map>
#include <stddef.h>
#include <cmath>
#include <iostream>


/**
 * class of the naive bayes algorithm
 * X_TYPE, the type of the input data, eg -> string 
 * category type is default to size_t, which must be 0...K, K is the number of type
 */
template<typename X_TYPE, typename Y_TYPE = unsigned int>
class NaiveBayes {
public:

	NaiveBayes() : n_category(0) {}

	NaiveBayes(size_t n_cat) : n_category(n_cat) {
		//
	}

	~NaiveBayes() {
		clear();
	}

	/* struct to label xj = l */
	template <typename _TYPE>
	class jl_pair {
	public:
		size_t j;
		_TYPE l;
		jl_pair(size_t _j, _TYPE _l) : j(_j), l(_l) {}
		bool operator<(const jl_pair & ct) const {    // to be the key of the std::map, operator< must be overrided
			if (j < ct.j) {
				return true;
			} 
			if (j == ct.j) {
				if (l < ct.l) {
					return true;
				}
			}
			return false;
		}
	};

	void add(const std::vector<X_TYPE> &input, const Y_TYPE &cat) {
		training_vec.push_back(input);
		category_vec.push_back(cat);
	}

	void train();

	Y_TYPE classify(const std::vector<X_TYPE> &input);

	void setMaxC(Y_TYPE c) {
		n_category = c;
	}

	void clear() {
		training_vec.clear();
		category_vec.clear();
		N_Yk_vec.clear();
		P_Yk_vec.clear();
		P_Yk_Xjl_vec.clear();
		//vocab_map.clear();
	}

	void print() const;

	void print_nodes() const;

private:
	Y_TYPE n_category;
	std::vector< std::vector< X_TYPE > > training_vec;     /* vector of the input labeled data */
	std::vector<Y_TYPE> category_vec;                      /* the label of the training data */

	std::vector<size_t> N_Yk_vec;              /* number of the input which is labeled to category k */
	std::vector<double> P_Yk_vec;              /* the probability of the category k */

	typedef std::pair<jl_pair<Y_TYPE>, double> PjlTuple;
	std::vector< std::map<jl_pair<Y_TYPE>, double> > P_Yk_Xjl_vec;       /* the probability of conditional probability, xj = ajl */

	//std::map<X_TYPE, int> vocab_map;       /* map of the vocabulary */
};

template<typename X_TYPE, typename Y_TYPE>
Y_TYPE NaiveBayes<X_TYPE, Y_TYPE>::classify(const std::vector<X_TYPE> &input) {
	train();
	// calculate the probability of each class(category) */
	typename std::vector<std::vector< X_TYPE > >::size_type x_dim = training_vec.at(0).size();
	typename std::map<jl_pair<Y_TYPE>, double>::const_iterator p_it;
	Y_TYPE ck;
	double max_ck_pro;
	bool first_ck = true;
	for(Y_TYPE i=0; i<n_category; ++i) {
		double sum_pro = log(P_Yk_vec.at(i));    /* be aware of that log(P_Yk_vec.at(i)) is less than 0; */
		for(size_t j=0; j<x_dim; ++j) {
			jl_pair<Y_TYPE> jl(j, input.at(j));
			p_it = P_Yk_Xjl_vec.at(i).find(jl);
			if (p_it == P_Yk_Xjl_vec.at(i).end()) {
				// std::cout<<"error: uncatched value met"<<std::endl;
				continue;
			}
			if (0 != p_it->second) {
				sum_pro += log(p_it->second);  /* log is used to avoid the underflow of the double value */
			}
		}
		if (first_ck) {
			max_ck_pro = sum_pro;
			ck = static_cast<Y_TYPE>(i);
			first_ck = false;
			continue;
		}
		if (sum_pro > max_ck_pro) {
			max_ck_pro = sum_pro;
			ck = static_cast<Y_TYPE>(i);
		}
	}
	return ck;
}


template<typename X_TYPE, typename Y_TYPE>
void NaiveBayes<X_TYPE, Y_TYPE>::train() {
	N_Yk_vec.resize(n_category, 0);
	P_Yk_vec.resize(n_category, 0);
	P_Yk_Xjl_vec.resize(n_category, std::map<jl_pair<Y_TYPE>, double>());

	std::vector<size_t>::size_type c_size = category_vec.size();
	typename std::vector<std::vector< X_TYPE > >::size_type x_dim = training_vec.at(0).size();

	// calculate the N_Yk
	for(std::vector<size_t>::size_type i=0; i<c_size; i++) {
		N_Yk_vec[category_vec.at(i)] += 1;
	}
	// calculate the P_Yk
	for(std::vector<double>::size_type i=0; i<static_cast<std::vector<double>::size_type>(n_category); i++) {
		P_Yk_vec[i] = static_cast<double>(N_Yk_vec[i]) / c_size;   /* static_cast<double> is for type convertion, and /* or Laplace smoothing may be used */ 
	}
	// calculate the P_Yk_Xjl, first to calculate the number of each, and then calculate the percentage
	std::vector< std::map<jl_pair<Y_TYPE>, size_t> > N_Yk_Xjl_vec(n_category, std::map<jl_pair<Y_TYPE>, size_t>());    
	typename std::map<jl_pair<Y_TYPE>, size_t>::iterator m_it;
	for(typename std::vector<std::vector< X_TYPE > >::size_type i=0; i<c_size; ++i) {
		// iterate the input data vector
		Y_TYPE ck = category_vec.at(i);
		//std::map<jl_pair<Y_TYPE>, size_t> tmp_map = N_Yk_Xjl_vec.at(l);
		for(typename std::vector< X_TYPE >::size_type j=0; j<x_dim; ++j) {
			X_TYPE l = training_vec.at(i).at(j);
			jl_pair<Y_TYPE> jl(j, l);
			m_it = N_Yk_Xjl_vec.at(ck).find(jl);
			if(m_it == (N_Yk_Xjl_vec.at(ck)).end()) { // first added
				N_Yk_Xjl_vec.at(ck).insert(std::pair<jl_pair<Y_TYPE>, size_t>(jl, 1));
			} else {
				N_Yk_Xjl_vec.at(ck)[jl] = m_it->second + 1;
			}
		}
	}
	// calculate the percentage
	for(size_t ck=0; ck<n_category; ++ck) {
		for(m_it=N_Yk_Xjl_vec.at(ck).begin(); m_it!=N_Yk_Xjl_vec.at(ck).end(); ++m_it) {
			// iterate the calculated map
			P_Yk_Xjl_vec.at(ck).insert(PjlTuple( m_it->first, static_cast<double>(m_it->second) / N_Yk_vec.at(ck) ));   /* or Laplace smoothing may be used */
		}
	}

}

template<typename X_TYPE, typename Y_TYPE>
void NaiveBayes<X_TYPE, Y_TYPE>::print() const {
	std::cout<<"N_Yk_vec"<<std::endl;
	for(std::vector<size_t>::size_type i=0; i<n_category; ++i) {
		std::cout<<"C"<<i<<"-"<<N_Yk_vec[i]<<"|";
	}
	std::cout<<std::endl;

	std::cout<<"P_Yk_vec"<<std::endl;
	for(std::vector<double>::size_type i=0; i<n_category; ++i) {
		std::cout<<"C"<<i<<"-"<<P_Yk_vec[i]<<"|";
	}
	std::cout<<std::endl;

	std::cout<<"P_Yk_Xjl_vec"<<std::endl;
	typename std::map<jl_pair<Y_TYPE>, double>::const_iterator p_it;   // the type must be const_iterator
	for(size_t ck=0; ck<n_category; ++ck) {
		std::cout<<"catetory->"<<ck<<std::endl;
		for(p_it=P_Yk_Xjl_vec.at(ck).begin(); p_it!=P_Yk_Xjl_vec.at(ck).end(); ++p_it) {
			std::cout<<" (j->"<<p_it->first.j<<",l->"<<p_it->first.l<<")=>p->"<<p_it->second<<"|";
		}
		std::cout<<std::endl;
	}
}

template<typename X_TYPE, typename Y_TYPE>
void NaiveBayes<X_TYPE, Y_TYPE>::print_nodes() const {
	typename std::vector<Y_TYPE>::size_type cat_size = category_vec.size();
	if (cat_size <= 0)
		return;
	std::cout<<cat_size<<std::endl;
	typename std::vector<X_TYPE>::size_type dim = training_vec.at(0).size();
	for(typename std::vector<Y_TYPE>::size_type i=0; i<cat_size; ++i) {
		std::cout<<i<<" | input-> ";
		for(typename std::vector<X_TYPE>::size_type j=0; j<dim; ++j) {
			std::cout<<training_vec.at(i).at(j)<<"-";
		}
		std::cout<<" category-> "<<category_vec.at(i)<<std::endl;
	}
	std::cout<<std::endl;
}


/**  // test 
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
*/

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */