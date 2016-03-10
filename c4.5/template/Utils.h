#ifndef UTILS_H
#define UTILS_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Mar 7, 2016 4:25:45 PM
 * c4.5
 */

#include <cmath>
#include <map>
#include <iostream>
#include <cassert>

typedef unsigned int Label;
typedef size_t class_id;
typedef size_t feature_id;

#define UNUSED_FEATURE_ID 0

double log2(double d) {
    return log(d) / log(2.0);
}

#define LONG_VAL   1
#define DOUBLE_VAL 2
#define STRING_VAL 4

typedef union _feature_value {
	long lval;
	double dval;
	struct {
		char* val;
		int len;
	} str;
} feature_value;

class Feature;

class Feature {
public:
	feature_value  value;          /* value of the feature */
	int            value_type;     /* value type of the feature: LONG_VAL | DOUBLE_VAL | STRING_VAL */
	Feature() {};
	Feature(int val) { set(val); }
	Feature(unsigned int val) { set(val); }
	Feature(double val) { set(val); }
	Feature(long val) { set(val); }
	Feature(char* val, int len = 0) { set(val, len); }
	/* set functions */
	void set(int val);
	void set(unsigned int val);
	void set(double val);
	void set(long val);
	void set(char* val, int len = 0);

	bool operator ==(const Feature &f) const ;
	bool operator <(const Feature &f) const;
	friend std::ostream& operator<<(std::ostream& out, const Feature& f);
};

void Feature::set(double val) { 
	value.dval = val; 
	value_type = DOUBLE_VAL; 
}
void Feature::set(int val) {
	value.lval = static_cast<long>(val);
	value_type = LONG_VAL;
}
void Feature::set(unsigned int val) {
	value.lval = static_cast<long>(val);
	value_type = LONG_VAL;
}
void Feature::set(long val) { 
	value.lval = val; 
	value_type = LONG_VAL; 
}
void Feature::set(char* val, int len) { 
	value.str.val = val; 
	value.str.len = len; 
	value_type = STRING_VAL; 
}

int strcmp(const char *s, const char *t)   
{   
    assert(s != NULL && t != NULL);   
    while(*s && *t && *s == *t)   
    {   
        ++ s;   
        ++ t;   
    }
    return (*s - *t);   
}   

bool Feature::operator <(const Feature &f) const {
	if(STRING_VAL == value_type) {
		assert((value.str.val != NULL) && (f.value.str.val != NULL));
		char *s = value.str.val;
		char *t = f.value.str.val;
	    while(*s && *t && *s == *t)   
	    {   
	        ++ s;   
	        ++ t;   
	    }
	    auto i = (*s - *t);
	    return i < 0;
	}
	if(LONG_VAL == value_type) {
		return value.lval < f.value.lval;
	}
	if(DOUBLE_VAL == value_type) {
		return value.dval < f.value.dval;
	}
	return true;
}

bool Feature::operator ==(const Feature &f) const {
	if(STRING_VAL == value_type) {
		if(value.str.len != f.value.str.len) {
			return false;
		}
		for(int i=0; i<value.str.len; ++i) {
			if(value.str.val[i] != f.value.str.val[i])
				return false;
		} 
		return true;
	}
	if(LONG_VAL == value_type) {
		return value.lval == f.value.lval;
	}
	if(DOUBLE_VAL == value_type) {
		return value.dval == f.value.dval;
	}
	return true;
}

std::ostream& operator<<(std::ostream& out, const Feature& f) {
    if(LONG_VAL == f.value_type) {
    	out<<"long->"<<f.value.lval;
    }
    if(DOUBLE_VAL == f.value_type) {
    	out<<"double->"<<f.value.lval;
    }
    if(STRING_VAL == f.value_type) {
    	out<<"string->";
    	for(int i=0; i<f.value.str.len; ++i)
    		out<<*(f.value.str.val+i);
    }
    return out;
}

/** 
 * get the max label of the input vector
 *
 */
template<typename T>
T getMaxCountT(typename std::vector< T > &label) {
	T max_label;
	size_t max_label_cnt = 0;
	typename std::map<T, size_t> label_cnt_map;
	typename std::map<T, size_t>::iterator it;
	auto size = label.size();
	for(auto i=0; i<size; ++i) {
		it = label_cnt_map.find(label.at(i));
		if (it != label_cnt_map.end()) {
			auto count = it->second + 1;
			if(count > max_label_cnt) {
				max_label_cnt = count;
				max_label = it->first;
			}
			label_cnt_map[it->first] = count;
		} else {
			label_cnt_map.insert(std::pair<T, size_t>(label.at(i), 1));
		}
	}
	return max_label;
}

/**
 * calculate the count of the label
 * std::vector< Label > &label, the label of the input data
 * std::map< Label, size_t > &c_count_map, the count map of each class of each possible value of the feature
 * 
 */
template<typename Label>
bool countLabel(std::vector< Label > &label, 
					std::map< Label, size_t > &c_count_map) {
	c_count_map.clear();
	if(label.empty()) {
		return true;
	}
	auto size = label.size();
	typename std::map< Label, size_t >::iterator c_it;	
	// calculate count of each label value, Ck
	for(auto k=0; k<size; ++k) {
		Label l = label.at(k);
		c_it = c_count_map.find(l);
		if(c_it != c_count_map.end()) {
			c_count_map[l] = c_it->second + 1;
		} else {
			c_count_map.insert(std::pair<Label, size_t>(l, 1));
		}		
	}
	return true;
}

/** 
 * calculate the count of the feature and feature label
 * std::vector< std::vector< Feature > > &input, the input data
 * std::vector< Label > &label, the label of the input data
 * feature_id feature_used, the feature to use, which is the subscript of the corresponding input
 * std::map< Feature, size_t > &f_count_map, the count map of each possible value of the feature
 * std::map< Feature, std::map< Label, size_t > > &f_c_count_map, the count of each of the corresponding feature value and class 
 */
template<typename Feature, typename Label>
bool countFeatureAndLabel(std::vector< std::vector< Feature > > &input,
						std::vector< Label > &label, 
						feature_id feature_used,
						std::map< Feature, size_t > &f_count_map,
						//std::map< Label, size_t > &c_count_map,
						std::map< Feature, std::map< Label, size_t > > &f_c_count_map) {
	if (input.size() != label.size()) {
		return false;
	}
	f_count_map.clear();
	//c_count_map.clear();
	f_c_count_map.clear();

	typename std::map< Feature, size_t >::iterator f_it;
	typename std::map< Feature, std::map< Label, size_t > >::iterator f_l_it;
	typename std::map< Label, size_t >::iterator l_it;

	auto size = input.size();
	// calculate count of each feature value, and calculate the count of each class the feature belong to, named as Di and Dik
	for(auto i=0; i<size; ++i) {
		Feature f = input.at(i).at(feature_used);
		Label l = label.at(i);
		// calculate count of exact feature
		f_it = f_count_map.find(f);
		if(f_it != f_count_map.end()) {
			f_count_map[f] = f_it->second + 1;
		} else {
			f_count_map.insert(std::pair<Feature, size_t>(f, 1));
		}

		// calculate count of exact feature and exact label
		f_l_it = f_c_count_map.find(f);
		if(f_l_it == f_c_count_map.end()) {    // new feature
			std::map< Label, size_t > tmp_map;
			tmp_map.insert(std::pair<Label, size_t>(l, 1));
			f_c_count_map.insert(std::pair< Feature, std::map< Label, size_t > >(f, tmp_map));
			continue;
		}
		l_it = f_l_it->second.find(l);
		if(l_it == f_l_it->second.end()) {     // new label
			f_l_it->second.insert(std::pair<Label, size_t>(l, 1));
			continue;
		}
		auto fl_count = l_it->second;
		f_c_count_map[f][l] = l_it->second + 1;
	}

/**
	typename std::map< Label, size_t >::iterator c_it;	
	// calculate count of each label value, Ck
	for(auto k=0; k<size; ++k) {
		Label l = label.at(k);
		c_it = c_count_map.find(l);
		if(c_it != c_count_map.end()) {
			c_count_map[l] = c_it->second + 1;
		} else {
			c_count_map.insert(std::pair<Label, size_t>(l, 1));
		}		
	}
*/
	return true;
}

template<typename Feature, typename Label>
bool countAllFeatureAndLabel(
						std::vector< std::vector< Feature > > &input,
						std::vector< Label > &label, 
						std::vector<feature_id> feature_unused,
						std::map< feature_id, std::map< Feature, size_t > > &fv_count_map,
						std::map< Label, size_t > &c_count_map,
						std::map< feature_id, std::map< Feature, std::map< Label, size_t > > > &fv_c_count_map) {
	if (input.size() != label.size()) {
		return false;
	}

	fv_count_map.clear();
	c_count_map.clear();
	fv_c_count_map.clear();

	auto f_unused_size = feature_unused.size();
	bool result = countLabel(label, c_count_map);

	for(auto i=0; i<f_unused_size; ++i) {
		feature_id fid = feature_unused.at(i);
		if(0 == fid) {
			continue;
		}
		std::map< Feature, size_t > f_count_map_tmp;
		std::map< Feature, std::map< Label, size_t > > f_c_count_map_tmp;
		result = countFeatureAndLabel(input, label, fid, f_count_map_tmp, f_c_count_map_tmp) && result;
		
		fv_count_map.insert(std::pair< feature_id, std::map< Feature, size_t > >(fid, f_count_map_tmp));
		fv_c_count_map.insert(std::pair< feature_id, std::map< Feature, std::map< Label, size_t > > >(fid, f_c_count_map_tmp));
	}

	return result;
}

#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */
