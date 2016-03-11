#ifndef C45DECISIONTREE_H
#define C45DECISIONTREE_H

/**
 * one realization of C4.5 algorithm
 * @author ZS (dragon_201209@126.com)
 * @date Mar 8, 2016 4:25:45 PM
 * c4.5
 */

#include <vector>
#include <map>
#include <memory>

#include "Utils.h"

/**
 * We assume that the input feature is discrete integers,
 * which can be mapped to before we build the Decision Tree
 * The tree node has three properties: 
 *         is_leaf to mark whether the node is leaf, true is leaf
 *         category to mark the class it belongs to
 *         fid to mark the feature used to classify the node
 */


/**
 * struct C45TreeNode
 * node of the C45 Tree
 * is_leaf: to mark whether the node is leaf
 * category: the class the node represent
 * fid: the feature used to classify
 * children: children of the tree node 
 */
typedef struct _C45TreeNode {
	// basic properties
	bool is_leaf;
	Feature feature;
	Label category;
	feature_id fid;
	// children
	std::vector<_C45TreeNode*> children;
	_C45TreeNode(bool leaf=false, Label cat=0, feature_id id=UNUSED_FEATURE_ID, Feature f=0) : is_leaf(leaf), category(cat), fid(id), feature(f) {}
	// functions
	void set(bool leaf, Label cat);
	void set(bool leaf, Label cat, feature_id id, Feature f=0);
	~_C45TreeNode() {}
} C45TreeNode;

void C45TreeNode::set(bool leaf, Label cat) {
	is_leaf = leaf;
	category = cat;
}

void C45TreeNode::set(bool leaf, Label cat, feature_id id, Feature f) {
	is_leaf = leaf;
	category = cat;
	fid = id;
	feature = f;
}

/**
 * class C45DecisionTree
 * decision tree of the C4.5 algorithm
 * using supervised data to build the tree and then to predict
 *
 */
class C45DecisionTree {
public:
	C45DecisionTree();
	~C45DecisionTree();
	bool train();
	bool prune();
	Label predict(std::vector<Feature> &input);
	void addData(std::vector< Feature > &input, Label &l);
	void addFeature(std::vector<Feature> &f_vec);
	void clear();

	void printData();
private:
	C45TreeNode* root;        /* the Decision Tree root */
	double epsilon;           /* threshold of the gain */

	bool makeTree(C45TreeNode* node, std::vector< std::vector< Feature > > &input, std::vector< Label > &label, std::vector<feature_id> &feature_used);

	std::vector< std::vector<Feature> > feature_table;   /* to describe feature values */
	std::vector< std::vector< Feature > > input_vec;     /* input of the labeled data */
	std::vector<Label> label_vec;                        /* label of the labeled data */
	//std::vector<feature_id> feature_vec;                 /* feature ids of the data */

	/* calculate the Info Gain Ratio of each feature split*/
	void calculateInfoGainRatio(std::map< feature_id, std::map< Feature, size_t > > &fv_count_map, std::map< Label, size_t > &c_count_map, std::map< feature_id, std::map< Feature, std::map< Label, size_t > > > &fv_c_count_map, std::vector<feature_id> &unused_feature, size_t D, std::vector<double> &result);
	/* split the input data by the given feature */
	bool splitByFeature(std::vector< std::vector< Feature > > &input, std::vector< Label > &label, feature_id f_index, std::map< Feature, std::vector< std::vector<Feature> > > &input_split_map, std::map< Feature, std::vector<Label> > &label_split_map);
};

C45DecisionTree::C45DecisionTree() {
	root = new C45TreeNode();
}
C45DecisionTree::~C45DecisionTree() {
	delete root;
	clear();
}

/**
 * predict the class of the input
 */
Label C45DecisionTree::predict(std::vector<Feature> &input) {
	feature_id fid = root->fid;
	Feature f = input.at(fid);
	// next node iterator
	if(root->is_leaf) {
		return root->category;
	}

	C45TreeNode* current_node = root;
	while(!current_node->is_leaf) {
		fid = current_node->fid;
		for(auto i=0; i<current_node->children.size(); ++i) {
			C45TreeNode* child = current_node->children.at(i);
			if(input.at(fid) == child->feature) {
				current_node=child;
			}
		}
	}

	return current_node->category;
}

/**
 * train the tree according to the given training data 
 */
bool C45DecisionTree::train() {
	std::vector<feature_id> feature_vec(feature_table.size(), 0);
	for(auto i=0; i<feature_vec.size(); ++i) {
		feature_vec[i] = i;
	}
	return makeTree(root, input_vec, label_vec, feature_vec);
}

/**
 * add data 
 */
void C45DecisionTree::addData(std::vector< Feature > &input, Label &l) {
	input_vec.push_back(input);
	label_vec.push_back(l);
	return;
}

/**
 * add feature and its possible values
 */
void C45DecisionTree::addFeature(std::vector<Feature> &f_vec) {
	feature_table.push_back(f_vec);
	return;
}

/**
 * clear the vector
 */
void C45DecisionTree::clear() {
	feature_table.clear();
	input_vec.clear();
	label_vec.clear();
}

/**
 * makeTree
 * @param C45TreeNode* node                              the current node to process
 * @param std::vector< std::vector< Feature > > &input   the input data
 * @param std::vector< Label > &label                    the label of input data
 * @param std::vector<feature_id> &feature_used          the feature can be used to classify
 * @return boolean                                       true for success build the tree
 */
bool C45DecisionTree::makeTree(C45TreeNode* node,                                       \
				std::vector< std::vector< Feature > > &input,   \
				std::vector< Label > &label,                    \
				std::vector<feature_id> &feature_used) {        
	if (input.size() != label.size()) {
		return false;
	}
	node->children.clear();
	if (input.empty()) {
		node->is_leaf = true;
		return true;
	}

	// all data belong to one class
	auto i_size = input.size();
	auto i=i_size;   // for exact type
	Label l0 = label.at(0);
	for(i=0; i<i_size; ++i) {
		if(label.at(i) != l0) {
			break;
		}
	}
	if (i == i_size) {
		node->set(true, l0);
		return true;
	}

	// unused feature is empty
	if(feature_used.empty()) {
		l0 = getMaxCountT(label);
		node->set(true, l0);
		return true;
	}

	// for all feature unused, calculate the gain
	auto f_size = feature_used.size();
	std::vector<double> feature_gain(f_size, 0.0);

	std::map< feature_id, std::map< Feature, size_t > > fv_count_map;
	std::map< Label, size_t > c_count_map;
	std::map< feature_id, std::map< Feature, std::map< Label, size_t > > > fv_c_count_map;

	countAllFeatureAndLabel(input, label, feature_used, fv_count_map, c_count_map, fv_c_count_map);
	this->calculateInfoGainRatio(fv_count_map, c_count_map, fv_c_count_map, feature_used, i_size, feature_gain);

	size_t max_gain_feature = 0;
	double max_gain = 0.0;
	for(auto i=0; i<f_size; i++) {
		if(feature_gain.at(i) > max_gain) {
			max_gain = feature_gain.at(i);
			max_gain_feature = feature_used.at(i);
		}
	}

	// if max_gain < epsilon, then single node tree get
	if(max_gain < epsilon) {
		l0 = getMaxCountT(label);
		node->set(true, l0, UNUSED_FEATURE_ID);
		return true;
	}

	// max_gain >= epsilon, build the tree
	//l0 = getMaxCountT(label);
	node->set(false, UNUSED_LABEL_ID, max_gain_feature);

	std::map< Feature, std::vector< std::vector<Feature> > > input_map;
	std::map< Feature, std::vector<Label> > label_map;

	// get the feature remained
	std::vector<feature_id> feature_remained = feature_used;
	std::vector<feature_id>::iterator v_it;
	for(v_it=feature_remained.begin(); v_it!=feature_remained.end(); ++v_it) {
		if(*v_it == max_gain_feature) {
			feature_remained.erase(v_it);
			break;
		}
	}
	// split by max_gain_feature
	splitByFeature(input, label, max_gain_feature, input_map, label_map);
	// build each subtree
	std::map< Feature, std::vector< std::vector<Feature> > >::iterator f_it;
	Feature f;
	for(f_it=input_map.begin(); f_it!=input_map.end(); ++f_it) {
		f = f_it->first;
		C45TreeNode* child = new C45TreeNode();
		child->feature = f;
		bool success = makeTree(child, input_map[f], label_map[f], feature_remained);
		if(success) {
			child->feature = f;
			node->children.push_back(child);
		}
	}
	return true;
}

/**
 * split the data into different parts according to the feature values of the given feature
 * @param std::vector< std::vector< Feature > > &input,
 * @param std::vector< Label > &label
 * @param feature_id f_index,
 * 
 * @return std::map< Feature, std::vector<Feature> > &input_split_map
 * @return std::map< Feature, std::vector<Label> > &label_split_map
 * 
 */
bool C45DecisionTree::splitByFeature(std::vector< std::vector< Feature > > &input, 
			std::vector< Label > &label,                                           
			feature_id f_index,                                                        
			std::map< Feature, std::vector< std::vector<Feature> > > &input_split_map,          
			std::map< Feature, std::vector<Label> > &label_split_map
			) {
	if(input.size() != label.size()) {
		return false;
	}

	input_split_map.clear();
	label_split_map.clear();

	std::map< Feature, std::vector< std::vector<Feature> > >::iterator f_it;
	std::map< Feature, std::vector<Label> >::iterator l_it;

	auto size = input.size();
	for(auto i=0; i<size; ++i) {
		Feature f = input.at(i).at(f_index);
		// input process & label process, for the reason of the feature
		f_it = input_split_map.find(f);
		if(f_it == input_split_map.end()) {                      // first met
			std::vector< std::vector< Feature > > f_tmp;
			f_tmp.push_back(input.at(i));
			input_split_map.insert(std::pair<Feature, std::vector< std::vector< Feature > > >(f, f_tmp));

			std::vector<Label> l_tmp;
			l_tmp.push_back(label.at(i));
			label_split_map.insert(std::pair<Feature, std::vector<Label> >(f, l_tmp));
		} else {                                                // already added
			input_split_map[f].push_back(input.at(i));
			label_split_map[f].push_back(label.at(i));
		}
	}
	return true;
}

/**
 * calculate the Info Gain Ratio of each feature
 *
 * @param std::map< feature_id, std::map< Feature, size_t > > &fv_count_map
 * @param std::map< Label, size_t > &c_count_map
 * @param std::map< feature_id, std::map< Feature, std::map< Label, size_t > > > &fv_c_count_map
 * @param D, size of the data vector
 * 
 * @return std::vector<double> &result, the gain vector result
 */
void C45DecisionTree::calculateInfoGainRatio(
		std::map< feature_id, std::map< Feature, size_t > > &fv_count_map, 
		std::map< Label, size_t > &c_count_map, 
		std::map< feature_id, std::map< Feature, std::map< Label, size_t > > > &fv_c_count_map, 
		std::vector<feature_id> &unused_feature, 
		size_t D, 
		std::vector<double> &result) {

	auto f_size = unused_feature.size();
	result.resize(f_size, 0);

	std::map< Label, size_t >::iterator c_it;
	std::map< Feature, std::map< Label, size_t > >::iterator f_l_it;
	std::map< Label, size_t >::iterator l_it;

	for(auto i=0; i<f_size; ++i) {
		double H_D = 0.0;     /* H(D) = sum_up_1toj(Ck/D * log(Ck/D)) */
		double H_D_A = 0.0;   /* H(D|A) = sum_up_1ton(Di/D * sum_up_1tok(Dik/Di * log(Dik/Di))) */
		double G_D_A = 0.0;   /* G(D|A) = H(D) - H(D|A) */
		double Gr_D_A = 0.0;  /* Gr(D|A) = G(D|A) / H(D) */

		feature_id fid = unused_feature.at(i);
		// make sure fv_count_map.find(fid) exists, or error will happen
		if(fv_count_map.end() == fv_count_map.find(fid)) {
			continue;
		}
		if(fv_c_count_map.end() == fv_c_count_map.find(fid)) {
			continue;
		}

		// call make_shared to get shared_ptr
		auto f_p = std::make_shared< std::map< Feature, size_t > >(fv_count_map[fid]);
		auto f_c_p = std::make_shared< std::map< Feature, std::map< Label, size_t > > >(fv_c_count_map[fid]);

		if (0 == f_p.use_count() || 0 == f_c_p.use_count()) {
			continue;
		}

		for(c_it=c_count_map.begin(); c_it!=c_count_map.end(); ++c_it) {
			double percent = static_cast<double>(c_it->second) / D;
			H_D += percent * log(percent);
		}

		for(f_l_it=f_c_p->begin(); f_l_it!=f_c_p->end(); ++f_l_it) {
			auto Di = fv_count_map[fid][f_l_it->first];
			double tmp = 0;
			for(l_it=f_l_it->second.begin(); l_it!=f_l_it->second.end(); ++l_it) {
				double percent = static_cast<double>(l_it->second) / Di;
				tmp += percent * log(percent);
			}
			H_D_A += (static_cast<double>(Di) / D) * tmp;
		}

		G_D_A = H_D - H_D_A;
		Gr_D_A = G_D_A / H_D;
		result[i] = Gr_D_A;
	}
}

void C45DecisionTree::printData() {
	auto d_size = input_vec.size();
	auto i = d_size;
	auto j = d_size;
	std::cout<<"The training data:"<<std::endl;
	for(i=0; i<d_size; ++i) {
		std::cout<<"input->"<<i<<"----";
		for(j=0; j<input_vec.at(i).size(); ++j) {
			std::cout<<"f"<<j<<":"<<input_vec.at(i).at(j)<<" ";
		}
		std::cout<<" label->"<<label_vec.at(i)<<std::endl;
	}

	std::cout<<"The feature and possible values:"<<std::endl;
	auto f_size = feature_table.size();
	auto fv = f_size;
	for(fv=0; fv<f_size; ++fv) {
		std::cout<<"feature:"<<fv<<std::endl;
		for(auto v_i=0; v_i<feature_table.at(fv).size(); ++v_i) {
			std::cout<<feature_table.at(fv).at(v_i)<<" ";
		}
		std::cout<<std::endl;
	}
}

/**  // test
	#include <iostream>
	#include <vector>
	#include "Utils.h"
	#include "C45DecisionTree.h"

	using namespace std;

	int main() {
	    // age: 1-> Young 2 -> middle-aged 3-> elderly
	    // having job: 1 -> yes 2 no
	    // having house: 1 yes 2 no
	    // credit conditions: 1 -> normal 2 -> good 3 very good
	    // class: 1 -> yes 2 no

	    vector<Feature> age_vec;
	    age_vec.push_back(1); age_vec.push_back(2); age_vec.push_back(3);
	    vector<Feature> job_vec;
	    job_vec.push_back(1); job_vec.push_back(2);
	    vector<Feature> house_vec;
	    house_vec.push_back(1); house_vec.push_back(2);
	    vector<Feature> credit_vec;
	    credit_vec.push_back(1); credit_vec.push_back(2); credit_vec.push_back(3);

	    vector< vector<Feature> > feature_table;
	    feature_table.push_back(age_vec);
	    feature_table.push_back(job_vec);
	    feature_table.push_back(house_vec);
	    feature_table.push_back(credit_vec);

	    const unsigned att_num = 4;
	    const unsigned rule_num = 15;
	    // the last colomn is the label
	    int train_data[rule_num][att_num + 1] = {    
	                        {1, 2, 2, 1, 2},
	                        {1, 2, 2, 2, 2},
	                        {1, 1, 2, 2, 1},
	                        {1, 1, 1, 1, 1},
	                        {1, 2, 2, 1, 2},

	                        {2, 1, 2, 1, 2},
	                        {2, 1, 2, 2, 2},
	                        {2, 2, 1, 2, 1},
	                        {2, 1, 1, 3, 1},
	                        {2, 1, 1, 3, 1},

	                        {3, 1, 1, 3, 1},
	                        {3, 1, 1, 2, 1},
	                        {3, 2, 2, 2, 1},
	                        {3, 2, 2, 3, 1},
	                        {3, 1, 2, 1, 2}
	                    };

	    // get init vector
	    vector< vector< Feature > > input_vec;
	    vector<Label> label_vec;
	    for(unsigned int i=0; i<rule_num; ++i) {
	        vector< Feature > input_tmp;
	        for(unsigned int j=0; j<att_num; ++j) {
	            Feature f_tmp;
	            f_tmp.set(train_data[i][j]);
	            input_tmp.push_back(f_tmp);
	        }
	        input_vec.push_back(input_tmp);
	        label_vec.push_back(Label(train_data[i][att_num]));
	    }

	    // test print
	    auto size = input_vec.size();
	    for(auto i=0; i<size; ++i) {
	        cout<<i+1<<endl;
	        for(auto j=0; j<input_vec.at(i).size(); ++j) {
	            cout<<"    "<<input_vec.at(i).at(j)<<endl;
	        }
	        cout<<"    "<<label_vec.at(i)<<endl;
	    }

	    // init data
	    C45DecisionTree c45tree;

	    for(auto i=0; i<feature_table.size(); ++i) {
	        c45tree.addFeature(feature_table.at(i));
	    }

	    for(auto i=0; i<input_vec.size(); ++i) {
	        c45tree.addData(input_vec.at(i), label_vec.at(i));
	    }

	    c45tree.printData();
	    c45tree.train();

	    vector<Feature> test;
	    test.push_back(1); test.push_back(2); test.push_back(2); test.push_back(1);
	    Label l_out = c45tree.predict(test);
	    cout<<"The class predicted is -> "<<l_out<<endl;
	    cout<<endl;
	}
*/
#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */