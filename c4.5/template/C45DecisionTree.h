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
	class_id category;
	feature_id fid;
	// children
	std::vector<C45TreeNode*> children;
	// functions
	void set(bool leaf, class_id cat, feature_id f=UNUSED_FEATURE_ID) : is_leaf(leaf), category(cat), fid(f) {}
} C45TreeNode;

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
private:
	C45TreeNode* root;        /* the Decision Tree root */
	double epsilon;           /* threshold of the gain */

	bool makeTree(C45TreeNode* node, std::vector< std::vector< Feature > > &input, std::vector< Label > &label, std::vector<feature_id> &feature_used);

	std::vector< std::vector<Feature> > feature_table;   /* to describe feature values */
	std::vector< std::vector< Feature > > input_vec;     /* input of the labeled data */
	std::vector<Label> label_vec;                        /* label of the labeled data */
	std::vector<feature_id> feature_vec;                 /* feature ids of the data */

	/* calculate the Info Gain Ratio of each feature split*/
	void calculateInfoGainRatio(std::map< Feature, size_t > &f_count_map, std::map< Label, size_t > &c_count_map, std::map< Feature, std::map< Label, size_t > > &f_c_count_map, std::vector<feature_id> &unused_feature, size_t D, std::vector<double> &result);
	/* split the input data by the given feature */
	bool splitByFeature(std::vector< std::vector< Feature > > &input, std::vector< Label > &label, feature_id f_index, std::map< Feature, std::vector<Feature> > &input_split_map, std::map< Feature, std::vector<Label> > &label_split_map);
};

Label C45DecisionTree::predict(std::vector<Feature> &input) {
	feature_id fid = root->fid;
	Feature f = input.at(fid);
	// next node iterator
	return Label(0);
}

/**
 * train the tree according to the given training data 
 */
bool C45DecisionTree::train() {
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
	feature_vec.clear();
}

/**
 * makeTree
 * @param C45TreeNode* node                              the current node to process
 * @param std::vector< std::vector< Feature > > &input   the input data
 * @param std::vector< Label > &label                    the label of input data
 * @param std::vector<feature_id> &feature_used          the feature can be used to classify
 * @return boolean                                       true for success build the tree
 */
bool makeTree(C45TreeNode* node,                                       \
				std::vector< std::vector< Feature > > &input,   \
				std::vector< Label > &label,                    \
				std::vector<feature_id> &feature_used) {        
	if (input.size() != label.size()) {
		return false;
	}
	node->children.clear();
	if (input.empty()) {
		return true;
	}

	// all data belong to one class
	auto i_size = input.size();
	auto i=i_size;   // for exact type
	Label l0;
	for(i=0; i<i_size; ++i) {
		if(label.at(i) != l0) {
			break;
		}
	}
	if (i == i_size) {
		node->set(true, l0, UNUSED_FEATURE_ID);
		return true;
	}

	// unused feature is empty
	if(empty(feature_used)) {
		l0 = getMaxCountLabel(label);
		node->set(true, l0, UNUSED_FEATURE_ID)
		return true;
	}

	// for all feature unused, calculate the gain
	auto f_size = feature_used.size();
	std::vector<double> feature_gain(f_size, 0.0);

	std::map< Feature, size_t > f_count_map;
	std::map< Feature, std::map< Label, size_t > > f_c_count_map;
	std::map< Label, size_t > c_count_map;
	countFeatureAndLabel(input, label, feature_used, f_count_map, c_count_map, f_c_count_map);
	calculateInfoGainRatio(f_count_map, c_count_map, f_c_count_map, feature_used, i_size, feature_gain);

	size_t max_gain_feature = 0;
	double max_gain = 0.0;
	for(auto i=0; i<f_size; i++) {
		if(feature_gain.at(i) > max_gain) {
			max_gain = feature_gain.at(i);
			max_gain_feature = unused_feature.at(i);
		}
	}

	// if max_gain < epsilon, then single node tree get
	if(max_gain < epsilon) {
		l0 = getMaxCountLabel(label);
		node->set(true, l0, UNUSED_FEATURE_ID);
		return true;
	}

	// max_gain >= epsilon, build the tree
	l0 = getMaxCountLabel(label);
	node->set(true, l0, max_gain_feature);

	std::map< Feature, std::vector<Feature> > input_map;
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
	std::map< Feature, std::vector<Feature> >::iterator f_it;
	Feature f;
	for(f_it=input_map.begin(); f_it!=input_map.end(); ++f_it) {
		f = f_it->first;
		C45TreeNode* child = new C45TreeNode();
		bool success = makeTree(child, input_map[f], label_map[f], feature_remained);
		if(success) {
			node->children.push_back(child);
		}
	}
	return true;
}

bool C45DecisionTree::splitByFeature(std::vector< std::vector< Feature > > &input,  \
			std::vector< Label > &label,                                            \
			feature_id f_index,                                                     \    
			std::map< Feature, std::vector<Feature> > &input_split_map,             \
			std::map< Feature, std::vector<Label> > &label_split_map
			) {
	if(input.size() != label.size()) {
		return false;
	}

	input_split_map.clear();
	label_split_map.clear();
	std::map< Feature, std::vector< std::vector< Feature > > >::iterator f_it;
	std::map< Feature, std::vector<Label> >::iterator l_it;

	auto size = input.size();
	for(auto i=0; i<size(); ++i) {
		Feature f = input.at(i).at(f_index);
		// input process & label process, for the reason of the feature
		f_it = input_split_map.find(f);
		if(f_it == input_split_map.end()) {                      // first met
			std::vector< std::vector< Feature > > f_tmp;
			f_tmp.push_back(input.at(i));
			input_split_map.insert(std::pair<Feature, std::vector< std::vector< Feature > > >(f, f_tmp));

			std::vector<Label> l_tmp;
			l_tmp.push_back(label.at(i));
			label_split_map.insert(std::pair<Feature, std::vector<Lable> >(f, l_tmp));
		} else {                                                // already added
			input_split_map[f].push_back(input.at(i));
			label_split_map[f].push_back(label.at(i));
		}
	}
	return true;
}

void C45DecisionTree::calculateInfoGainRatio(std::map< Feature, size_t > &f_count_map, std::map< Label, size_t > &c_count_map, std::map< Feature, std::map< Label, size_t > > &f_c_count_map, std::vector<size_t> &unused_feature, size_t D, std::vector<double> &result) {
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

		for(c_it=f_count_map.begin(); c_it!=f_count_map.end(); ++c_it) {
			double percent = static_cast<double>(c_it->second) / D;
			H_D += percent * log(percent);
		}

		for(f_l_it=f_c_count_map.begin(); f_l_it!=f_c_count_map.end(); ++f_l_it) {
			auto Di = f_count_map[f_l_it->first];
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


#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */