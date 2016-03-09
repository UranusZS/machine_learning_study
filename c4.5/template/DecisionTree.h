#ifndef DECISIONTREE_H
#define DECISIONTREE_H

/**
 * @author ZS (dragon_201209@126.com)
 * @date Mar 7, 2016 4:25:45 PM
 * c4.5
 */

#include <vector>
#include <map>

#include "Utils.h"

class TreeNode {
public:
	TreeNode* root;
	std::vector< TreeNode* > child;
	
	Label label;  // feature

	    unsigned root;//节点属性值
    vector<unsigned> branches;//节点可能取值
};

class DecisionTree {
public:
	DecisionTree();
	~DecisionTree();
	bool train();
	bool prune();
	Label predict(std::vector<Feature> &input);
private:
	TreeNode* root;
	std::vector<size_t> feature_used;
	std::vector< std::vector<feature> > feature_table;

	std::vector< std::vector< feature > > train_input_vec;
	std::vector<Label> train_label_vec;

	double epsilon;
	bool makeTree(TreeNode* node);
	std::vector<size_t> getUnusedFeature();                  /* get the feature subscript of the features that unused */
	Label getMaxCountLabel(std::vector< Label > &label);     /* get the label of the majority */
	/* calculate the Info Gain Ratio of each feature split*/
	void calculateInfoGainRatio(std::map< feature, size_t > &f_count_map, std::map< Label, size_t > &c_count_map, std::map< feature, std::map< Label, size_t > > &f_c_count_map, std::vector<size_t> &unused_feature, size_t D, std::vector<double> &result);
};


DecisionTree::DecisionTree() {

}

DecisionTree::~DecisionTree() {
	
}

bool DecisionTree::train() {
	return true;
}

bool DecisionTree::makeTree(TreeNode* node, std::vector< std::vector< feature > > &input, std::vector< Label > &label) {
	if (input.size() != label.size()) {
		return false;
	}

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
	if (i == size) {
		node->label = l0;
		return true;
	}

	// unused feature is empty
	std::vector<size_t> unused_feature = getUnusedFeature();
	if(empty(unused_feature)) {
		node->label = getMaxCountLabel(label);
		return true;
	}

	// for all feature unused, calculate the gain
	auto f_size = unused_feature.size();
	std::vector<double> feature_gain(f_size, 0.0);

	std::map< feature, size_t > f_count_map;
	std::map< feature, std::map< Label, size_t > > f_c_count_map;
	std::map< Label, size_t > c_count_map;
	countFeatureAndLabel(input, label, feature_used, f_count_map, c_count_map, f_c_count_map);
	calculateInfoGainRatio(f_count_map, c_count_map, f_c_count_map, unused_feature, i_size, feature_gain);

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
		node->label = getMaxCountLabel(label);
		return true;
	}

	// max_gain >= epsilon, build the tree
	split by max_gain_feature
	build each subtree
	return true;
}

/**
 * getUnusedFeature: return the feature subscript of the features that unused
 *
 */
std::vector<size_t> DecisionTree::getUnusedFeature() {
	std::vector<size_t> feature_unused;
	auto size = feature_used.size();
	for(auto i=0; i<size; ++i) {
		if(0 == feature_used) {
			feature_unused.push_back(i);
		}
	}
	return feature_unused;
}

Label DecisionTree::getMaxCountLabel(std::vector< Label > &label) {
	Label max_label = getMaxCountT(label);
	return max_label;
}

void DecisionTree::calculateInfoGainRatio(std::map< feature, size_t > &f_count_map, std::map< Label, size_t > &c_count_map, std::map< feature, std::map< Label, size_t > > &f_c_count_map, std::vector<size_t> &unused_feature, size_t D, std::vector<double> &result) {
	auto f_size = unused_feature.size();
	result.resize(f_size, 0);

	std::map< Label, size_t >::iterator c_it;
	std::map< feature, std::map< Label, size_t > >::iterator f_l_it;
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