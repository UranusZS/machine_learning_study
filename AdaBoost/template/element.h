#ifndef ELEMENT_H
#define ELEMENT_H


/**
 * to read formatted data
 * @author ZS (dragon_201209@126.com)
 * @date April 6, 2016 11:25:45 AM
 * AdaBoost
 */

#include <vector>


struct FeatureElement {
    int    index;
    double value;
};

struct Record {
	std::vector<FeatureElement> input;
	int 						label;
};



#endif
/* vim: set expandtab ts=4 sw=4 sts=4 tw=100: */