//============================================================================
// Name        : HMM.cpp
// Author      : zs
// Version     : 1.0
// Description : in C++, Ansi-style
//============================================================================

#include <iostream>
#include "HMM.h"
using namespace std;

int main() {
	HMM<int, int> hmm;
	hmm.loadHMM("hmm.model");
	hmm.printHMM();
	return 0;
}
