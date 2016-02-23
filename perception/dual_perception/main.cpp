#include "dual_perception.h"

int main(){
    std::vector<std::vector<double> >test_x(3);
    test_x[0].push_back(3);test_x[0].push_back(3);
    test_x[1].push_back(4);test_x[1].push_back(3);
    test_x[2].push_back(1);test_x[2].push_back(1);
    std::vector<int> test_y(3);
    test_y[0] = 1;
    test_y[1] = 1;
    test_y[2] = -1;
   
    DualPerception *model = new DualPerception(1, 1.0, 100);
    model->train(test_x,test_y);
    model->printPerceptronModel();
}
