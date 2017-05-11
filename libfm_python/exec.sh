#!/bin/bash
# just for test the lib reader
./bin/libFM -dim "1,1,10" -iter 5 -learn_rate 0.1 -task c -method sgd -train heart_scale -test heart_scale  -save_model fm.model
echo -ne "1\t1\t14\t10\n" > head
cat head fm.model > fm.model_new
mv fm.model_new fm.model
