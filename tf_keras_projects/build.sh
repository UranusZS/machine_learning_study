#!/bin/sh
project=ml_common
rm -rf ${project}_bak
cp -rf ${project} ${project}_bak

dirs=("${project}/" "${project}/common" "${project}/config"  "${project}/loss" "${project}/model" "${project}/optimizer" "${project}/script" "${project}/test" "${project}/utils")
for d in ${dirs[@]}; do
    # echo $d
    rm -rf "${d}/__pycache__"
done

mkdir -p target
rm -rf ${project}/build
rm -rf ${project}/dist
rm -rf ${project}/ml_common.egg-info
# cd ${project} && python setup.py bdist_egg && cd ..
