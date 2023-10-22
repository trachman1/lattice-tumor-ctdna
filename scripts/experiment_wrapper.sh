#!/bin/bash
#this only works with the old experiments 
set -e
export PATH
mkdir py3-numpy
tar -xzf py3-numpy.tar.gz -C py3-numpy
. py3-numpy/bin/activate

python3  experiment_runner.py $1 $2 $3 $4 $5 
tar -czvf inner_$1_outer_$2_br_$3_s_$4_rep_$5.tar.gz inner_$1_outer_$2_br_$3_s_$4_rep_$5

  
