#!/bin/sh
# Created by Chengyu on 2021/12/5.

# compile.
cd ./src
make cleanall
make
cd ..

# Configuration.
data_source="../../data/"
OUTDIR="output/synthetic_data/"

d=4
num_group=5
num_ts_in_group=20

for (( i=1; i<=$num_group; i++ ))
do
  INPUTDIR=$data_source"synthetic_data_AutoPlait/dataset"$i"/"
  dblist=$INPUTDIR"list"
  outdir=$OUTDIR"dataset"$i"/"
  for (( j=1; j<=$num_ts_in_group; j++ ))
  do
    output=$outdir"dat"$j"/"
    mkdir -p $output
    input=$output"input"
    awk '{if(NR=='$j') print $0}'# $dblist > $input
    echo $input
    ./src/autoplait $d $input $output
  done
done