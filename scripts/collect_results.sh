#!/bin/bash
#script to collect results from single run of simulation

foldername=$1
echo "$foldername" > "foldername.txt"
bash scripts/mk_csv_list.sh "foldername.txt"
bash scripts/cat_csvs.sh "all_csv_files.txt" "demo-output/simulation-results.csv" #concatenate all the timepoints
python scripts/process_timeseries.py "demo-output/simulation-results.csv" "demo-output" #postproccessing 
#delete temporary files
rm "foldername.txt"
rm "all_csv_files.txt"
rm -r demo-output/timepoints
 