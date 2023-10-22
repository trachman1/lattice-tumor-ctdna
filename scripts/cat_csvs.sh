#!/bin/bash

# Get file name from command line argument
file=$1

# Check if the file exists
if [ ! -f $file ]; then
  echo "File $file does not exist."
  exit 1
fi

# Get the first line of the first file to use as the header
header=$(head -n 1 $(head -n 1 $file))

# Create a new file to hold the concatenated data
output_file=$2
echo $header > $output_file

# Concatenate all the csv files while skipping the header line
while read line; do
  if [[ $line == *.csv ]]; then
    tail -n +2 $line >> $output_file
  else
    echo "Skipping file $line since it is not a csv"
  fi
done < $file

echo "Concatenation complete. Result saved to $output_file."
