#!/bin/bash
set -e
# Input file containing list of folders
folder_list=$1

# Output file to store list of csv files
output_file="all_csv_files.txt"

# Remove output file if it already exists
rm -f $output_file

# Loop through each folder in the input file
while read folder; do
    # Strip any trailing newline character
    #folder=$(echo $folder | tr -d '\n')
    echo "$folder"
    # Check if folder exists
    if [ -d "$folder" ]; then
        # Loop through each csv file in folder and write folder/csv_file to output file
        for file in $folder/*.csv; do
            echo "$folder/${file##*/}" >> $output_file
        done
    else
        echo "Folder $folder not found. Skipping."
    fi
done < $folder_list

echo "List of csv files written to $output_file"
