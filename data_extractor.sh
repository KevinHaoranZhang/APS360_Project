#!/bin/usr/bash
rm -rf math_symbol_data
unzip -qq './math_symbol_data.zip'

mkdir math_symbol_data/training_dataset/
mkdir math_symbol_data/training_dataset_rapper/
mkdir math_symbol_data/training_dataset_rapper/all/
mkdir math_symbol_data/validation_dataset/
mkdir math_symbol_data/validation_dataset_rapper/
mkdir math_symbol_data/validation_dataset_rapper/all/
mkdir math_symbol_data/testing_dataset/
mkdir math_symbol_data/testing_dataset_rapper/
mkdir math_symbol_data/testing_dataset_rapper/all/

declare -a symbols=("div" "eq" "lb" "rb" "add" "sub")
for i in "${symbols[@]}"
do
  # Get the total number of data for each symbol
  declare -i total_data_num=0
  for file in math_symbol_data/$i/*
  do
    total_data_num+=1
  done
  # Create a sub-dataset for each symbol
  mkdir math_symbol_data/training_dataset/$i
  mkdir math_symbol_data/validation_dataset/$i
  mkdir math_symbol_data/testing_dataset/$i
  declare -i cur_data_num=0
  # Negligible rounding error by bash
  # Limit data size to 5500
  declare -i data_limit=5500
  if [ $total_data_num -ge $data_limit ]; then
    total_data_num=5500
  fi
  # Training data (70%)
  declare -i training_data_num=total_data_num/10*7
  # Validation data (15%)
  declare -i validation_data_num=total_data_num/20*3
  # Testing data (15%)
  declare -i testing_data_num=total_data_num-training_data_num-validation_data_num
  echo "Symbol $i data: Total ($total_data_num) Traning ($training_data_num) Validation ($validation_data_num) Testing ($testing_data_num)"
  declare -i validation_data_bound=$training_data_num+$validation_data_num
  declare -i testing_data_bound=$training_data_num+$validation_data_num+$testing_data_num
  # Copy image into corresponding dataset
  for file in math_symbol_data/$i/*
  do
    cur_data_num+=1
    if [ $cur_data_num -le $training_data_num ]; then
      cp "$file" "math_symbol_data/training_dataset/$i/"
      cp "$file" "math_symbol_data/training_dataset_rapper/all/"
    elif [ $cur_data_num -le $validation_data_bound ]; then
      cp "$file" "math_symbol_data/validation_dataset/$i/"
      cp "$file" "math_symbol_data/validation_dataset_rapper/all/"
    elif [ $cur_data_num -le $testing_data_bound ]; then
      cp "$file" "math_symbol_data/testing_dataset/$i/"
      cp "$file" "math_symbol_data/testing_dataset_rapper/all/"
    fi
  done
done

