#!/bin/usr/env bash

declare -i cur_data_num=1
for file in ./math_symbol_data/add/*
do
    mv "$file" "./math_symbol_data/add/add_$cur_data_num.jpg"
    cur_data_num+=1
done
