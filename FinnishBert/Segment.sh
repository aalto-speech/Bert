#!/bin/bash
for file in Segments/*
do 
python Evaluate_nbest1000_0407.py --input_segments=$file
done