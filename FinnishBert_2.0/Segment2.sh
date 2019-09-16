#!/bin/bash
for file in Segment2/*
do 
python Evaluate_nbest1000_1507.py --input_segments=$file
done