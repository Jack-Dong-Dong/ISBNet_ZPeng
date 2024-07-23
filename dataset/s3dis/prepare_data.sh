#!/bin/bash
echo Prepare raw data
python prepare_s3dis.py
echo Prepare superpoints
python prepare_superpoints.py