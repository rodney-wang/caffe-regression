#!/usr/bin/env sh

caffe train -solver kaggle_prototxt/fkp_solver_1.3.prototxt -gpu 2,3  -weights /ssd/zq/parkinglot_pipeline/carplate/caffe-regression/kaggle_prototxt/model_v1.3/20181201_fg_1e-6_iter_100000.caffemodel 
