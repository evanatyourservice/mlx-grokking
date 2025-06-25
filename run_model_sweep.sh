#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main_model_sweep.py --early-stop --dim-sweep 32,64,128,256 "$@" 
CUDA_VISIBLE_DEVICES=0 python main_model_sweep.py --early-stop --depth-sweep 1,2,3,4 "$@" 
CUDA_VISIBLE_DEVICES=0 python main_model_sweep.py --early-stop --heads-sweep 1,2,4,8 "$@" 
CUDA_VISIBLE_DEVICES=0 python main_model_sweep.py --early-stop --dropout-sweep 0.0,0.1,0.2,0.3 "$@" 
