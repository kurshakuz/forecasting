# Augmenting Human-Robot Collaboration Task By Human Hand Position Forecasting

This repository contains the code for the "Augmenting Human-Robot Collaboration Task By Human Hand Position Forecasting" master's thesis submitted in partial fulfillment of the requirements for the degree of Master of Science at the Department of Advanced Computing Sciences of Maastricht University.

The repository consists of multiple folders, each containing source code for each module.

## Ego4D
The `ego4d` folder contains Ego4D Future Hand Prediction dataset preparation helper tools and I3D model training and inference pipeline. Setting up instructions are contained inside the `THESIS.md` file.

## 100DoH Serilization
The `100doh_serialization` contains helper tools for 100DoH dataset preparation. It consists of a detection pipeline combined with the method for automatic hand trajectory annotation proposed by [Liu et al.](https://arxiv.org/abs/2204.01696). Setting up instructions are contained inside the `THESIS.md` file.

## DynamicVNS
The `dynamic_vns` folder contains the implementation of the DynamicVNS algorithm. Setting up instructions are contained inside the `main.py` file with corresponding command line arguments.

## UniFormer
The `uniformer` folder contains the implementation, training, and inference pipelines of the UniFormer model. Setting up instructions and script configurations are contained inside the `THESIS.md` file.

## UniFormerV2
The `uniformer_v2` folder contains the implementation, training, and inference pipelines of the UniFormerV2 model that are based on EgoVLP pretrained transformer backbones. This setup and corresponding experiments are not mentioned in the thesis, due to low performance.