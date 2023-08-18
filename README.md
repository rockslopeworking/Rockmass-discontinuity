# Rockmass-discontinuity
## Introduction
We introduces an innovative semi-automatic method to identify discontinuities from point clouds. An artificial neural network designed to identify discontinuity sets was established. Inputs to the neural network consist of normal vectors, curvature computed by K-nearest Neighbor and Principal Component Analysis, as well as point cloud coordinates. The output layer generates the value i corresponding to the determined ith discontinuity set. Learning samples for network training were randomly selected from point clouds and automatically classified using FCM and PSO algorithms. The orientations of individual discontinuities, identified from the discontinuity set using Density-Based Spatial Clustering of Applications with Noise, were calculated.
## Citing our work
If you find our works useful in your research, please consider citing:
Cao B, Zhu X, Lin Z, Li Y, Bai D,Lu G,Zhu Z. Identification of Rock Mass Discontinuity from 3D Point Clouds Using Improved Fuzzy C-Means and Artificial Neural Network
