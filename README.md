# Rockmass-discontinuity
## Introduction
We introduces an innovative semi-automatic method to identify discontinuities from point clouds. An artificial neural network designed to identify discontinuity sets was established. Inputs to the neural network consist of normal vectors, curvature computed by K-nearest Neighbor and Principal Component Analysis, as well as point cloud coordinates. The output layer generates the value i corresponding to the determined ith discontinuity set. Learning samples for network training were randomly selected from point clouds and automatically classified using FCM and PSO algorithms. The orientations of individual discontinuities, identified from the discontinuity set using Density-Based Spatial Clustering of Applications with Noise, were calculated.
## Citing our work
If you find our works useful in your research, please consider citing:

    Cao B, Zhu X, Lin Z, Li Y, Bai D,Lu G,Zhu Z. Identification of Rock Mass Discontinuity from 3D Point Clouds Using Improved Fuzzy C-Means and Artificial Neural Network. 
    
## Install & complie
Please directly copy all the code into your workspace and complie it with any complier that supports MATLAB 2021b and later versions. It dose not require linking any additional libraries.

## Sample usage:
The mainFunction.m include nine sections. Enter the required parameters according to the prompts of each part. Run it from top to bottom one by one, that is, you can run the next section only after section of the operation is complete.

      Section 1: importData.      
          Click on "run section" in the 'editor', and a file selection dialog will appear. Choose the raw point cloud data file you want to input.
          
      Section 2: Parameter Calculation.
          Click on "run section" in the 'editor', and input the vaule K, then wait for the calculation to complete.

      Section 3: Flip the normal vector if it is not pointing towards the sensor.
          Click on "run section" in the 'editor', and wait for the calculation to complete.

      Section 4: Discard the edges.
          Click on "run section" in the 'editor', and input the vaule cumulative probability, then wait for the calculation to complete.
          
      Section 5: Categorize randomly selected points based on FCM with PSO.
          Click on "run section" in the 'editor', and input two parameters:the number of discontinuity set and ratio of downsmple, then wait for the calculation to complete.

      Section 6: Categorize the entire point cloud using neural network trained by learning samples.
          Click on "run section" in the 'editor', and wait for the calculation to complete.
          
      Section 7: Show the classification results with one color per discontinuity set.
          Click on "run section" in the 'editor', and wait for the calculation to complete.
        
      Section 8: DBSCAN is used to segment discontinuity set to obtain individual discontinuities.
          Click on "run section" in the 'editor', and input two parameters:eps and minCluster according to the content displayed in the 'Command Window'. Then, wait for the calculation to complete.
          
      Section 9: Calculate the orientation
          Click on "run section" in the 'editor', and wait for the calculation to complete.
## Contact
Please feel free to leave suggestions or comments to Dr. Cao (beicao@csu.edu.cn), or Prof. Lu (luguangyin@csu.edu.cn)
