# Rockmass-discontinuity
## Introduction
Accurately obtaining rock mass discontinuity information holds particular significance for slope stability analysis and rock mass classification. Currently, non-contact measurement methods have increasingly become a supplementary means to traditional techniques, especially in hazardous and inaccessible areas. This study introduces an innovative semi-automatic method to identify discontinuities from point clouds. A modified convolutional neural network, AlexNet, was established to identify discontinuity sets. This architecture consists of five convolutional layers and three fully connected layers, utilizing 1×3 normal vectors computed by K-nearest Neighbor and Principal Component Analysis as input and generating an output value "i" that represents the identified discontinuity set associated with the "i" category. Learning samples for network training were randomly selected from point clouds and automatically categorized using improved Fuzzy C-Means based on Particle Swarm Optimization. The orientations of individual discontinuities, identified from the discontinuity set using Hierarchical Density-Based Spatial Clustering of Applications with Noise, were calculated. Two publicly available outcrop cases were employed to validate the efficacy of the proposed method, and parameter analysis was conducted to determine optimal parameters. The results demonstrated the reliability of the method and highlighted improvements in automation and computational efficiency.
## Citing our work
If you find our works useful in your research, please consider citing:

    Lu G, Cao B, Zhu X, Lin Z, Bai D, Tao C, Li Y. Identification of Rock Mass Discontinuity from 3D Point Clouds Using Improved Fuzzy C-Means and Convolutional Neural Network. 
    
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
        
      Section 8: HDBSCAN is used to segment discontinuity set to obtain individual discontinuities.
          First run 'section 8.1' in matlab
          Secend run 'section 8.2' in python
          The hdbscan program for segment discontinuity set runs in python. See hdbscan.py for the code When executing hdbscan in python, import the result with the section 8.3 command
          Finally run 'section 8.3' in matlab
      Section 9: Calculate the orientation
          Click on "run section" in the 'editor', and wait for the calculation to complete.
## Contact
Please feel free to leave suggestions or comments to Dr. Cao (beicao@csu.edu.cn), or Prof. Lu (luguangyin@csu.edu.cn)
