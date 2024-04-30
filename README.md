
# Removing light artifacts 

This code is useful for removing light artifacts from the underwater images caused by artificila light.


1 Removing light artifacts from the illumination channel in underwater images

We can utilize the illumination channel model to eliminate light artifacts and haze from underwater images

2  Fine Detail Recovery from the Reflection Channel of Underwater Images
The reflection channel correction module enables the recovery of details from the reflection channel

3  AB channel correction

The AB channel correction network is employed to balance the color in underwater images.







Here is the list of libraries you need to install to execute the code:

python = 3.6
cv2
numpy
scipy
matplotlib
scikit-image
natsort
math
datetime
```
    
1 Complete the running environment configuration;
2 Replace  the inputs images path to corresponding place given in  the code
3  Run Python illumination.py, reflection.py and ab channel correction.py;
4 Find the enhanced/restored images 
Datasets can be downloaded using the provided links:
(i) https://github.com/lilala0/UIALN
(ii) https://li-chongyi.github.io/proj_benchmark.html   
(iii)  https://www.kaggle.com/datasets/pamuduranasinghe/euvp-dataset   
(iv) https://lintaopeng.github.io/_pages/UIE%20Project%20Page.html