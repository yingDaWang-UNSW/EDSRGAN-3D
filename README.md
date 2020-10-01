# Super Resolution by Neural Networks
## Introduction
Implementation for the super resolution of both 2D and 3D images. Involved Neural Networks: EDSR, SR-Resnet Residual, SRGAN, WDSR, SRCNN.

## Authors
[YingDa Wang](https://github.com/yingDaWang-UNSW "GitHub Account")<br>
[Name](social media account link "hover information")<br>
[Name](social media account link "hover information")<br>
[Name](social media account link "hover information")<br>
...

(Department, UNSW)

## Installation
**Version 0.1**<br>
This software is compatible with windows, mac, and linux  machines. It will install anaconda3, along with the necessary python packages in a containerised anaconda environment. <br>

**Linux and Mac**

Open a terminal window at the directory where the file “installSR.sh” is located and type “bash installSR.sh”. 



Step1: 

Open a “Anaconda cmd Prompt” terminal window at the directory where the file “installSR.sh” is located and  type “bash installSR.sh”. 
```
$ bash installSR.sh
```

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step1.png)


Step2: 

Please input this command: 

For Mac, please use

```
$ bash installSRMac.sh
```

For Linux, please use

```
$ bash installSRLinux.sh
```

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step2.png)

Step3: 

Always type “y” for yes if it is required. 

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step3.png)

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step4.png)

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step5.png)

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step6.png)

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step7.png)

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step8.png)

Step4: 

Please input this command to run the software: 

For Mac, please use
```
$ bash runSRMac.sh
```

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step9.png)

For Linux, please use
```
$ bash runSRLinux.sh
```

<br>FOR ADVANCED USERS: if you already have anaconda3, or wish to install the packages yourself, a full list of conda packages used by this software are shown below:<br>

```
conda install tensorflow=1.13.0 

conda install matplotlib 

conda install pillow 

conda install -c conda-forge gooey  

pip install tensorlayer==1.11 

pip install argparse 
```

**Windows**

For Windows system, we reccomand you to install via the conda promt.

<br> If you do not have the anaconda3, you may download it via this link:<br>
https://repo.anaconda.com/archive/Anaconda3-2019.03-Windows-x86_64.exe

For Linux system, click this link:
https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh

Then, please open a “Anaconda Prompt” terminal window shown in the below image.  

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step11.png)

The specific installation steps are as follow. Always type "y" for "yes".
```
conda update -n base -c defaults conda

conda create --name srRockEnv2 python=3.6

conda activate srRockEnv2
```

For linux system, you may prefer to type "conda install tensorflow-gpu=1.13.0":
```
conda install tensorflow-gpu=1.13.0
```

For Mac system, "conda install tensorflow=1.13.0" instead:
```
conda install tensorflow=1.13.0
```

Then, please type these lines:
```
conda install matplotlib

conda install pillow

conda install -c conda-forge gooey

pip install tensorlayer==1.11

pip install argparse
```

##Running Guideline##

A desktop shortcut icon should have been created as part of the installation. Please double click it. In case we can't get that to work:<br>
Open a terminal window at the directory where the file “runProgram.sh” is located and type “bash runProgram.sh”.

**Input Format**
There are 5 different formats of input image: .png, .jpg, .mat, .nc, .tif.

**2D Images**
All input 5 input formats listed above are acceptable for 2D images resolution.

**3D Images**
The format of .mat, .nc and .tif are acceptable for 3D image super resolution.

|parameter|usage|
|----------| :-------: |
|Input images|Folder which containing input images, named “srtestfolder”|
|Input format|The format of input image|
|Scale factor|Up sampling factor|
|Output format|The format of output image|
|Bit depth|Bit depth of your input images|
|Image dimension|2D or 3D dimension of input images|
|Use CPU|Force the network to use the CPU as default, if no compatible GPU is detected|
|Use GAN|Force the network to use the GAN as default|

**Output format and Visualisation**

The output images are stored in “srtestfolder\srOutputs”.

For 2D images, output is slice-by-slice, and readable my standard image reading software.

For 3D images, we recommend ImageJ: https://imagej.net/Fiji/Downloads

**Examples**

***2D Images***

1. choose the folder containing the input images on your computer;

2. choose the up-sampling scale factor you want, such as 4, 16 or 64;

3. when you choose to resolution 2D images, it may not be necessary to use 3D patches;

4. there are only two types of 2D output formats, “.png” and “.jpg”, could be used;

5. bit depth: uint8 means unsigned 8-bits integer;

6. choose 2D as image dimension;

7. if you choose “yes” as “USE CPU”, the software will force the network to use CPU rather than GPU. And CPU will as default if you do not choose;

8. there are two types of neural network could be provided, such as CNN and GAN. And the CNN will as default if you do not choose;

9. determine to use checkpoints for training 2D images or not.

10. Start to resolution your images!

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step2d.png)

***3D Images***

The steps for 3D images are almost similar with those for 2D images, except:

-	use 3D patches as recommended for 3D images;

-	there are three types of 3D output formats, “.mat”, “.nc” and “.tiff”, could be used;

-	choose 3D as image dimension;

![image](https://github.com/LiLeaf/SRInstall_images/blob/master/step3d.png)
