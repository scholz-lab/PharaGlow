# PharaGlow

TODO: add nice subtitle (PharaGlow: ...)

TODO: insert nice images here AND/OR below each step below

PharaGlow is a python package for tracking and analyzing C. elegans pharynx from movies.
 Tracking is based on the package trackPy (http://soft-matter.github.io/trackpy/v0.4.2/).
 The package can be used to simply track labelled pharynxes
 (or animals from brightfield) as a simple center of mass tracker,
 but it also has a pipeline to extract pharyngeal pumping and features of the pharynx.

Raw data are tiff files
 typically obtained from simultaneously recording of up to 50 adults worms at 30 frames per second
 at 1x magnification.
 Typical use is by interacting through the notebook which contains the whole pipeline from raw movies to final data.
 It comprises three stages on analysis which can be done sequentially and are independent.
 Analyses can be interrupted at the end of each stage after saving the output dataframe.


**1. Step -  Basic object detection**
    This step creates a "_features.json" file
	which contains a table of objects (worms) detected in each frame.
	It creates also a stack of images that contain a cropped area around each worm.
    
**2. Step - Linking objects into trajectories**
    This step results in individual files "_trajectory.json" for each tracked animal.
    
**3. Step - Analyzing the details of object shapes**
    This step is doing the heavy lifting:
	It extracts centerlines, widths, contours and other object descriptors from the objects.
	It results in individual files "_result.json" for each tracked animal.
	
All subsequent analyses steps add 'columns' to the dataframe,
 and thus features is a subset of trajectories is a subset of results.



## Installation

### (1) Install Anaconda
You need to have Anaconda (https://www.anaconda.com/products/individual), 
Python >3.7 and Git installed. 
We recommend using Anaconda to install Python and Git.

### (2) Clone PharaGlow repository from Github in your local directory
 
 1/ Copy the repository link from Github in https://scholz-lab.github.io/PharaGlow/ 
 (in Branch Master > Code > HTTPS OR SSH)
 
 2/ In the terminal (Linux)/Anaconda Command Prompt (Windows),
 navigate to the directory where to clone PharaGlow
 and write the git clone command:

```
git clone https://github.com/scholz-lab/PharaGlow.git
```

Note that you can also download PharaGlow from our Github repository (Branch Master > Code > Download ZIP)
to have the current copy of PharaGlow

### (3) Create and activate the required anaconda environment
In the terminal (Linux)/Anaconda Command Prompt (Windows),
 naviguate to your newly cloned PharaGlow directory
 and run:

```bash
conda env create --file environmentPumping.yml
```

You can now use this environment by running:

```
conda activate pumping
```


### (4) Install PharaGlow

Last step, (don't forget to activate the pumping environment) run: 
```
python setup.py install --user
```

### (*optional*)
 *Create a dedicated environment kernel*

```
conda activate myenv
python -m ipykernel install --user --name myenv --display-name "Python (myenv)"
```

*And remove notebook output before committing*

```
conda install -c conda-forge nbstripout
```

## Quick Start
### Run PharaGlow on a demo dataset
We provide a demo data set with 1000 frames of 1x magnification
 (30fps, 2.34um per pixel),
 showing *C. elegans* expressing myo-2::mCherry.
 (TODO : add link here)

Before analyzing your data, we recommend to check your installation
 and familiarize yourself with the code by running the jupyter notebook
 "PharaGlowMain.ipynb" (in "notebooks") on this dataset.
 using the parameter file "AnalysisParameters_1x.json"
 
You can also find the expected outputs in TODO : add link here

TODO add PharaGlowMain.ipynb run on the demodataset with the output

### Run PharaGlow on your data
#### Raw files requirement
#### Parameters file
PharaGlow requires a json parameter file with the parameters that are editable by you. 
A default file comes with the repository, you can use it as a starting point (AnalysisParameters_1x.json)
These parameters are:


| **Parameters**  | **Value**  |                                                                                                                      |
| subtract        | 0          | subtract the background from the movie for detection. Helps particularly with the higher resolution movies (0 or 1)  |
| smooth          | 3          | should the image be smoothed. This helps to avoid breaking up the pharynx into two parts (integer >=0 in px)         |


#### Run PharaGlow on a single data set
system requirement

#### Run PharaGlow in parallel processing
system requirement


## Code contributors
## Community Support, Developers, & Help
## References
## License



