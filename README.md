# PharaGlow

TODO: add nice subtitle (PharaGlow: ...)

TODO: insert nice images here AND/OR below each step below

PharaGlow is a python package for tracking and analyzing C. elegans pharynx from movies. Tracking is based on the package trackPy (http://soft-matter.github.io/trackpy/v0.4.2/). The package can be used to simply track labelled pharynxes or as a simple center of mass tracker for brightfield, but it also has a pipeline to extract pharyngeal pumping and features of the pharynx.

Typical raw data are tiff files from simultaneously recording of up to 50 adults worms at 30 frames per second and 1x magnification. Typical use is by interacting through the notebook which contains the whole pipeline from raw movies to final data. It comprises three stages on analysis which can be done sequentially and are independent. Analyses can be interrupted at the end of each stage after saving the output dataframe.


**1. Step -  Basic object detection**
    This step creates a "_features.json" file which contains a table of objects (worms) detected in each frame.
	It creates also a stack of images that contain a cropped area around each worm.
    
**2. Step - Linking objects into trajectories**
    This results in individual files "_trajectory.json" for each tracked animal.
    
**3. Step - Analyzing the details of object shapes**
    This step is doing the heavy lifting: It extracts centerlines, widths, contours and other object descriptors from the objects

All subsequent analyses steps add 'columns' to the dataframe, and thus features is a subset of trajectories is a subset of results.



## Installation


## Quick Start
### Run PharaGlow on a demo dataset
### Run PharaGlow on your data
#### Raw files requirement
#### Run PharaGlow XX
#### Run PharaGlow in parallel processing

## Code contributors
## Community Support, Developers, & Help
## References
## License



