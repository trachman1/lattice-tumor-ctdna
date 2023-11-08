# lattice-tumor-ctdna
## This repo includes the code used to generate the results in our paper:
dependencies: `python3 v11, numpy, pandas, matplotlib, seaborn` 
## Tumor simulation code:
`classes.py`: class definitions for the lattice simulation\
`main.py`: wrapper functions to configure parameters and call the simulator\
`utils.py`: various helper functions and plotters\
`demo.ipynb`: notebook showing an example run of the simulator and the resulting trajectory of ctDNA, tissue clone fractions and overall population. The results are stored in `demo-output`. The file `simulation-results.csv` contain data for each individual cell at each saved timepoint. The file `demo-output/postprocessed/postprocessed.csv` summarizes the results for each clone as described in the Data section. 
## Code to reproduce figures:
`make_figs.py`: code to produce subfigures used in all results and supplementary material\
`code-for-figures.ipynb`: notebook that loads raw data and runs the methods in `make_figs.py` to produce all subfigures. 
## Data:
The raw data used to produce all results in the paper are stored in `figure-data.tgz` which is 900MB and 3.6GB unpacked. 

The folder `figure-data` contains all raw simulation output in .csv format.
Each folder has a naming structure identifying the simulation scenario:
\<type of selection\>-driv-\<dependent or independent\>-r-\<sanctuary radius\>-<\quiescent or non-quiescent\> 

so that "birthbased-driv-dep-r-60-nq" is a simulation run with selection on birth, driver-dependent invasion, sanctuary site radius of 60 voxels, non-quiescent (proliferative). 

Each folder contains a single .csv called `results.csv`

Each row in `results.csv` contains the data for a specific clone per replicate per timestamp. 
The columns are: 
	**rep**: replicate number\ 
	**t**: simulation timestamp\ 
	**genotype**: genotype number \
	**tissue**: tissue frequency \
	**blood**: blood frequency \
	**popsize**: population of the overall tumor \
	**cellhge**: death rate scaled population size \
	**diff**: difference between blood and tissue clone fraction \
	**norm_t**: normalized timestamp \
	**norm_t_binned**: binned normalized timestamp \
	**r_mean**: mean radius of clone from tumor center \
	**r_std**: standard deviation of clone from tumor center \
	**centroid_x**: the mean x coordinate of the tumor clone \
	**centroid_y**: the mean y coordinate of the tumor clone \
	**centroid_r**: the mean displacement of the tumor clone from the center \ 
	**norm_age**: normalized age of clone \
	**drivers**: tuple of driver mutation IDs in this clone` 
