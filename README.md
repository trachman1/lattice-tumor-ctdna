
### Modeling the effect of spatial structure on solid tumor evolution and ctDNA composition 
Thomas Rachman<sup>1,2</sup>, David Bartlett<sup>3</sup>, William Laframboise<sup>3</sup>, Patrick Wagner<sup>3</sup>, Russell Schwartz<sup>1</sup>, Oana Carja<sup>1</sup> 

<sup>1</sup>Computational Biology Department, School of Computer Science, Carnegie Mellon University, Pittsburgh, PA, USA;
<sup>2</sup>Joint Carnegie Mellon University-University of Pittsburgh Ph.D. Program in Computational Biology;
<sup>3</sup>Allegheny Cancer Institute, Allegheny Health Network, Pittsburgh PA

### Abstract
Circulating tumor DNA (ctDNA) monitoring, while sufficiently advanced to reflect tumor evolution in real time and inform on cancer diagnosis, treatment, and prognosis, mainly relies on DNA that originates from cell death via apoptosis or necrosis. In solid tumors, chemotherapy and immune infiltration can induce spatially variable rates of cell death, with the potential to bias and distort the clonal composition of ctDNA. Using a stochastic evolutionary model of boundary-driven growth, we study how elevated cell death on the edge of a tumor can simultaneously impact driver mutation accumulation and the representation of tumor clones and mutation detectability in ctDNA. We describe conditions in which invasive clones end up over-represented in ctDNA, clonal diversity can appear elevated in the blood, and spatial bias in shedding can inflate subclonal variant allele frequencies (VAFs). Additionally, we find that tumors that are mostly quiescent can display similar biases, but are far less detectable, and the extent of perceptible spatial bias strongly depends on sequence detection limits. Overall, we show that spatially structured shedding might cause liquid biopsies to provide highly biased profiles of tumor state.  While this may enable more sensitive detection of expanding clones, it could also increase the risk of targeting a subclonal variant for treatment. Our results indicate that the effects and clinical consequences of spatially variable cell death on ctDNA composition present an important area for future work.
## Tumor simulation code:
 dependencies: `python 3.11, numpy, scipy, pandas, matplotlib, seaborn` \
`classes.py`: class definitions for the lattice simulation\
`main.py`: wrapper functions to configure parameters and call the simulator\
`utils.py`: various helper functions and plotters\
`demo.ipynb`: notebook showing an example run of the simulator and the resulting trajectory of ctDNA, tissue clone fractions and overall population. The results are stored in `demo-output`. The file `simulation-results.csv` contain data for each individual cell at each saved timepoint. The file `demo-output/postprocessed/postprocessed.csv` summarizes the results for each clone in the same way described in the Data section. 
## Code to reproduce figures:
`make_figs.py`: code to produce subfigures used in all results and supplementary material\
`code-for-figures.ipynb`: notebook that loads raw data and runs the methods in `make_figs.py` to produce all subfigures. 
## Data:
The raw data used to produce all results in the paper are stored in `figure-data.tgz` which is 900MB and 3.6GB unpacked. 

The folder `figure-data` contains all raw simulation output in .csv format.
Each folder has a naming structure identifying the simulation scenario:
\<type of selection\>-driv-\<dependent or independent\>-r-\<sanctuary radius\>-\<quiescent or non-quiescent\> 

so that "birthbased-driv-dep-r-60-nq" is a simulation run with selection on birth, driver-dependent invasion, sanctuary site radius of 60 voxels, non-quiescent (proliferative). 

Each folder contains a single .csv called `results.csv`

Each row in `results.csv` contains the data for a specific clone per replicate per timestamp. 
The columns are:\
	**rep**: replicate number\
	**t**: simulation timestamp\ 
	**genotype**: genotype number\
	**tissue**: tissue frequency\
	**blood**: blood frequency\
	**popsize**: population of the overall tumor\
	**cellhge**: death rate scaled population size\
	**diff**: difference between blood and tissue clone fraction\
	**norm_t**: normalized timestamp\
	**norm_t_binned**: binned normalized timestamp\
	**r_mean**: mean radius of clone from tumor center\
	**r_std**: standard deviation of clone from tumor center \
	**centroid_x**: the mean x coordinate of the tumor clone \
	**centroid_y**: the mean y coordinate of the tumor clone \
	**centroid_r**: the mean displacement of the tumor clone from the center   \ 
	**norm_age**: normalized age of clone \
	**drivers**: tuple of driver mutation IDs in this clone \
