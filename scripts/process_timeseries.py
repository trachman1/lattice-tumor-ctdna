"""Script to take in dataframe with summary of tumor information for multiple replicates at multiple timepoints. 
See simulation.tumor_summary for details on the format of the csv. 
    Inputs: a file string for the csv 
    Outputs: plots aggregated by replicate measuring the distortion of clonal fraction over time in ctDNA compared to tissue. The estimates of ctDNA fraction are based on the crude assumption 
    that the clone frequencies in the blood are the weighted average of the tissue frequencies, scaled by the cell death rate. """
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
#read file 
#arguments:  path shed1 (optional), shed2 (optional), radius (optional)
DATAPATH = sys.argv[1]
OUTDIR_LABEL = sys.argv[2]
SHED1 = 1#float(sys.argv[2]) 
SHED2 = 1#float(sys.argv[3]) 
RADIUS = 0#float(sys.argv[4]) #radius where shedding rate changes...does not need to be the same as the death rate radius

print('loading data')

OUTDIR = f'demo-output/postprocessed'
SAVEFIGFOLDER = os.path.join(OUTDIR,'figs')
Path(OUTDIR).mkdir(parents=True, exist_ok=True)
Path(SAVEFIGFOLDER).mkdir(parents=True, exist_ok=True)

#titlestring = 'expon. growth death-based selection'
TITLESTRING = 'bdg moving edge birth-based selection'
MIN_FREQUENCY = 0
N_CLONES_TO_PLOT = 100

timedata = pd.read_csv(DATAPATH)


timedata['is_outer'] = timedata['r'] > RADIUS
timedata['shed_rate'] = SHED1
#timedata['shed_rate'][timedata['is_outer']] = SHED2
timedata['cell_hge'] = timedata['shed_rate']*timedata['death_rate']#/(timedata['birth_rate']-timedata['death_rate']+33)
grouped_rep_t_gen = timedata.groupby(['rep','t','genotype','drivers'])
grouped_rep_t = timedata.groupby(['rep','t'])




#make comparison file
print('making comparison file')
tissue = grouped_rep_t_gen['genotype'].count()/grouped_rep_t['genotype'].size()
blood = grouped_rep_t_gen['cell_hge'].sum()/grouped_rep_t['cell_hge'].sum()
popsize = grouped_rep_t_gen['genotype'].count()
cellhge = grouped_rep_t_gen['cell_hge'].sum()
comparison = tissue.compare(blood, keep_shape = True, keep_equal = True)
comparison.columns = ['tissue','blood']
comparison['diff'] = (comparison['blood'] - comparison['tissue'])
comparison['pcterr'] = 100*(comparison['blood'] - comparison['tissue'])/comparison['tissue']
comparison['logprob'] = np.log10((1e-10+comparison['blood'])/(1e-10+comparison['tissue']))
comparison['popsize'] = popsize
comparison['cellhge'] = cellhge
comp_reset = comparison.reset_index()
comp_reset['cell_ID'] = timedata['cell_ID']
comp_reset['is_outer'] = timedata['is_outer']
scaled_time = comp_reset.groupby('rep')['t'].transform(lambda x: x/x.max())
binned_time = pd.cut(scaled_time, bins = 20).apply(lambda x: np.round(x.mid,2))
comp_reset['norm_t'] = scaled_time
comp_reset['norm_t_binned'] = binned_time.astype(float)
age = comp_reset.groupby(['rep','genotype'])['t'].transform(lambda x: (x - x.min()))
comp_reset['norm_age'] = age
try:
    comp_reset['norm_age'] = comp_reset.groupby('rep').apply(lambda x: x['norm_age']/x['t'].max()).reset_index()['norm_age']
except(KeyError):
    #hot fix for case of only 1 replicate 
    comp_reset['norm_age'] = comp_reset['norm_age']/comp_reset['t'].max()
#get mean r and sd per clone 
rmean= grouped_rep_t_gen['r'].mean().reset_index()['r']
rstd  = grouped_rep_t_gen['r'].std().reset_index()['r']
comp_reset['r_mean'] = rmean
comp_reset['r_std'] = rstd
centroid_x = grouped_rep_t_gen['x'].mean().reset_index()['x']
centroid_y = grouped_rep_t_gen['y'].mean().reset_index()['y']
comp_reset['centroid_x'] = centroid_x
comp_reset['centroid_y'] = centroid_y
comp_reset['centroid_r'] = np.sqrt(centroid_x**2+centroid_y**2)





#plot clone frequencies for each replicate
print('plotting clone frequencies for each replicate')
replist = comp_reset['rep'].unique()
replist.sort()

plt.figure(figsize = (7,7))
for rep in replist:
     #get comparison as 2d dataframe
    current = comp_reset[comp_reset['rep']==rep] #slice by replicate
    max_freqs = current.groupby('genotype')['blood'].max()
    top_gens = max_freqs.sort_values()[-N_CLONES_TO_PLOT:]
    pop_size = current.groupby('t')['popsize'].sum().reset_index()

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax2.set_ylabel('population size')
    ax1.set_ylabel('clonal fraction')
    for gen in top_gens.index:
        data = current[current['genotype']==gen]
        time = data['t']
        line,  = ax1.plot(time, data['blood'], label = f'clone # {gen} (blood)',linestyle = '--')
        ax1.plot(time, data['tissue'], label = f'clone # {gen} (tissue)', color = line.get_color())
    ax2.plot(pop_size['t'], pop_size['popsize'], color = 'black', label = 'pop. size')
    plt.title(f'rep {rep}')
    plt.legend()
    ax1.set_xlabel('time (a.u.)')
    plt.tight_layout()
    plt.savefig(f'{SAVEFIGFOLDER}/timeplot_rep_{rep}.png')
    plt.show(block = False)
    plt.close()
print('done')
################################################
#save comparison file
comp_reset.to_csv(os.path.join(OUTDIR,'postprocessed.csv'))
print('done with everything')
