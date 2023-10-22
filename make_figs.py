import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import sys
from scipy.signal import find_peaks 
import os
import ast
import multiprocessing as mp
#######################################################################################################
#globals
#######################################################################################################

NG_DNA_PER_ML = 29 #Phallen 
NG_DNA_PER_HGE = .0033 #Avanzani and others
HGE_PER_ML = NG_DNA_PER_ML/NG_DNA_PER_HGE
ELIMINATION_RATE = 33 #Avanzani
DETECT_RADIUS = 90
TOTAL_ML_BLOOD = 5000
PLASMA_PER_BLOOD = .55
TOTAL_HGE = PLASMA_PER_BLOOD*TOTAL_ML_BLOOD*HGE_PER_ML
PLOIDY = 2
CM_PER_VOXEL = 2e-2 #10 cells with diameter of 2e-3 cm accross 1 voxel 

#######################################################################################################
#helpers
#######################################################################################################
def display_label(label):
    """display label of form {selection}-driv-{dependence}-r-{santuary radius}-{model}"""
    decode = {"deathbased": "death-based selection", "birthbased": "birth-based selection", "q":"quiescent", "nq": "proliferative", "20": "small","60":"large", "dep": "dependent", "ind":"independent"}
    arr = label.split('-')
    return f"{decode[arr[0]]}, driver-{decode[arr[2]]} invasion with a {decode[arr[4]]} sanctuary radius, {decode[arr[5]]} model"
    
def get_peak_t(t, popsize):
    peaks = find_peaks(popsize)[0]
    min_peak_idx = peaks[0]
    return t[min_peak_idx]

def estimate_3d_population(n_voxels, mutation_factor = 100):
    """function to estimate the number of cells in the tumor given a 2D cross-section of voxels
    This assumes that the 2D tumor is approximately the great circle of a spherical tumor, 
    and that the mutation factor is the number of cells (in 2d) per voxel"""
    n_real = (4/3)*np.pi*np.power(mutation_factor*n_voxels/np.pi, 3/2)
    return n_real

def calc_shedding_rate(time_of_diag, comp_df, set_tumor_fraction):
    """"to be used at at time of diagnosis, assume tumor fraction is set_tumor_fraction, infer shedding rate
    based on global parameters"""
    pop_by_time = comp_df.groupby(['rep','t'])['popsize', 'cellhge'].sum().reset_index()
    detect_size = pop_by_time[pop_by_time['t'].isin(time_of_diag)]
    dbar = detect_size['cellhge']/detect_size['popsize']
    shed_rate = set_tumor_fraction*TOTAL_ML_BLOOD*HGE_PER_ML*ELIMINATION_RATE/dbar/estimate_3d_population(detect_size['popsize'])
    return shed_rate

def estimate_shedding_rate(tf, d1 = .1, detect_size = 3e9):
    return tf*TOTAL_HGE*ELIMINATION_RATE/detect_size/d1

def get_time_of_diag(comp_df):
    """find the time of the first peak of population size, which we assume to be the time of diagnosis, corresponding
    to roughly 3 Billion cells """
    pop_by_time = comp_df.groupby(['rep','t'])['popsize'].sum().reset_index()
    time_of_diag = pop_by_time.groupby('rep').apply(lambda rep: get_peak_t(rep['t'].values, rep['popsize'].values))
    return time_of_diag
   


def calc_tumor_fraction(comp_df, shedding_rate):
    pop_by_time = comp_df.groupby(['rep','t'])[['popsize', 'cellhge']].sum()
    dbar = pop_by_time['cellhge']/pop_by_time['popsize']
    realsize = estimate_3d_population(pop_by_time['popsize']) 
    n_hge = (realsize*shedding_rate*dbar/ELIMINATION_RATE).reset_index()
    n_hge['n_hge'] = n_hge[0]
    n_hge['tf'] = n_hge['n_hge']/TOTAL_HGE
    return n_hge


def clones_to_vafs(data):
    """"""
    if type(data['drivers'].iloc[0]) ==str:    
        data['drivers'] = data['drivers'].transform(ast.literal_eval)
        data['drivers'] = data['drivers'].apply(lambda x: tuple(np.append(x, -1)))
   
    expl = data.explode('drivers')
    expl['scaled_cent'] = (expl['centroid_r']*expl['tissue'])
    
        
    
    
    mut_freqs = expl.groupby(['rep','t', 'drivers', 'realpop'])[['blood', 'tissue']].sum().reset_index()
    
    vaf_centroids = ((expl.groupby(['rep','t','drivers'])['scaled_cent'].sum())/(expl.groupby(['rep','t','drivers'])['tissue'].sum())).reset_index()[0]
    

    mut_freqs['vaf_centroid'] = vaf_centroids
    dbar = (expl.groupby(['rep','t', 'drivers'])['cellhge'].sum()/expl.groupby(['rep','t', 'drivers'])['popsize'].sum()).reset_index()[0]
    mut_freqs['mean_death_rate'] = dbar
    
    return mut_freqs

def resample_blood_vafs(vafs, clones, set_tf, ml_blood_draw = 15):
    """given an array of death rate scaled vafs, total hge, blood
    hge: pd dataframe with a single number of hge for each timepoint for each replicate
    shedrate: series with shedding rate for each replicate 
    blood_vafs: pd dataframe with frequencies of every mutation for timepoint for each replicate
    ml_blood_draw:  constant ml blood drawn"""
    
    shedrate = estimate_shedding_rate(set_tf)
    blood_fraction = ml_blood_draw/TOTAL_ML_BLOOD
    vafs = vafs.set_index(['rep','t'])
    vafs['whole_tumor_mean_dr'] = clones.groupby(['rep','t'])['cellhge'].sum()/clones.groupby(['rep','t'])['popsize'].sum()

    
    vafs = vafs.reset_index() 
    vafs['spatial_mut_hge'] = shedrate*blood_fraction*vafs['mean_death_rate']*vafs['tissue']*vafs['realpop']
    vafs['nonspatial_mut_hge'] = shedrate*blood_fraction*vafs['whole_tumor_mean_dr']*vafs['realpop']*vafs['tissue']
    #vafs = vafs.merge(realpop, on = ['rep','t'])
    #vafs['mut_hge'] *= vafs['realpop']
    vafs['spatial_stoch_mut_hge'] = np.random.poisson(vafs['spatial_mut_hge'].values)
    vafs['nonspatial_stoch_mut_hge'] = np.random.poisson(vafs['nonspatial_mut_hge'].values)
    total_hge = blood_fraction*TOTAL_HGE
    total_stoch_hge = np.random.poisson(total_hge)
    vafs['spatial_vaf'] = vafs['spatial_mut_hge']/total_hge/PLOIDY
    vafs['nonspatial_vaf'] = vafs['nonspatial_mut_hge']/total_hge/PLOIDY
    vafs['spatial_stoch_vaf'] = vafs['spatial_stoch_mut_hge']/total_stoch_hge/PLOIDY
    vafs['nonspatial_stoch_vaf'] = vafs['nonspatial_stoch_mut_hge']/total_stoch_hge/PLOIDY

    return vafs

def get_detectable_vafs(vafs, detection_limit, min_vaf = 0, bins = None):
    largevafs = vafs[vafs['tissue']>min_vaf].copy()
    largevafs['vafdiff'] = largevafs['spatial_vaf']-largevafs['nonspatial_vaf']
    largevafs['nonspatial_detectable'] = largevafs['nonspatial_stoch_vaf'] > detection_limit
    largevafs['spatial_detectable'] = largevafs['spatial_stoch_vaf'] > detection_limit
    detectable = pd.DataFrame()
    detectable[['rep','t','spatial_count']] = largevafs.groupby(['rep','t'])['spatial_detectable'].sum().reset_index()
    detectable['nonspatial_count'] = largevafs.groupby(['rep','t'])['nonspatial_detectable'].sum().reset_index()['nonspatial_detectable']
    detectable['spatial_pct'] = 100*detectable['spatial_count']/((largevafs.groupby(['rep','t'])['spatial_detectable'].count()).reset_index()['spatial_detectable'])
    detectable['nonspatial_pct'] = 100*detectable['nonspatial_count']/((largevafs.groupby(['rep','t'])['nonspatial_detectable'].count()).reset_index()['nonspatial_detectable'])
    detectable['diff'] = detectable['spatial_count']-detectable['nonspatial_count']
    detectable['pctdiff'] = 100*detectable['diff']/detectable['nonspatial_count']
    detectable['ratio'] = detectable['spatial_count']/detectable['nonspatial_count']
    detectable = detectable.set_index(['rep','t'])
    detectable['realpop'] = vafs.groupby(['rep','t'])['realpop'].mean()
    detectable = detectable.reset_index()
    #testrep.groupby(['t'])['realpop'].mean().plot()
    if bins is None:
        bins = np.logspace(np.log10(detectable['realpop'].min()),np.log10(detectable['realpop'].max()),100)
    detectable['realpop_binned'] = pd.cut(detectable['realpop'],bins = bins).apply(lambda x: np.round(x.mid,2)).astype(float)
    return detectable

def get_div(data, xname):
    
    data['tiss_whole_div'] = data.groupby(['rep',xname])['tissue'].transform(inv_simpson_ind)
    data['bl_whole_div'] = data.groupby(['rep',xname])['blood'].transform(inv_simpson_ind)
    return data


def inv_simpson_ind(vec):
    cutoff  = 0
    """input: vector of frequencies.
    output: inverse simpson diversity index"""
    vec = vec/vec.sum()
    vec[vec < cutoff] = 0
    if vec is None or len(vec) == 0 or vec.sum() == 0:
        return 0
    return 1/(np.power(vec, 2).sum())

def get_peak_bias_size(vafs, thresh):
    detectable = get_detectable_vafs(vafs, detection_limit=thresh, bins = 50)
    #detectable = detectable.groupby(['rep','realpop_binned'])['pctdiff'].mean().reset_index()
    detectable = detectable.groupby(['realpop_binned'])['pctdiff'].mean().reset_index()
    return detectable['realpop_binned'].iloc[detectable['pctdiff'].argmax()]


#######################################################################################################
#plotters
#######################################################################################################
def plot_relapse_time(data_dict):
    """plot the distribution of relapse times for proliferative and 
    quiescent (driver-dependent) tumors"""
    
    dfs = []
    for label in data_dict.keys():
        data = data_dict[label]
        tmax = data["data"].groupby('rep')["t"].max().reset_index()
        
        endtimes = pd.DataFrame()
        endtimes['tmax'] = tmax
        endtimes['label'] = label
        dfs.append(endtimes)
    endtimes = pd.concat(dfs)
    sns.boxplot(endtimes, y = "tmax")

def plot_tumor_fraction(data_dict):
    
    fig, (ax1, ax2) = plt.subplots(nrows = 1, ncols = 2)
    fig2, ax21  = plt.subplots(1,1)
    for label, color in zip(data_dict.keys(), ['r-','y-', 'g-', 'b-','r--','y--', 'g--', 'b--']):
        print(label)
        comp = data_dict[label]["data"]
       
        # get the index of median column in the original dataframe
    
        pop_size = estimate_3d_population(comp.reset_index().groupby(['rep','t'])['popsize'].sum()).reset_index().set_index(['rep','t'])#replace with med_df for normalized time plots
        tfs = calc_tumor_fraction(comp, sr)
        tf_col = tfs[['rep','t','n_hge','tf']].set_index(['rep','t'])
        comp = comp.set_index(['rep','t'])
        comp[['n_hge','tf']] = tf_col
        comp['log10_n_hge'] = np.log10(comp['n_hge'])
        comp['log10_tf'] = np.log10(comp['tf'])
        comp['realpop'] = pop_size
        comp['logrealpop'] = np.log10(pop_size)
        binned_time = pd.cut(comp['norm_t'],30).apply(lambda x: np.round(x.mid,2))
        comp['norm_t_binned'] = binned_time.astype(float)
        
        comp_reset = comp.reset_index()
        data_dict[label]["data"] = comp_reset
        maxt = comp_reset.groupby(['rep'])['t'].max()
        maxdiff = comp_reset.groupby(['rep'])['diff'].max()
        meddiff = maxdiff.median()
        medt = maxt.median()
        medoid = maxt.iloc[np.argmin((maxt-medt)**2+(maxdiff-meddiff)**2)]
        """for i in med_df['rep'].unique():
            data = comp_reset[comp_reset['rep']==i]
            if i==0:
                plt.plot(data['t'], data['log10_tf'], color = color, label = label)
            else:
                plt.plot(data['t'], data['log10_tf'], color = color)
            #sns.lineplot(data = data, x = 'norm_t_binned', y = 'log10_tf', errorbar  = 'sd', label = label)"""
        rep = comp_reset[comp_reset['t'] == medoid]['rep'].iloc[0]
        data_dict[label]['chosen_rep'] = rep
        print(rep)
        data = comp_reset[comp_reset['rep']==rep]
        dbar = (data.groupby(['t'])['cellhge'].sum()/data.groupby(['t'])['popsize'].sum()).reset_index()
        
        ax1.plot(data['t'], data['log10_tf'], color, linewidth = 2)
        ax1.set_title("cfDNA tumor fraction")
        #ax2.set_title("population size")
        ax2.plot(data['t'], data['logrealpop'], color, label = f'{plotlabels[data_dict[label]["r"]]},{plotlabels[data_dict[label]["model"]]}', linewidth = 2)
        ax21.plot(dbar['t'],dbar[0],color, label = f'{plotlabels[data_dict[label]["r"]]},{plotlabels[data_dict[label]["model"]]}', linewidth = 2)
    plt.legend()
    ax1.set_ylabel("log10")
    ax1.set_xlabel("time (arbitrary units)")
    plt.savefig("tumor_fraction_plot.png")
    plt.savefig("tumor_fraction_plot.pdf")
    
    plt.show(block = False)

def make_diff_plots(data_dict, min_t = 0, nbins = 100, xname = "realpop"):
    for label in data_dict.keys():
        print(f"plotting c.f. difference \n{display_label(label)}")
        plt.figure(figsize = (5,7))
        
        
        data = data_dict[label]["data"].copy()
        data = data[(data['t']>=min_t) & (data['realpop']<=1e10)&(data['realpop']>=1e7)]

        if xname == "realpop":
            bins = np.logspace(np.log10(data['realpop'].min()),np.log10(data['realpop'].max()),nbins)
            plt.xscale("log")
            plt.xlabel("Tumor Size")
            
        elif xname == "norm_t":
            bins = np.linspace(0,1,nbins)
            plt.xlabel("Normalized Time")
        else:
            print("invalid xname")
            exit()
        xname_binned = xname+"_binned"
        data[xname_binned] = pd.cut(data[xname], bins = bins).apply(lambda x: np.round(x.mid,2)).astype(float)
        for_line = data.groupby(['rep',xname_binned,"genotype"])[['blood','tissue','diff']].mean().reset_index()
        for_line['diffpos'] = for_line['diff'] > 0
        forline = for_line[(for_line['blood']>=.1) | (for_line['tissue']>=.1 )]
        forscatter = data
        sns.lineplot(data = forline, x = xname_binned, y = 'diff',errorbar = "sd", hue = 'diffpos',color = 'C0',legend = False, linewidth = 5)
        sns.scatterplot(data = forscatter, x = xname_binned, y = 'diff', hue = 'norm_age', alpha = .2,palette = 'viridis_r', legend = True)
        
        plt.ylabel("blood clone fraction - tissue clone fraction")
        plt.xticks(fontsize = 'xx-large')
        plt.yticks(fontsize = 'xx-large')
        #plt.ylim([-.5,.5])
        #plt.savefig(f"diff-{label}-{xname.split('_binned')[0]}.png")
        #plt.savefig(f"diff-{label}-{xname.split('_binned')[0]}.pdf")
        
        plt.show()
        plt.close()

def plot_div(data, label, xname):
    print(f'plotting div lineplot for \n{display_label(label)}')
    for_line = data[[xname, 'tiss_whole_div', 'bl_whole_div']]
    for_line.columns = [xname, 'tissue', 'blood']
    for_line_melted = pd.melt(for_line, id_vars = [xname], value_vars = ['tissue','blood']) #hue_kws = {'label': {'tiss_whole_div'}})
    sns.lineplot(data = for_line_melted, x = xname, y = 'value', hue = 'variable', err_style = 'band', errorbar = 'sd', palette = ['b', 'r'], legend = True)
    #for_scatter = data[['realpop', 'tiss_whole_div', 'bl_whole_div']]
    #for_scatter.columns = ['realpop', 'tissue', 'blood']
    #for_scatter_melted = pd.melt(for_scatter, id_vars = ['realpop'], value_vars = ['tissue','blood'])
    #sns.scatterplot(for_scatter_melted, x = 'realpop',y = 'value', hue = 'variable',palette = ['b', 'r'], legend = False, alpha = .1) 
    plt.yticks(fontsize = 'xx-large')
    plt.title('Blood vs Tissue ITH')
    plt.ylabel('Inv. Simpson D')
    if xname == "realpop_binned":
        plt.xscale("log")
        plt.xlabel("Tumor Size")
    else:
        plt.xlabel("normalized time")
    #plt.ylim([0,27])
    #plt.grid(color = 'gray', linestyle = '--')
    
    #plt.legend(labels=['tissue','blood'])
    #plt.savefig(f"diversity-{label}-{xname.split('_binned')[0]}.png")
    #plt.savefig(f"diversity-{label}-{xname.split('_binned')[0]}.pdf")
    
    plt.show()
    plt.close()

def make_diversity_plots(data_dict, min_t = 0, nbins = 50, xname = "realpop"):
    for label in data_dict.keys():
        data = data_dict[label]["data"].copy()
        data = data[(data["t"] >= min_t) & (data["realpop"] <= 1e10)]
        xname_binned = xname+"_binned"
        #reset_index to convert groupby object to DataFrame
    # get the index of median column in the original dataframe
        
        if xname == "realpop":
            bins = np.logspace(np.log10(data['realpop'].min()),np.log10(data['realpop'].max()),nbins)
        elif xname == "norm_t":
            #renormalize time with 1e10 cutoff
            data['norm_t'] = data.groupby('rep')['t'].transform(lambda x: x/x.max())
            bins = np.linspace(0,1,nbins)
        else:
            print("invalid xname")
            exit()
       
        data[xname_binned] = pd.cut(data[xname], bins = bins).apply(lambda x: np.round(x.mid,2)).astype(float)
        mean_resampled = data.groupby(['rep',xname_binned, "genotype"])[['blood','tissue']].mean().reset_index()
       
        to_plot = get_div(mean_resampled, xname_binned)
        plot_div(to_plot, label, xname = xname_binned)

def make_detection_plots(data_dict, y, thresholds = [1e-5,1e-4,1e-3,1e-2], min_t = 0, min_pop = 0,plot_group = '-dep', xname = "realpop",nbins = 100): 
    """makes plots of either number or percentage of detectable driver mutations over either 
    population size or time. One plot contains all driver-dependent or driver-independent simulations for a 
    given detection threshold"""
    assert(plot_group in ["-ind","-dep"])
    for threshold in thresholds:
        plt.figure(figsize = (7,3))
        print(f"making detection plot for plot group {plot_group} with thresh. {threshold}")        
        for label in data_dict.keys():
            if label.split('driv')[1].startswith(plot_group):
                print(label)
                data = data_dict[label]["vafs"].copy()
                data = data[(data['t']>min_t) & (data['realpop'] >=min_pop)]
                
                detectable = get_detectable_vafs(data, detection_limit = threshold)
                
                xname_binned = xname+"_binned"
                if xname == "realpop":
                    bins = np.logspace(np.log10(data['realpop'].min()),np.log10(data['realpop'].max()),nbins)
                    plt.xscale("log")
                    plt.xlim([1e7,1e10])
                elif xname == "norm_t":
                    #renormalize time with 1e10 cutoff
                    detectable['norm_t'] = detectable.groupby('rep')['t'].transform(lambda x: x/x.max())
                    bins = np.linspace(0,1,nbins)
                    plt.xlim([0,1])
                else:
                    print("invalid xname")
                    exit()
                detectable[xname_binned] = pd.cut(detectable[xname], bins = bins).apply(lambda x: np.round(x.mid,2)).astype(float)
                detectable = detectable.groupby(['rep',xname_binned])[y].mean().reset_index()

                sns.lineplot(data = detectable, x = xname_binned, y = y,label = label, errorbar = 'sd',legend=False,linewidth = 2)
                
                plt.title(f"detection limit = {threshold}")
               
                plt.ylabel(f" {y} detectable driver mutations")
                plt.xlabel("tumor size")
                plt.xticks(fontsize = 'xx-large')
                plt.yticks(fontsize = 'xx-large')
                
        #plt.savefig(f"detection{plot_group}-{xname}-{y}-lim-{threshold}.png")
        #plt.savefig(f"detection{plot_group}-{xname}-{y}-lim-{threshold}.pdf")
        plt.show()
        plt.close() 
        
def make_vafratio_plots(data_dict):
    plt.figure(figsize = (5,7))
    for label in data_dict.keys():
        print(label)
        comptfs = data_dict[label]['data']
        vafs = data_dict[label]["vafs"]
        vafs['log10pop'] = np.log10(vafs['realpop'])
        vafs['vafratio'] = vafs['spatial_vaf']/vafs['nonspatial_vaf']
        bins = np.logspace(np.log10(comptfs['realpop'].min()),np.log10(comptfs['realpop'].max()),50)
        vafs['realpop_binned'] = pd.cut(vafs['realpop'], bins = bins).apply(lambda x: np.round(x.mid,2)).astype(float)
        vafs['logrealpop_binned'] = np.log10(vafs['realpop_binned'])
        vafs['>=1'] = vafs['vafratio'] >= 1
        toplot = vafs[((comptfs['blood']>=.1) | (comptfs['tissue']>=.1 )) & (vafs['t']>0)]
        sns.lineplot(data = toplot, x = 'realpop_binned', y = 'vafratio',errorbar = "sd", label = label, legend = False, hue = ">=1")
        plt.xscale("log")
        plt.yscale("log",nonpositive = 'clip', base = 2)
        plt.ylabel("Blood VAF/Tissue VAF")
        plt.xlabel("tumor size")
        plt.xticks(fontsize = 'xx-large')
        plt.yticks(fontsize = 'xx-large')
        plt.savefig("")
    plt.show()

def make_vaf_centroid_plot(data_dict, min_t = 100, outdir = 'fig-data'):
    
    for label in data_dict.keys():
        
        for thresh in [1e-4, 1e-3]:
            print(f"plotting spatial vaf distribution at peak bias for \n{display_label(label)}, detection limit = {thresh}")
            comp = data_dict[label]["data"] #comparison data (clone fractions)
            vafs = data_dict[label]["vafs"] #vaf data
            r = float(label.split("-")[4]) #sanctuary radius
            max_bias_size = get_peak_bias_size(vafs[vafs['t']>min_t], thresh)#get max bias population size 
            #print(max_bias_size)
            to_plot = []
            vaf_subset = []
            for rep in range(50):
                c = comp[(comp['rep']==rep) & (comp['t']> min_t)]
                v = vafs[(vafs['rep']==rep) & (vafs['t']> min_t)]
                tpop = c['t'].iloc[((c['realpop']-max_bias_size)**2).argmin()]
                to_plot.append(c[c['t']==tpop])
                vaf_subset.append(v[v['t']==tpop])
            compsub = pd.concat(to_plot)
            vafsub = pd.concat(vaf_subset)
            compsub['binned_cent'] = pd.cut(compsub['centroid_r'], 100).apply(lambda x: np.round(x.mid,2))
            #compsub['drivers'] = compsub['drivers'].transform(ast.literal_eval)
            compsub['drivers'] = compsub['drivers'].apply(lambda x: tuple(np.append(x, -1)))
            expl = compsub.explode('drivers')
            expl['scaled_cent'] = (expl['centroid_r']*expl['tissue'])
            vaf_centroids = (expl.groupby(['rep','t','drivers'])['scaled_cent'].sum()/expl.groupby(['rep','t','drivers'])['tissue'].sum())
            vafsub = vafsub.set_index(['rep','t','drivers'])
            vafsub['centroid'] = vaf_centroids
            vafsub = vafsub.reset_index()
            vafsub['vafratio'] = vafsub['spatial_vaf']/(vafsub['nonspatial_vaf']+1e-10)
            vafsub['1/vafratio'] = vafsub['nonspatial_vaf']/(vafsub['spatial_vaf']+1e-10)
            vafsub['centroid_real'] = vafsub['centroid']*CM_PER_VOXEL
            vafsub['fold'] = np.log2(vafsub['vafratio']+1e-10)
            vafsub['negfold'] = -vafsub['fold']
            
            maxsize = vafsub[vafsub['spatial_vaf']>1e-6]['vafratio'].max()
            
            vafsub['binned_cent'] = pd.cut(vafsub['centroid'], 20).apply(lambda x: np.round(x.mid,2)).astype(float)*2e-2
            sns.scatterplot(data = vafsub[vafsub['spatial_vaf']>1e-6], x = 'centroid_real', y = 'spatial_vaf', color = 'red', alpha = .2, size = "vafratio", size_norm = (.5,maxsize),sizes = (10,100) , legend = "brief")
            sns.scatterplot(data = vafsub[vafsub['nonspatial_vaf']>1e-6], x = 'centroid_real', y = 'nonspatial_vaf', color = 'blue', alpha = .2, size = "1/vafratio", size_norm = (.5,maxsize), sizes = (10,100),  legend = None)#, size = "1/vafratio", legend = None)
            plt.legend(title = f"VAF Ratio")
            plt.yscale("log")
            plt.ylim([1e-6, 1e-1])
            plt.hlines([1e-5, 1e-4, 1e-3],xmin = 0, xmax = vafsub['centroid_real'].max(), linestyles = ['--'], colors = ['k'], label = "detection limit")
            plt.vlines([r*2e-2], ymin = 1e-6,ymax = 1e-1, colors = ["gray"], label = "sanctuary radius")
            plt.ylabel("VAF")

            plt.xlabel("distance from tumor center (cm)")
            plt.xticks(fontsize = 'xx-large')
            plt.yticks(fontsize = 'xx-large')
            #plt.savefig(f"{outdir}/vaf-snapshot-{label}-{thresh}.pdf")
            plt.show()
           

def load_data_files(selection = "birthbased", calibration_tumor_fraction = 1e-4):
    set_tf = calibration_tumor_fraction
    sr = estimate_shedding_rate(set_tf)
    data_dict = {}
    print("loading data")
    for dependence in ["dep","ind"]:
        for r in [20,60]:
            for model in ["q","nq"]:
                label = f'{selection}-driv-{dependence}-r-{r}-{model}'
                datadir = f'figure-data/{label}'
                readin = pd.read_csv(f'{datadir}/results.csv', index_col = [0])
                data_dict[label] = {"datadir":datadir, "data": readin}
                data_dict[label]['model'] = model
                data_dict[label]['r'] = r
                print(f'loaded {label}')
    print("inferring tumor fraction and VAFs")
    for label in data_dict.keys():
        comp = data_dict[label]["data"]
        
        # get the index of median column in the original dataframe
    
        pop_size = estimate_3d_population(comp.reset_index().groupby(['rep','t'])['popsize'].sum()).reset_index().set_index(['rep','t'])#replace with med_df for normalized time plots
        tfs = calc_tumor_fraction(comp, sr)
        tf_col = tfs[['rep','t','n_hge','tf']].set_index(['rep','t'])
        comp = comp.set_index(['rep','t'])
        comp[['n_hge','tf']] = tf_col
        comp['log10_n_hge'] = np.log10(comp['n_hge'])
        comp['log10_tf'] = np.log10(comp['tf'])
        comp['realpop'] = pop_size
        comp['logrealpop'] = np.log10(pop_size)
        binned_time = pd.cut(comp['norm_t'],30).apply(lambda x: np.round(x.mid,2))
        comp['norm_t_binned'] = binned_time.astype(float)
        
        comp_reset = comp.reset_index()
        data_dict[label]["data"] = comp_reset
        #tdiag = get_time_of_diag(comp)
        sr = estimate_shedding_rate(set_tf)#calc_shedding_rate(tdiag, comp, set_tf)
        data_dict[label]['shedrates'] = sr
        df = data_dict[label]["data"]
        #df.to_csv(f'{data_dict[label]["datadir"]}/comp-tfs.csv')
        original_vafs = clones_to_vafs(df)
        vafs = resample_blood_vafs(original_vafs, df, set_tf = set_tf)
        data_dict[label]['vafs'] = vafs
        #print(f"writing {label} to files")
        #vafs.to_csv(f'{data_dict[label]["datadir"]}/comp-vafs.csv')

    print("done")
    return data_dict

def centroid_vaf_plot(comp, vafs):
    pass#merged = pd.merge(vafs, comp, how = 'left', on = 'drivers',

def load_data_files_birthbased():
    print("loading data")
    data_dict = {}
    for r in [20,60]:
        for model in ["q","nq"]:
            label = f'r-{r}-{model}-d' #easier labeling than "experiment1"
            datadir = f'/home/trachman/sim2/treatment-based/experiment1/analysis-{f"r-{r}-{model}"}/'
            comptfs = pd.read_csv(f'{datadir}/comp-tfs.csv')
            vafs =  pd.read_csv(f'{datadir}/comp-vafs.csv')
            data_dict[label] = {"datadir":datadir, "data": comptfs, "vafs": vafs}
            data_dict[label]['model'] = model
            data_dict[label]['r'] = r
            print(f"loaded {label}")
    for r in [20,60]:
        for model in ["q","nq"]:
            label = f'r-{r}-{model}-i'
            datadir = f'/home/trachman/sim2/treatment-based/experiment3/analysis-{f"r-{r}-{model}"}/'
            comptfs = pd.read_csv(f'{datadir}/comp-tfs.csv')
            vafs =  pd.read_csv(f'{datadir}/comp-vafs.csv')
            data_dict[label] = {"datadir":datadir, "data": comptfs, "vafs": vafs}
            data_dict[label]['model'] = model
            data_dict[label]['r'] = r
            print(f"loaded {label}")
    print("done")
    return data_dict
def load_data_files_deathbased():
    print("loading data")
    data_dict = {}
    for r in [20,60]:
        for model in ["q","nq"]:
            label = f'death-r-{r}-{model}-d' 
            datadir = f'/home/trachman/sim2/treatment-based/experiment2/analysis-{f"r-{r}-{model}"}/'
            comptfs = pd.read_csv(f'{datadir}/comp-tfs.csv')
            vafs =  pd.read_csv(f'{datadir}/comp-vafs.csv')
            data_dict[label] = {"datadir":datadir, "data": comptfs, "vafs": vafs}
            data_dict[label]['model'] = model
            data_dict[label]['r'] = r
            print(f"loaded {label}")
    for r in [20,60]:
        for model in ["q","nq"]:
            label = f'death-r-{r}-{model}-i'
            datadir = f'/home/trachman/sim2/treatment-based/experiment4/analysis-{f"r-{r}-{model}"}/'
            comptfs = pd.read_csv(f'{datadir}/comp-tfs.csv')
            vafs =  pd.read_csv(f'{datadir}/comp-vafs.csv')
            data_dict[label] = {"datadir":datadir, "data": comptfs, "vafs": vafs}
            data_dict[label]['model'] = model
            data_dict[label]['r'] = r
            print(f"loaded {label}")
    print("done")
    return data_dict
def test_load():
    print("loading data")
    data_dict = {}
    label = f'r-20-nq-i'
    datadir = f'/home/trachman/sim2/treatment-based/experiment3/analysis-r-60-nq/'
    comptfs = pd.read_csv(f'{datadir}/comp-tfs.csv')
    vafs =  pd.read_csv(f'{datadir}/comp-vafs.csv')
    data_dict[label] = {"datadir":datadir, "data": comptfs, "vafs": vafs}
    data_dict[label]['model'] = "nq"
    data_dict[label]['r'] = 60
    print(f"loaded {label}")
    return data_dict

if __name__ == "__main__":
    #make_data_files()
    data_dict = load_data_files_deathbased()

    make_diversity_plots(data_dict, min_t = 0, nbins = 50, xname = "norm_t")
    make_diff_plots(data_dict, min_t = 0, nbins = 50, xname = "norm_t")

    make_diversity_plots(data_dict, min_t = 100, nbins = 50, xname = "realpop")
    make_diff_plots(data_dict, min_t = 0, nbins = 50, xname = "realpop")

    for plot_group in  ["-ind","-dep"]:
        for xname, min_t, min_pop in zip(["norm_t","realpop"],[0,100 if plot_group == '-dep' else 0],[0,1e7]):
            for y in ["spatial_count", "spatial_pct", "pctdiff","nonspatial_count"]:
                print(f"plotting {y}")
                make_detection_plots(data_dict, thresholds = [1e-5, 1e-4, 1e-3], y = y, min_t = min_t, min_pop = min_pop, plot_group = plot_group,xname = xname, nbins = 50)

    #make_detection_plots(data_dict, thresholds = [1e-4, 1e-3, 1e-2], y = "pctdiff",min_t = 100, min_pop = 1e7, plot_group = '-d',xname = "realpop", nbins = 50)
    #make_detection_plots(data_dict, thresholds = [1e-4, 1e-3, 1e-2], y = "pctdiff",min_t = 100, min_pop = 1e7, plot_group = '-i',xname = "realpop", nbins = 50)
    #make_detection_plots(data_dict, thresholds = [1e-4, 1e-3, 1e-2], y = "pctdiff",min_t = 0, min_pop = 0, plot_group = '-d',xname = "norm_t", nbins = 50)
    #make_detection_plots(data_dict, thresholds = [1e-4, 1e-3, 1e-2], y = "pctdiff",min_t = 0, min_pop = 0, plot_group = '-i',xname = "norm_t", nbins = 50)"""