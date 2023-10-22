#file with end-user functions to set parameters and run simulation 
import numpy as np
import pickle
import classes
import os
from pathlib import Path
import sys
import json
from multiprocessing import Pool
import itertools
import utils

#absolute path to parameter file 
PARAMS_PATH = '' #change to program directory 
#defualt parameters: 
DIM = 2
INIT_BIRTH_RATE = 1#np.log(2)
DRIVER_ADVANTAGE = 0
INIT_DEATH_RATE = .95*INIT_BIRTH_RATE
PAD = 5 #fraction of radius to pad boundary 
MUTATOR_FACTOR = 0
BOUNDARY = 300
MAX_ITER = int(1e9)
MAX_POP = 50000
PUSH = 0
PASSEN_RATE = 0#.02 #taken from Waclaw et al. 2015 
DRIVER_RATE = 1e-2#4e-5
MUTATOR_RATE = 0#4e-6 #need to implement way of increasing mutation probability in individual cell lines 




def get_kwargs_from_file(path):
    """
    Retrieves arguments from a file.
    
    Parameters:
    - path: Path to the file containing arguments.
    
    Returns: 
    - A dictionary containing the arguments if the path is valid, otherwise an empty dictionary.
    """
    if path is None:
        return {}

    ext = path.split('.')[-1]
    obj = None
    if ext == 'pkl':
        obj =  classes.load_object(path)
    if ext == 'txt' or ext == 'json':
        with open(path,'r') as f:
            obj = json.load(f)
        f.close()
    try:
        assert(type(obj) is dict)
    except(AssertionError):
        print(f'expected dict but got {type(obj)}')
        print('exiting...')
        sys.exit() 
    return obj



def config_params(kwargs):
    """wrapper to handle user parameters. Accepts a file or a set of keyword arguments, sets defaults, runs santity checks,
    returns configured parameters used to initialize the simulation object. 
     """

    if 'dim' not in kwargs:
            kwargs['dim']=DIM #dimension of simulation can be 2 or 3
    if 'n_cells' not in kwargs:
            kwargs['n_cells'] = MAX_POP 
    if 'neighborhood' not in kwargs:
            kwargs['neighborhood'] = 'moore'
    if kwargs['neighborhood']!= 'moore':
        kwargs['neighborhood'] = 'von_neumann'
    
    #if 'driver_advantage' not in kwargs:
    #        kwargs['driver_advantage'] = DRIVER_ADVANTAGE
    if 'select_birth' not in kwargs:
            kwargs['select_birth'] = False #seleciton actions on death by default and can only act on 1 
    if 'driver_rate' not in kwargs:
            kwargs['driver_rate'] = DRIVER_RATE
    if 'mutator_factor' not in kwargs:
        kwargs['mutator_factor']= MUTATOR_FACTOR
    if 'max_iter' not in kwargs:
        kwargs['max_iter'] = MAX_ITER
    if 'passen_rate' not in kwargs:
        kwargs['passen_rate'] = PASSEN_RATE
    if 'mutator_rate' not in kwargs:
        kwargs['mutator_rate'] = MUTATOR_RATE
    if 'progression' not in kwargs:
        kwargs['progression'] = None
    if 'exp_path' not in kwargs:
        kwargs['exp_path'] = PARAMS_PATH
    if 'reps' not in kwargs:
        kwargs['reps'] = 1

    if 'first_rep' not in kwargs and 'last_rep' not in kwargs:
        kwargs['first_rep'] = 0
        kwargs['last_rep'] = kwargs['reps']-1
    elif 'last_rep' not in kwargs and 'first_rep' in kwargs:
        kwargs['last_rep'] = kwargs['first_rep'] + kwargs['reps']-1
    else:
        kwargs['reps'] = kwargs['last_rep'] - kwargs['first_rep']+1

    if 'n_migrations' not in kwargs:
        kwargs['n_migrations'] = 0

    if 'time_save_interval' not in kwargs: 
        kwargs['time_save_interval'] = np.infty
    if 'cell_save_interval' not in kwargs: 
        kwargs['cell_save_interval'] = np.infty
    if 'only_save_extremes' not in kwargs: #whether or not to save only the min/max difference at each save interval
        kwargs['only_save_extremes'] = False
    #check growth params    
    if 'dr_params' not in kwargs and 'dr_function' not in kwargs:
        kwargs['dr_function'] = 'default'
        kwargs['dr_params'] = {'init_rate': kwargs['init_death_rate']}

    elif 'dr_params' in kwargs and 'dr_function' not in kwargs:
        print('must specify dr function if dr params are specified\nexiting...')
        sys.exit()
    elif 'dr_params' not in kwargs:
        kwargs['dr_params'] = {}
    
    if kwargs['dr_function'] == 'resistance_model':
        assert(kwargs['mutate_function'] == 'resistance_model')

    if 'br_params' not in kwargs and 'br_function' not in kwargs:
        kwargs['br_function'] = 'default'
        kwargs['br_params'] = {'init_rate': kwargs['init_birth_rate']}

    if 'br_params' in kwargs and 'br_function' not in kwargs:
        print('must specify dr function if br params are specified\nexiting...')
        sys.exit()

    elif 'br_params' not in kwargs:
        kwargs['br_params'] = {}
    
    if 'mutate_params' not in kwargs:
        kwargs['mutate_function'] = 'default'
        kwargs['mutate_params'] = {}

    if kwargs['mutate_function'] == 'resistance_model':
        assert(kwargs['dr_function'] == 'resistance_model')
        assert(kwargs['driver_advantage']==0) 
        #print(f'resistance model selected. Death rate params are: {kwargs["dr_params"]}')
        
    if 'push_params' not in kwargs:
        kwargs['push_function'] = 'default'
        kwargs['push_params'] = {}

    if 'model_params' not in kwargs:
        kwargs['model_params'] = {}

    print('starting sanity checks...')
    #check replicate number validity
    if kwargs['first_rep'] < 0 or kwargs['last_rep'] < 0 or kwargs['reps'] < 0:
        print('first_rep, last_rep, and reps must be nonnegative \n exiting...')
        sys.exit() 
    if kwargs['last_rep'] - kwargs['first_rep'] +1 != kwargs['reps']:
        print(f'last_rep ({kwargs["last_rep"]}) - first_rep ({kwargs["first_rep"]}) +1 must equal number of reps ({kwargs["reps"]})  \n exiting...')
        sys.exit()
    
        
   
    #set dependent parameters, sanity checks
    r = utils.calc_radius(kwargs['n_cells'],kwargs['dim'])
    kwargs['boundary'] = int(r*(1+PAD)) #set boundary to be larger than the minimal lattice size to surround a spherical tumor with the max number of cells 

    Path(kwargs['exp_path']).mkdir(parents=True, exist_ok=True)
    
    return kwargs
   

def simulateTumor(kwargs):
    """function to configure and run simulation with given parameters. Can include multiple replicates. 
    Returns either a single simulation object or a list of simulation objects. Restarts simulations max_attempts times in the case of extinctions."""
    max_attempts = 10000 #maximum number of times to try a particular rep before moving on
    params = config_params(kwargs)
    first_rep = params['first_rep']
    cur_rep = first_rep
    sim_list = []
    last_rep = params['last_rep']
    attempts = 0
    """print(f'params are:')
    for k in params.keys():
        print(f'{k}:\n\t{params[k]}\n')
        
    print('\n\n')"""
    
    print('starting simulation...')
    print(cur_rep)
    while cur_rep < last_rep +1: 
        sim = classes.Simulation(params)
        while sim.tumor.N < 2 and attempts < max_attempts:
           sim = classes.Simulation(params)
           print(f'trying rep {cur_rep}')
           sim.run(cur_rep) 
           attempts +=1
        cur_rep+=1
        sim_list.append(sim)
    print('done!')

    return sim_list[0] if len(sim_list)==1 else sim_list

def single_run(cur_rep,kwargs):

    max_attempts = 10000
    attempts = 0
    params = config_params(kwargs)
    sim = classes.Simulation(params)
    while sim.tumor.N < 2 and attempts < max_attempts:
        sim = classes.Simulation(params)
        print(f'trying rep {cur_rep}')
        sim.run(cur_rep) 
        attempts +=1
    return sim
        
def simulateTumor_mp(kwargs):
    params = config_params(kwargs)
    first_rep = params['first_rep']
    last_rep = params['last_rep']
    replist = np.arange(first_rep, last_rep+1)
    print('starting simulation...')
    n_workers = os.cpu_count()
    sim_list = Pool(n_workers).starmap(single_run,replist,itertools.repeat(params))
    print('done!')
    return sim_list[0] if len(sim_list)==1 else sim_list

if __name__ == '__main__': 
    try:
        config_file = sys.argv[1]
    except(IndexError):
        config_file = None
    try: 
        display = sys.argv[2]
    except(IndexError):
        display = False
    kwargs = get_kwargs_from_file(config_file)
    out = simulateTumor(**kwargs)
    if display:
        print('plotting tumor...')
        import simulation
        import seaborn as sns
        import matplotlib.pyplot as plt
        g = simulation.get_fitness_graph(out.tumor)
        sns.heatmap(g)
        plt.show()
        simulation.plot_growth(out.tumor)
        plt.show()
        summary = simulation.tumor_summary(out.tumor)
        simulation.tumor_scatter(summary['x'], summary['y'], summary['death_rate'])
        plt.title('death_rate distribution')
        plt.show()
        simulation.tumor_scatter(summary['x'], summary['y'], summary['birth_rate'])
        plt.title('birth_rate distribution')
        plt.show()


    