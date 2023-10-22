#code for simulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import ast
from scipy.stats import binned_statistic
import sys


#tumor scatterplot
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def tumor_scatter(x,y,c, cmap_name = None, dim = 80, show = False):
    """return 2d tumor scatterplot, accepts the x and y coords and contour value"""
    unqs, idx = np.unique(c,return_inverse = True)
    cmap = get_cmap(len(unqs), name = cmap_name)
    plt.xlim((-dim,dim))
    plt.ylim((-dim,dim))
    plt.axis('square')
    plt.scatter(x=x,y=y, c = c, cmap = cmap,marker = 's',s = 1)
    if show:
        plt.show() 
def tumor_summary(tumor):
    """turn tumor into dataframe with all the information: 
        ncells x ? matrix where columns are 'cell_ID' 'x','y' 'r' 'angle' 'genotype' 'n_drivers' 'drivers' 
        'death rate' 'cell_hge' 

        Version to work with tumor object as defined in classes.py as of the last commit on 11.18.22 
        """
    #turn list of cells into columns of information 
    
    decay = 33 #decay rate based on 30min half life of ctDNA from Reiter et al. 
    mat = tumor.graph
    cell_ID = mat[mat > 0]
    x, y = np.indices(mat.shape)
    x = x[mat >0] - tumor.center[0]
    y = y[mat>0] - tumor.center[1]
    r = np.sqrt(x**2 + y**2)
    angle = (360/2/np.pi)*np.arctan2(y,x)
    
    genotype = np.array([tumor.cells.get_item(id).gen.ID for id in cell_ID], dtype = int)
    n_drivers = np.array([tumor.cells.get_item(id).gen.n_drivers for id in cell_ID], dtype = int)
    drivers = [tuple(tumor.cells.get_item(id).gen.drivers) for id in cell_ID]
    passengers = [tuple(tumor.cells.get_item(id).gen.passen) for id in cell_ID]
    parents = [tumor.cells.get_item(id).gen.parent.ID if tumor.cells.get_item(id).gen.parent is not None else np.nan for id in cell_ID]
    
    
    #account for spatial death - CHANGE THIS IN Cell class !!!!!! spatial effect on rate should be defined by growth model
    if tumor.params['model'] == 'bdg_spatialDeath':
        death_rate = [0 if (tumor.cells.get_item(id).get_empty_nbrs()).shape[0]==0 else tumor.cells.get_item(id).get_death_rate() for id in cell_ID]
    else:
        death_rate = [tumor.cells.get_item(id).get_death_rate() for id in cell_ID]
    if tumor.params['model'].split('_')[0] == 'bdg':
        birth_rate = [0 if (tumor.cells.get_item(id).get_empty_nbrs()).shape[0]==0 else tumor.cells.get_item(id).get_birth_rate() for id in cell_ID]
    else:
        birth_rate = [tumor.cells.get_item(id).get_birth_rate() for id in cell_ID]

    df = pd.DataFrame({'cell_ID' : cell_ID, 'x':x,'y':y,'r':r,'angle':angle, 
    'genotype':genotype,'n_drivers': n_drivers, 'drivers':drivers,'passengers': passengers, 'parent': parents, 'death_rate': death_rate, 'birth_rate': birth_rate})

    #df['cell_hge'] = df['death_rate']/(df['birth_rate'] - df['death_rate'] + decay)
    df['rep'] = tumor.params['rep']
    df['t'] = tumor.t
    return df

def load_tumor_summary(path):
    """loads a csv with the output from tumor_summary. Note: this only includes genotype number information as of now"""
    output = pd.read_csv(path)
    return output 
def get_extremes(timedata):
    """return the data on the min and max difference clone fractions in a tumor summary file
    elements are returned as a dict with the following keys:
    diffmax: maximum clone fraction difference 
    tmax: time at which this occurs 
    nmax: populatin size at which this occcurs
    diffmin: 
    tmin:
    nmin: 
    tissfreq: actual tissue clone fraction of the clone 
    """
    n_timepoints = timedata['t'].unique().shape[0]
    n_reps = timedata['rep'].unique().shape[0]
    if n_timepoints == 1 and n_reps == 1:
        grouped = timedata.groupby(['genotype'])
        tissue = grouped['genotype'].count()/timedata['genotype'].size
        blood = grouped['death_rate'].sum()/timedata['death_rate'].sum()
        
        diff = blood - tissue
        diffmax = diff.max()
        diffmin = diff.min()
        tmax = timedata['t'].values[0]
        tmin = tmax
        nmax = timedata.shape[0]
        nmin = nmax
        
        tiss_freq_max = tissue[diff==diffmax].values[0]
        tiss_freq_min = tissue[diff==diffmin].values[0]
        
    else:
        grouped_rep_t_gen = timedata.groupby(['rep','t','genotype'])

        tissue = grouped_rep_t_gen['genotype'].count()/timedata.groupby(['t','rep'])['genotype'].size()
        blood = grouped_rep_t_gen['death_rate'].sum()/timedata.groupby(['t','rep'])['death_rate'].sum()
        comparison = tissue.compare(blood)
        comparison.columns = ['tissue','blood']
        comparison['diff'] = (comparison['blood'] - comparison['tissue'])
        diffmax = comparison['diff'].max()
        diffmin = comparison['diff'].min()
        reset = comparison.reset_index()
        tmax = reset[reset['diff']==diffmax]['t'].iloc[0]
        tmin = reset[reset['diff']==diffmin]['t'].iloc[0]
        nmax = timedata[timedata['t']==tmax].shape[0]
        nmin = timedata[timedata['t']==tmin].shape[0]
        tiss_freq_max = reset[reset['diff']==diffmax]['tissue'].iloc[0]
        tiss_freq_min = reset[reset['diff']==diffmin]['tissue'].iloc[0]
    out = {"diffmax": diffmax, "tmax":tmax, "nmax":nmax, 
    "diffmin": diffmin, "tmin": tmin, "nmin": nmin, "tiss_freq_max":tiss_freq_max, 
    "tiss_freq_min":tiss_freq_min}
    return out
    
def plot_tumor(tumor,drivers=False,trim = 0):
    graph = tumor.graph.copy()
    pos = np.array([np.array(item.pos) for item in tumor.cells.items])
    pos = tuple([tuple(row) for row in pos.T])
    if drivers:
        gens = [item.gen.n_drivers for item in tumor.cells.items]
    else:
        gens = [item.gen.ID for item in tumor.cells.items]
    graph[pos] = gens
    if len(tumor.graph.shape) ==2:
        graph = graph[trim:-(1+trim),trim:-(1+trim)]
        graph[graph==0]=-np.max(graph.flatten())
        sns.heatmap(graph,cbar= False,square = True)
        #plt.show()
    else:
        raise(NotImplementedError)
def plot_slice(tumor, ax = 0,trim=0):
    if len(tumor.graph.shape)==2:
        plot_tumor(tumor,trim)
    else:
        graph = tumor.graph.copy()
        pos = np.array([np.array(item.pos) for item in tumor.cells.items])
        pos = tuple([tuple(row) for row in pos.T])
        gens = [item.gen.ID for item in tumor.cells.items]
        graph[pos] = gens
        graph[graph==0] = -np.max(graph.flatten())
        if ax=='x' or ax==0:
            sns.heatmap(graph[tumor.center[0]])
        elif ax=='y' or ax==1:
            sns.heatmap(graph[:,tumor.center[0],:])
        elif ax=='z' or ax==2:
            sns.heatmap(graph[:,:,tumor.center[0]])
        else:
            print('invalid axis')
        plt.title(f'Slice along {ax} axis')
        plt.show()
        return graph

"""Given a tumor genotype ListDict, construct tree
    This tree includes possibly extinct genotypes and represents the true evolutionary history 
    of the simulation 
"""
def get_tumor_phylogeny(gens):
    
    root = gens.get_item(0)
    return make_newick(root)
    
"""Given a list of genotypes and root, return newick tree"""
def make_newick(root):
    #base case
    if len(root.children)==0:
        return f'{root.ID}'
    else:
        return f'{tuple([make_newick(child) for child in root.children])}{root.ID}'
"""given a list of genotypes, return genotypes sorted by pop and drivers"""
def get_genotype_dist(gens):
    genlist = gens.items
    df = pd.DataFrame([[g.ID,g.number,g.n_drivers] for g in genlist])
    df.columns = ['clone','number','n_drivers']
    bypop = df.sort_values(by = 'number')
    bydriv = df.sort_values(by='n_drivers')
    return bypop,bydriv
"""plot top <topn> genotypes by population and numebr of driver mutations, return correlation between groups """
def genotype_plot(bypop, bydriv, topn = 10):
    bypop.iloc[-topn:].plot.bar(x = 'clone', y='number')
   
    plt.show()
    bypop.iloc[-topn:].plot.bar(x='clone', y = 'n_drivers')
    plt.show()
    plt.plot(bypop.iloc[-topn:]['clone'],(bypop['number'].sum()/bypop.iloc[-topn:]['number'])**2,'.')
    plt.show()
def plot_genotype_dist(**kwargs):
    bypop, bydriv = get_genotype_dist(gens)
    genotype_plot(bypop = bypop, bydriv = bydriv, topn=10)
def plot_growth(tumor,**kwargs):
    ax = plt.scatter(tumor.sim.t_traj, tumor.sim.N_traj,**kwargs)
    plt.xlabel('time (days)')
    plt.ylabel('size (cells)')
    plt.xlim(left = 0)
    #plt.show()
    return ax
def plot_drivers(tumor, by_fitness = False, vmin = None, vmax = None):
    if not by_fitness:
        graph = get_driver_graph(tumor)
        ids, counts = np.unique(graph.flatten(), return_counts = True)
        ids = ids[2:]
        counts = counts[2:]
        img = graph.copy()
        scale = 100
        for i, clone in enumerate(ids):
            img[img==clone] = scale*(i+1)+200
            img[img > scale*len(ids)] = 0
            img[img<0] = -scale*len(ids)
        #cmap = matplotlib.colors.ListedColormap ( np.random.rand ( 256,3)),
        sns.heatmap(img,cbar = False)
    else:
        graph = get_fitness_graph(tumor)
        ids, counts = np.unique(graph.flatten(), return_counts = True)
        ids = ids[2:]
        counts = counts[2:]
        
        ax = sns.heatmap(graph,vmin, vmax)
        
    #plt.show()
    #df = pd.DataFrame([ids, counts]).T
    #df.plot.bar(x = 0,y=1)
    #plt.xlabel('driver mutation ID')
    #plt.ylabel('count')
    #plt.show()
    return ids, counts, ax
"""given an axis object, plot a circle of radius r from the center. Return axis with circle drawn"""
def plot_circle(ax, center, r):
    circ = plt.Circle(center, r, color='g', fill = False)
    ax.add_patch(circ)
    return ax
"""function to save entire tumor object and all related objects to folder"""
def save_tumor(tumor):
    #TODO
    pass
"""function to read tumor object from folder so that simulation can be started"""
def load_tumor(tumor):
    #TODO
    pass
"""Take bulk sample at specified location
inputs:
    tumor: tumor object to be sampled
    pos: position, 2d or 3d tuple defining center of cube to be sampled. must match dimensions of tumor
    length: side length of cube to be sampled 
    depth: read depth
    cutoff: minimal detectable VAF 
outputs:
    vector of VAFs 
"""
def bulk_sample(tumor, pos: tuple, length = None, depth: int = 0):
    #TODO 
    try:
        assert(len(tumor.graph.shape)==len(pos))
    except(AssertionError):
        print(f'expected dimension {tumor.graph.shape} but got {len(pos)}')
        print('exiting...')
        sys.exit()
   
    genmat = get_gen_graph(tumor)
    if length is not None: 
        try:
            l = int((length-1)/2)
            assert(length%2==1)
        except(AssertionError):
            print('length must be odd')
            print('exiting...')
            sys.exit()
        try:
            sample = genmat[pos[0]-l:pos[0]+l+1,pos[1]-l:pos[1]+l+1]
            
            if len(pos)==3:
                sample = sample[:,:,pos[2]-l:pos[2]+l+1]
            assert(sample.shape == (length, length))
            
        except(IndexError):
            print('sample out of bounds')
            print('exiting...')
            sys.exit()
        except(AssertionError):
            print(f'dimensions of sample are wrong, expected ___ but got {sample.shape}')
            print('exiting...')
    else:
        sample = genmat[genmat>0] #sample whole tumor
    
    
    #get frequencies of genotypes
    #gens, counts = np.unique(sample.flatten(),return_counts = True)

   # for g in sample.flatten():
   #     if g==1:
   #         tumor.gens.get_item(g).neut = np.array([-1])
   #     if g==2: 
   #         print(tumor.gens.get_item(g).neut)
   #     all_muts = np.append(all_muts, tumor.gens.get_item(g).neut)
   #     all_muts = np.append(all_muts, tumor.gens.get_item(g).drivers)

    all_muts = map(lambda genid: get_muts(genid, tumor), sample.flatten())

    all_muts = np.hstack(np.array(list(all_muts),dtype=object)).flatten()
    muts, counts = np.unique(all_muts,return_counts = True)
    vafs = counts/len(sample) if counts.sum() > 0 else counts
    #turn gens into VAFs
    #get all mutations in set, 
    
    if depth <1:
        m, f = muts, vafs
    else:
        #simulate sequence reads
        m, f = sequence(muts, vafs, depth)
    return m, f
def genotype_sample(tumor, pos, length):
    try:
        assert(len(tumor.graph.shape)==len(pos))
    except(AssertionError):
        print(f'expected dimension {tumor.graph.shape} but got {len(pos)}')
        print('exiting...')
        sys.exit()
   
    genmat = get_gen_graph(tumor)
    if length is not None: 
        try:
            l = int((length-1)/2)
            assert(length%2==1)
        except(AssertionError):
            print('length must be odd')
            print('exiting...')
            sys.exit()
        try:
            sample = genmat[pos[0]-l:pos[0]+l+1,pos[1]-l:pos[1]+l+1]
            
            if len(pos)==3:
                sample = sample[:,:,pos[2]-l:pos[2]+l+1]
            assert(sample.shape == (length, length))
            
        except(IndexError):
            print('sample out of bounds')
            print('exiting...')
            sys.exit()
        except(AssertionError):
            print(f'dimensions of sample are wrong, expected ___ but got {sample.shape}')
            print('exiting...')
    else:
        sample = genmat[genmat>0]
    print(sample)
    gens, counts = np.unique(sample.flatten(),return_counts = True)
    return gens, counts/counts.sum()
def get_muts(genID,tumor):
    if genID==0:
        return np.array([])
    gen = tumor.gens.get_item(genID)
    #if genID==1:
       #neut = np.array([-1])
    return np.append(gen.neut, gen.drivers)
"""Given a list of mutations and their frequencies, simulate bulk sequencing in the manner of
Chkhaidze et al: 
Get coverage with distribution pois(depth) for each mutation
Get get frequency as binom(vaf, coverage)
return only those greater than cutoff
"""
def sequence(muts, vafs,depth):
    coverage = np.random.poisson(depth, muts.shape[0])
    sampled_vafs = np.array([np.random.binomial(cov, f) for f, cov in zip(vafs, coverage)])
    sampled_vafs = sampled_vafs/coverage
    return muts, sampled_vafs
"""sample entire outer shell starting from a proportion of the tumor radius
inputs: tumor (Tumor)
        prop  (float)
outputs: muts (np array)
        vafs (np array)
"""
def shell_sample(tumor, prop):
    r = calc_radius(tumor.N, dim = len(tumor.graph.shape))
    shift = r*prop
    dist = get_dist_matrix(tumor.graph.shape, tumor.center)
    g = tumor.graph
    cells = g[dist > shift][g > 0]
    gens, counts = np.unique(cells.flatten(),return_counts = True)
    return gens, counts
def get_dist_matrix(shape, center):
    n = shape[0]
    if len(shape)==2:
        grid = np.ogrid[0:n,0:n]
        dist = np.sqrt((grid[0]-center[0])**2 + (grid[1]-center[1])**2)
    else:
        grid = np.ogrid[0:n,0:n,0:n]
        dist = np.sqrt((grid[0]-center[0])**2 + (grid[1]-center[1])**2 + (grid[2]-center[2])**2)
    return dist

"""return lattice populated by genotypes"""
def get_gen_graph(tumor):
    graph = tumor.graph.copy()
    pos = np.array([np.array(item.pos) for item in tumor.cells.items])
    pos = tuple([tuple(row) for row in pos.T])
    gens = [item.gen.ID for item in tumor.cells.items]
    graph[pos] = gens
    return graph
"""return lattice populated by unique driver clones"""
def get_driver_graph(tumor):
    graph = tumor.graph.copy()
    graph[graph==0] = -1
    pos = np.array([np.array(item.pos) for item in tumor.cells.items])
    pos = tuple([tuple(row) for row in pos.T])
    drivs = [item.gen.drivers[-1] if item.gen.n_drivers > 0 else 0 for item in tumor.cells.items]
    graph[pos] = drivs
    return graph
"""get lattice populated by number of driver mutations"""
def get_fitness_graph(tumor):
    graph = tumor.graph.copy()
    graph[graph==0] = -1
    pos = np.array([np.array(item.pos) for item in tumor.cells.items])
    pos = tuple([tuple(row) for row in pos.T])
    fitns = [item.gen.n_drivers for item in tumor.cells.items]
    graph[pos] = fitns
    return graph
#####Radial Stuff######
"""take 2d lattice populated by some feature (genotype ID, cell ID, # drivers etc), return pd dataframe
with x and y coords and radius from center"""
def radial_cluster(mat,center):
    #turn nxn into n^2 x 3 
    mat  = mat.copy() 
    x, y = np.indices(mat.shape)
    x = x.ravel(order='F')
    y = y.ravel(order='F')
    values = mat.ravel(order='F')
    feats =  np.array([x, y, values]).T
    feats = pd.DataFrame(feats, columns = ['x','y','val'])
    feats = feats[feats.val != -1]
    feats['x']-= center[0]
    feats['y']-=center[1]
    feats['r'] = np.sqrt(feats['x']**2 + feats['y']**2)
    feats['theta'] = np.arctan(feats['x']/feats['y'])
    return feats

def calc_radius(n_cells, dim):
    if dim==2:
        return np.sqrt(n_cells/np.pi)
    else:
        return np.cbrt(3*n_cells/4/np.pi)
#####TREE ANALYSIS#####
class Node():
    def __init__(self, ID, parent, children = None, number = 1):
        self.ID = ID
        self.parent = parent
        if type(children) is not list:
            self.children = []
        else:
            self.children = children
        self.number = number 
    def __repr__(self):
        return f'ID: {self.ID} Num: {self.number} Par: {self.parent.ID if self.parent is not None else "None"} Ch: {[child.ID for child in self.children]}'
    def copy(self):
        return Node(self.ID, self.parent, self.children, self.number)

def J_shannon(node):
    #not all recursive: one part is the subtree sums of all subtrees, one part is the recursive entropy calculation 
    s_istar = subtree_sum_no_root(node)
    w_i = 0
    b = len(node.children)
    if b>1:
        for ch in node.children:
            s_j = ch.number + subtree_sum_no_root(ch)
            p_ij = s_j/s_istar
            if s_j/s_istar>0:
                w_i += -p_ij*np.log(p_ij)/np.log(b)
    return (s_istar**2/(s_istar+node.number))*w_i
"""return total number in subtree excluding the root"""
def subtree_sum_no_root(node):
    if len(node.children)==0:
        return 0
    else:
        return np.sum([ch.number + subtree_sum_no_root(ch) for ch in node.children])

def traverse_sum(node, f):
    return f(node) + np.sum([traverse_sum(ch,f) for ch in node.children])
def J_index(node):
    if subtree_sum_no_root(node)+node.number <1:
        return 0
    return traverse_sum(node, J_shannon)/traverse_sum(node, subtree_sum_no_root)
"""copy data from TreeNode object (skbio) to Node so that it has the attribute 'number' to allow for J_index"""

def calc_radius(n_cells, dim):
    """return 2d or 3d spherical radius of tumor"""
    try:
        assert(dim==2 or dim==3)
    except(AssertionError):
        print(f"dim must equal 2 or 3 not {dim}")
    if dim==2:
        return np.sqrt(n_cells/np.pi)
    else:
        return np.cbrt(3*n_cells/4/np.pi)

def check_bound(arr, boundary):
        return (arr==boundary-1).any() or (arr==0).any()
  
def set_graph(params):
    l = int(params['boundary'])
    d = int(params['dim'])
    t = tuple(l*np.ones(d,dtype = int))
    return np.zeros(t,dtype=int)

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
def load_object(filename):
    with open(filename,'rb') as outp:
        obj = pickle.load(outp)
        return obj

def choose_direction(graph, start_pos, directions):
    """given a list of possible directions, choose randomly from non-infinite distances to 0 element"""
    distances = []
    for v in directions:
        cur_pos = np.array(start_pos)
        cur_id = graph[start_pos]
        dist = 0
        while cur_id > 0:
            cur_pos += v
            dist+=1
            try:
                cur_id = graph[tuple(cur_pos)]
            except(IndexError):
                dist = np.infty
                break
        distances = np.append(distances, dist)
    
    options = directions[distances < np.infty]
    direction = options[np.random.randint(len(options))]
    
    return direction

def set_nbrs(neighborhood, dim):
    """configure the neighboring cells on the lattice"""
    nbrs = []
    if neighborhood == 'moore':
            for i in [0,-1,1]:
                for j in [0,-1,1]:
                    if dim ==2:
                        nbrs.append([i,j])
                    else:
                        for k in [0,-1,1]:
                            nbrs.append([i,j,k])
            nbrs = np.array(nbrs)[1:]
    else:   
        if dim==3:
            nbrs = np.array([[1,0,0], [-1,0,0], [0,1,0],[0,-1,0],[0,0,1],[0,0,-1]])
        else:
            nbrs = np.array([[1,0],[-1,0],[0,1],[0,-1]])
    return nbrs




def get_closest_vector(v, vectors):
    """returns a vector from vectors that is the closest to v in angle
    inputs: v: 1d numpy array as row vector
            vectors: 2d numpy array where rows are the vectors of interest
    """
    if (v == 0).all():
        return vectors[np.random.randint(low = 0, high = len(vectors))]
    u = v/np.linalg.norm(v)
    return vectors[np.argmax(vectors@u.T)]
if __name__=="__main__":
    pass
