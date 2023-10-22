#import simulation
import numpy as np
import pickle
import sys
import utils
import pandas as pd
#classes and object methods
class Simulation():
    """Simulation Class
    
    Description: Represents a simulation of tumor growth.
    
    Attributes:
    - params: Dictionary of parameters for the simulation.
    - graph: Graphical representation of the tumor.
    - tumor: Instance of the Tumor class representing the tumor in the simulation.
    - t_traj: List to track the time trajectory.
    - N_traj: List to track the trajectory of the number of cells.
    - nbrs: Neighbors for the cells based on the given parameters.
    
    Methods:
    - __init__(self, params): Constructor method that initializes the simulation based on the provided parameters.
    - stopping_condition(self): Determines if the simulation should be stopped based on certain conditions, such as reaching a maximum number of cells, exceeding maximum iterations, etc.
    - run(self, rep=0): Executes the simulation. Keeps running the simulation while the stopping condition is not met. Provides periodic updates on tumor size and saves the state of the simulation at specified intervals.
    """

    def __init__(self, params):

        self.params = params
        self.graph = utils.set_graph(params)
        self.tumor = Tumor(self)
        self.t_traj = []
        self.N_traj = []
        self.nbrs = utils.set_nbrs(params['neighborhood'], params['dim'])

    def stopping_condition(self):
        """
        Description: Checks whether the simulation should stop based on certain conditions.
        
        Conditions:
        - Maximum number of cells reached.
        - Maximum number of iterations exceeded.
        - Tumor boundary hit.
        - No cells remaining in the tumor.
        
        Returns:
        - True if any of the stopping conditions are met, otherwise False.
        """
        nmax = self.params['n_cells']
        imax = self.params['max_iter']
        return self.tumor.N == nmax or self.tumor.iter > imax or self.tumor.hit_bound or self.tumor.N ==0
    
    def run(self,rep=0):
        """
        Description: Executes the tumor simulation for a given replicate.
        
        Inputs:
        - rep: Replicate number for the simulation.
        
        Behavior:
        - Continuously iterates the tumor growth until a stopping condition is met.
        - Prints the tumor size at regular intervals.
        - Saves tumor data based on specified save intervals.
        - Returns the final state of the tumor after the simulation.
        
        Returns:
        - A Tumor object representing the final state of the tumor after the simulation.
    """
        prev_N = 0 #previous population size 
        prev_t = 0 #previous timepoint
        self.params['rep'] = rep

        while not self.stopping_condition():
            self.tumor.iterate()
            if self.tumor.iter%40000==0:
                print(f'size = {self.tumor.N}')# at {int(self.tumor.t/365)} years {self.tumor.t%365} days')

            if self.tumor.N%self.params['cell_save_interval']==0 and self.tumor.N!=prev_N:
                #save_object(self, f'{self.params["exp_path"]}/rep={rep}_ncells={self.tumor.N}_time={self.tumor.t}.pkl')
                summary = utils.tumor_summary(self.tumor)
                if self.params['only_save_extremes']:
                    summary = utils.get_extremes(summary)
                    summary = pd.DataFrame([summary])
                summary.to_csv(f'{self.params["exp_path"]}/rep-{rep}-ncells-{self.tumor.N}-t-{self.tumor.t:.2f}.csv')
                prev_N = self.tumor.N #only save same cell size once 
            if int(self.tumor.t)%self.params['time_save_interval']==0 and int(self.tumor.t)!=prev_t:
                #save_object(self, f'{self.params["exp_path"]}/rep={rep}_ncells={self.tumor.N}_time={self.tumor.t:.2f}.pkl')
                summary = utils.tumor_summary(self.tumor)
                if self.params['only_save_extremes']:
                    summary = utils.get_extremes(summary)
                    summary = pd.DataFrame([summary])
                summary.to_csv(f'{self.params["exp_path"]}/rep-{rep}-ncells-{self.tumor.N}-t-{self.tumor.t:.2f}.csv')
                prev_t = int(self.tumor.t)
            
        summary = utils.tumor_summary(self.tumor)
        if self.params['only_save_extremes']:
            summary = utils.get_extremes(summary)
            summary = pd.DataFrame([summary])
        summary.to_csv(f'{self.params["exp_path"]}/rep-{rep}-ncells-{self.tumor.N}-t-{self.tumor.t:.2f}.csv')
        return self.tumor

    

       
class Cell():
    """Cell Class
    
    Description: Represents an individual cell, defined by its unique ID, position, and genotype.
    
    Attributes:
    - sim: Reference to the Simulation instance the cell belongs to.
    - ID: Unique identifier for the cell.
    - pos: Position of the cell, represented as a tuple.
    - gen: Genotype of the cell, represented by an instance of the Genotype class.
    
    Methods:
    - __init__(self, sim, ID, gen, pos): Constructor method that initializes the cell with the given simulation instance, ID, genotype, and position.
    - get_birth_rate(): Computes the birth rate of the cell.
    - get_death_rate(): Computes the death rate of the cell.
    - other rate functions based on different models and conditions.
    - default_mutate(): Default mutation behavior.
    - passen_mutate(): Mutation behavior for passengers.
    - driver_mutate(): Mutation behavior for drivers.
    - default_push(): Returns a guaranteed probability of pushing the cell.
    - prop_push(prop): Returns push based on the proportion of the cell's radius.
    - get_empty_nbrs(): Retrieves the empty neighboring positions for the cell.
    - get_all_nbrs(): Retrieves all neighboring positions for the cell.
    - __repr__(self): Returns a string representation of the cell, including its ID, genotype ID, and position.
    """
    def __init__(self,sim,ID, gen, pos) -> None:
        self.sim = sim 
        self.ID = ID
        self.pos = pos #nonneg DIM-tuple 
        self.gen = gen #Genotype 
        
        #self.death_rate = self.sim.params['init_death_rate'] if self.sim.params['fixed_death_rate'] else self.sim.params['init_death_rate']*np.power(self.sim.params['driver_dr_factor'], self.gen.n_drivers)
        BIRTH_FUNCTIONS = {'default': self.one_fixed_rate, 'fixed': self.one_fixed_rate, 'one_changing': self.one_changing_rate,
        'radial': self.radial_rate}

        DEATH_FUNCTIONS = {'default': self.one_changing_rate,
        'radial':self.radial_rate,'one_changing':self.one_changing_rate, 
        'radial_prop': self.radial_prop_rate, 'radial_bern': self.radial_bern_rate, 
        'nbr_based': self.nbr_based_rate, 'resistance_model': self.resistance_model_death, 'radial_nbr_hybrid':self.radial_nbr_hybrid,
        'radial_treatment': self.radial_treatment}

        MUTATE_FUNCTIONS = {'default': self.default_mutate, 'fixed_number': self.fixed_number_mutate, 
        'progression': self.progression_mutate}

        PUSH_FUNCTIONS = {'default': self.default_push, 'prop': self.prop_push}

         
        
        try:
            self.br_function = BIRTH_FUNCTIONS[self.sim.params['br_function']]
        except(KeyError):
            print(f'birth rate function not defined. Must be one of {[k for k in BIRTH_FUNCTIONS.keys()]}')
            print('exiting...')
            sys.exit()

        try:
            self.dr_function = DEATH_FUNCTIONS[self.sim.params['dr_function']]
        except(KeyError):
            print(f'death rate function not defined. Must be one of {[k for k in DEATH_FUNCTIONS.keys()]}')
            print('exiting...')
            sys.exit()

        try: 
            self.mutate = MUTATE_FUNCTIONS[self.sim.params['mutate_function']]
        except(KeyError):
            print(f'mutate function not defined. Must be one of {[k for k in MUTATE_FUNCTIONS.keys()]}')
            print('exiting...')
            sys.exit()
        
        try:
            self.get_push_rate = PUSH_FUNCTIONS[self.sim.params['push_function']]
        except(KeyError):
            print(f'birth rate function not defined. Must be one of {[k for k in BIRTH_FUNCTIONS.keys()]}')
            print('exiting...')
            sys.exit()

    def get_birth_rate(self):
        """wrapper function that returns the cell birth rate"""
        return max(0,min(1,self.br_function(**self.sim.params['br_params'], is_birth = True)))
    def get_death_rate(self):
        """wrapper function that returns the cell death rate"""
        return max(0, min(1,self.dr_function(**self.sim.params['dr_params'], is_birth = False)))

    
    #Growth rate functions: either birth or death 
    def one_fixed_rate(self, init_rate, is_birth = False):
        """default birth rate, just return the given birth rate"""
        return init_rate
   
    """Using params file, get cell death rate for a given configuration"""
    
        
    def one_changing_rate(self,init_rate, is_birth = False):
        """function to compute birth or death rate that is changeable due to selection
        Inputs:
            - init_rate: (float) initial birth or death rate
            - is_birth: (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate
        """
        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        return init_rate*np.power(1-s, self.gen.n_drivers)
        
    def radial_rate(self,radius, inner_rate, outer_rate, is_birth = False):
        """function to return a certain death rate based on radial position of cell
        Inputs:
            - radius (float): Distance from self.tumor.center up to when inner_rate is used 
            - inner_rate (float): birth/death rate used within radius
            - outer_rate (float) birth/death rate used at or beyond radius
            - is_birth (bool): (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate
        """
        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        a = np.array(self.pos)
        b = np.array(self.sim.tumor.center)
        if np.linalg.norm(a-b) < radius:
            #print('in')
            return inner_rate*np.power(1-s, self.gen.n_drivers)
        #print('out')
        return outer_rate*np.power(1-s, self.gen.n_drivers)

    def radial_prop_rate(self, prop, inner_rate, outer_rate, is_birth = False):
        """ inner death rate if within proportion of radius, otherwise inner
        Inputs:
            - prop (float): proportion of total estimated tumor radius up to when inner_rate is used 
            - inner_rate (float): birth/death rate used within radius
            - outer_rate (float) birth/death rate used at or beyond radius
            - is_birth (bool): (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate"""

        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        n_cells = self.sim.tumor.N 
        n_driv = self.gen.n_drivers
        r = utils.calc_radius(n_cells, self.sim.params['dim']) 
        a = np.array(self.pos)
        b = np.array(self.sim.tumor.center)
        #print(f'r = {r}')
        #print(f'dist = {np.linalg.norm(a-b)} and prop*r = {prop*r}')
        if np.linalg.norm(a-b) < prop*r:
            #print('inside radius')
            return inner_rate*np.power(1-s, n_driv)
        #print('outside radius')
        return outer_rate*np.power(1-s,n_driv)

    def radial_bern_rate(self, prop, inner_rate, outer_rate, is_birth = False):
        """calculate birth/death rate as a bernoulli r.v. with outer_rate with prob p = min(d(cell, center)/radius,1) and inner_rate with 
        prob 1-p

        Inputs:
            - prop (float): proportion of total estimated tumor radius up to when inner_rate is used 
            - inner_rate (float): birth/death rate used within radius
            - outer_rate (float) birth/death rate used at or beyond radius
            - is_birth (bool): (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate
            
        """
        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        n_cells = self.sim.tumor.N 
        n_driv = self.gen.n_drivers
        r = utils.calc_radius(n_cells, self.sim.params['dim']) 
        a = np.array(self.pos)
        b = np.array(self.sim.tumor.center)
        pcell = np.linalg.norm(a-b)/r
        probability = pcell/2/prop if pcell <=prop else (pcell+1-2*prop)/(2*(1-prop))
        if np.random.random() < probability:
            return outer_rate*np.power(1-s, n_driv)
        return inner_rate*np.power(1-s, n_driv)

    
    def nbr_based_rate(self, inner_rate, outer_rate, is_birth = False):
        """use inner_rate if cell's neighboorhood is full, use outer_rate otherwise  

        Inputs:
            - inner_rate (float): birth/death rate used within radius
            - outer_rate (float) birth/death rate used at or beyond radius
            - is_birth (bool): (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate
            
        """
        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        empty_nbrs = self.get_empty_nbrs()
        n_driv= self.gen.n_drivers
        if len(empty_nbrs) > 0:
            return outer_rate*np.power(1-s, n_driv)
        return inner_rate*np.power(1-s, n_driv)

    def resistance_model_death(self, **dr_params):
        """if cell has fewer mutations than needed for resistance, specified by the mutation
        function parameter 'muts_to_res', then death rate is radial_death rate. Otherwise
        return the inner death rate. Assumed that inner rate is lower, as this is a model 
        of acquired resistance to treatment"""

        if self.gen.n_drivers < self.sim.params['mutate_params']['muts_to_res']:
            return self.radial_rate(**dr_params)
        else: 
            return dr_params['inner_rate']

    def radial_nbr_hybrid(self, init_radius, inner_rate, outer_rate, is_birth = False):
        """assume as in radial_rate that the outer death rate is experienced past a fixed radius
        However, beyond that radius, only cells with empty neighbors experience the outer death rate. 
        This is necessary to simulate driver-dependent growth with the neighbor-based death rate (b < outer_rate) because the initial cell will not survive otherwise

        Inputs:
           - init_radius: (float) distance from self.tumor.center until which cells experience inner_rate (with selection)
           - inner_rate (float): birth/death rate used within init_radius, or as in nbr_based_rate afterwards
           - outer_rate (float) birth/death rate used at or beyond init_radius, or as in nbr_based_rate afterwards
           - is_birth (bool): (bool) True/False for birth/death
        Outputs:
            - (float) the computed rate
        
        """
        s = self.sim.params['driver_advantage']
        select_birth = self.sim.params['select_birth']
        if is_birth and select_birth:
            s = -s
        elif is_birth or select_birth:
            s = 0
        a = np.array(self.pos)
        b = np.array(self.sim.tumor.center)
        if np.linalg.norm(a-b) < init_radius: #if cell is in the inner region
            #print('in')
            return inner_rate*np.power(1-s, self.gen.n_drivers)
        else:
            return self.nbr_based_rate(inner_rate, outer_rate, is_birth)

    def radial_treatment(self, treat_size, radius, inner_rate, outer_rate, is_birth = False):
        """function to only introduce radial rate if treatment has started, which occurs after a detectable size
        Inputs:
            - treat_size: (float) Minimum size tumor must reach before the radial rate takes effect
            - radius, inner_rate, outer_rate, is_birth: args for radial_rate 
        Outputs:
            - (float) the computed rate
        
         """
        if self.sim.tumor.N >= treat_size and not self.sim.tumor.under_treatment:
            self.sim.tumor.under_treatment = True

        if not self.sim.tumor.under_treatment: 
            return self.one_changing_rate(init_rate = inner_rate)
        else:
            return self.radial_rate(radius = radius, inner_rate = inner_rate, outer_rate = outer_rate)
    
    #MUTATE FUNCTIONS
        
    def default_mutate(self):
        """mutate function depends only on tumor-level mutation order (infinite allele assumption)
        Samples poisson-distributed number of passenger mutations, driver mutations, and mutator mutations 
        If total is nonzero, creates a new genotype with new mutations added. Each mutation assigned a unique integer based on the order it appeared. Genotype's parent is current genotype. 
        """
        #print('starting mutation')
        #print('current cells are:')
        #print(tumor.cells)
        #sample number of mutations
        n_passen = np.random.poisson(self.gen.passen_rate)
        n_drivers = np.random.poisson(self.gen.driver_rate)
        n_mutators = np.random.poisson(self.gen.mutator_rate)
        n = self.sim.tumor.next_snp
        if n_passen + n_drivers + n_mutators >0:
            passen = None
            drivers = None
            mutators = None
            if n_passen>0:
                passen = np.append(self.gen.passen, np.arange(n,n+n_passen))
                n+=n_passen
            if n_drivers >0:
                #print(f"drivers {np.arange(n,n+n_drivers)} at size {self.sim.tumor.N}")
                drivers = np.append(self.gen.drivers, np.arange(n,n+n_drivers))
                n+=n_drivers
                self.sim.tumor.total_drivers+=n_drivers
            if n_mutators >0:
                #print(f"mutator at size {self.sim.tumor.N}")
                mutators = np.append(self.gen.mutators, np.arange(n,n+n_mutators))
                n+=n_mutators
            new_gen = Genotype(sim = self.sim,ID = self.sim.tumor.gens.len()+1, parent = self.gen, passen = passen, drivers = drivers, mutators = mutators)
            self.gen.children = np.append(self.gen.children, new_gen)
            self.gen.remove()
            self.gen = new_gen
            self.sim.tumor.add_genotype(new_gen)
        self.sim.tumor.next_snp = n
    
    def passen_mutate(self):
        """function to produce mutations in default mutate but only for passenger mutations
        outputs a list of ints containing the passenger mutations this round 
        """
        n = self.sim.tumor.next_snp
        n_new_passen  = np.random.poisson(self.gen.passen_rate)
        total_passen  = self.gen.passen
        if n_new_passen > 0:
            total_passen = np.append(total_passen, np.arange(n,n+n_new_passen))
        
        self.sim.tumor.next_snp += n_new_passen
        return total_passen

    def driver_mutate(self):
        """function to produce mutations in default mutate but only for driver mutations
        outputs a list of ints containing the driver mutations this round 
        """
        n = self.sim.tumor.next_snp
        n_new_driv  = np.random.poisson(self.gen.driver_rate)
        total_drivers  = self.gen.drivers
        if n_new_driv > 0:
            total_drivers = np.append(total_drivers, np.arange(n,n+n_new_driv))
        n += n_new_driv
        self.sim.tumor.next_snp += n_new_driv
        self.sim.tumor.total_drivers += n_new_driv
        return total_drivers

        

    def fixed_number_mutate(self, number = 1, by = 'random', value = 100):
        """allow a fixed number of mutations to appear in the whole population, and then forbid further mutations."""
        #needs work 
        new_drivers = []
        if by == 'size': #once tumor is 
            # a certain size, give out number of mutations to a group of cells all at the same time (meant to ensure some survive drift) until there are the desired number of mutations in the population
            if self.sim.tumor.N >= value and len(self.sim.tumor.tracked_variants) < number:
                snp = self.sim.tumor.next_snp
                new_drivers.append(snp)
                self.sim.tumor.next_snp += 1
                print(f'added driver to cell {self}')
                self.sim.tumor.tracked_variants.add(snp)
                print(f'added, set is now {self.sim.tumor.tracked_variants}')

        
        n_passen = np.random.poisson(self.gen.passen_rate)
        n = self.sim.tumor.next_snp
        if n_passen + len(new_drivers)>0:
            passen = np.append(self.gen.passen, np.arange(n,n+n_passen))
            n+=n_passen
            new_gen = Genotype(sim = self.sim,ID = self.sim.tumor.gens.len()+1, parent = self.gen, passen = passen, drivers = np.append(self.gen.drivers, new_drivers))
            self.gen.children = np.append(self.gen.children, new_gen)
            self.gen.remove()
            self.gen = new_gen
            self.sim.tumor.add_genotype(new_gen)
        self.sim.tumor.next_snp = n

        
        
    def progression_mutate(self, progression):
        """given a progression (directed acyclic graph of allowed mutations), sample cells according their progress in the progression"""
        raise(NotImplementedError)

    #PUSH FUNCTIONS

    def default_push(self):
        """return guaranteed probability of pushing"""
        return 0

    def prop_push(self, prop):
        """return push based on proportion of radius"""
        n_cells = self.sim.tumor.N 
        r = utils.calc_radius(n_cells, self.sim.params['dim']) 
        a = np.array(self.pos)
        b = np.array(self.sim.tumor.center)
        return int(np.linalg.norm(a-b) >= prop*r)
            

    def get_empty_nbrs(self):
        nbrs = (self.sim.nbrs + np.array(self.pos))
        tups = tuple([tuple(col) for col in nbrs.T])
        ids = self.sim.graph[tups]
        return nbrs[ids==0]

    def get_all_nbrs(self):
        nbrs = (self.sim.nbrs + np.array(self.pos)) #adding neighborhood to cell position to get position vectors 
        return nbrs

    

    def __repr__(self):
        return str(f'Cell# {self.ID} with gen {self.gen.ID} at {self.pos}')
    
        

class Genotype():
    """Genotype Class
    
    Description: Represents the genotype of a cell, storing mutations of a particular lineage. Mutations can be categorized as either passengers or drivers.
    
    Attributes:
    
    - ID: (int) The order in which the Genotype appeared during the simulation (a genotype id)
    - passen, drivers, mutators: (np array or None), a list of passenger/driver/mutator mutation IDs (ints) belonging to this genotype 
    - passen_rate, driver_rate, mutator_rate: (float) poisson-distributed rates of mutation accumulation for these types of mutations 
    - parent: (Genotype) the parent to this genotype
    - children: np array of genotype IDs of this genotype 
    - number: The number of cells existing with this genotype 
    
    Methods:
    - add: increment the number of cells with this genotype 
    - remove: decrement the number of cells with this genotype 
    - repr: String representation of this genotype 

    

    """
    def __init__(self, sim, ID=1, parent=None, passen=None, drivers=None, mutators = None) -> None:
        self.sim = sim
        self.ID = ID
        self.passen = np.array([], dtype = int) if passen is None else passen
        self.n_passen = self.passen.shape[0]
        self.drivers = np.array([], dtype = int) if drivers is None else drivers
        self.n_drivers = self.drivers.shape[0]
        self.mutators = np.array([], dtype = int) if mutators is None else mutators
        self.n_mutators = self.mutators.shape[0] #TODO: justify mutator 
        self.passen_rate = self.sim.params['passen_rate']*np.power(self.sim.params['mutator_factor'], self.n_mutators)
        self.driver_rate = self.sim.params['driver_rate']*np.power(self.sim.params['mutator_factor'], self.n_mutators)
        self.mutator_rate = self.sim.params['mutator_rate']*np.power(self.sim.params['mutator_factor'], self.n_mutators)
        
        self.parent = parent
        self.children = np.array([],dtype = int)
        self.number = 1
        
        #add genotype to genotype list
    def add(self):
        #self.sim.tumor.add_driver_count(self.n_drivers) #not necessary if death rate is not divided by dmax #comment out! 
        self.number+=1
    def remove(self):
        if self.number ==0:
            print("call to remove extinct genotype!!")
            raise(Exception)
        else:
            #if self.number==1:
                #self.sim.tumor.remove_driver_count(self.n_drivers) 
            self.number-=1
            if self.ID in self.sim.tumor.tracked_variants:
                self.sim.tumor.tracked_variants.pop(self.ID)
                print(f'popped, now is {self.sim.tumor.tracked_variants}')

    def __repr__(self):
        parID = 'none' if self.parent is None else self.parent.ID
        return str(f'Genotype {self.ID} from {parID} with {self.n_drivers} drivers')
    

class Tumor():
    """Tumor Class
    
    Description: Represents a tumor within the simulation, containing various cells and their genetic compositions.
    
    Attributes:
    - sim: Reference to the Simulation instance the tumor belongs to
    - graph: Graphical representation of the tumor
    - params: Parameters associated with the tumor
    - graph: the graph (lattice for now) containing genotype ids 
    - gens: Array of genotypes in the population 
    - cells: ListDict of cells in population 
    - center: coordinate of center 
    - N: population size (size of cells)
    - next_snp: next mutation when progression is None, None otherwise 
    - progression: matrix defining mutation tree to be sampled from if defined. Sampling depends on current genotype 
    - bmax: maximum population birth rate 
    - dmax: maximum population death rate
    - t: current time 
    - iter: current iteration 
    - hit_bound: boolean of whether tumor has reached boundary 
    - drivers: total driver mutations in tumor
    - mutators: total mutators in tumor 
    
    
    Methods:
    - __init__(self, sim): Constructor method that initializes the tumor based on the provided simulation instance.
    - run_growth_model
    - update_time 
    - iterate 
    growth models: 
        - bdg_spatialDeath: boundary driven growth with death dependent on available neighboring sites (quiescent core model)
        - bdg_nonSpatialDeath: boundary driven growth with death independent of neighboring sites (proliferative model) 
        - nonSpatialGrowth: model with spatially independent birth and death
    - simple_pushing: Simple form of cell pushing 
    - nbr_swap: Simple form of migration
    - add_cell 
    - remove_cell
    - add_genotype 
    - __repr__
    
    """
    def __init__(self,sim) -> None:
        self.sim = sim
        self.graph = sim.graph
        self.params = sim.params
        coord = int(self.params['boundary']/2)
        self.center = tuple(coord*np.ones(self.params['dim'],dtype = int))
        self.gens = ListDict()
        first_gen = Genotype(self.sim)
        self.gens.add_item(first_gen)
        self.cells = ListDict()
        first_cell = Cell(self.sim,ID = 1,gen = first_gen, pos = self.center)
        self.cells.add_item(first_cell)
        
        self.graph[self.center] = first_cell.ID
        self.N = 1
        self.hit_bound = False 
        self.tracked_variants = set()
        self.bmax = self.params['br_params']['init_rate']
        self.iter = 0
        self.t = 0
        self.next_snp = 0
        self.drivers = np.array([])
        self.mutators = np.array([])
        self.total_drivers = 0
        #new attribute
        self.under_treatment = False
       
        if self.params['model'] == 'default' or self.params['model'] == 'bdg_nonSpatialDeath':
            self.model = self.bdg_nonSpatialDeath
        elif self.params['model'] == 'bdg_spatialDeath':
            self.model = self.bdg_spatialDeath
        elif self.params['model'] == 'nonSpatial':
            self.model = self.nonSpatialGrowth
        else:
            print(f'model {self.params["model"]} not defined!\nexiting...')
            sys.exit()

    def run_growth_model(self):
        """wrapper to run growth model"""
        return self.model(**self.params['model_params'])
    def update_time(self):
        """update the time of the simulation after a birth/death event """
        self.t+=np.log(2)/(self.bmax*self.N)
        self.iter+=1

    def iterate(self):
        """do one iteration of birth, death, mutation, migration when applicable"""
        self.update_time()
        self.run_growth_model() 
        migrate_event = 0
        if self.N > 0:
            while migrate_event < np.random.poisson(self.params['n_migrations']):
                self.nbr_swap()
                migrate_event +=1
            

        if self.iter%400 == 0:
            self.sim.t_traj = np.append(self.sim.t_traj, self.t) #add iteration to time trajectory
            self.sim.N_traj = np.append(self.sim.N_traj, self.N) #add iteration to population trajectory 


    def bdg_spatialDeath(self):
        """Function to simulate birth, death, and mutation. This represents Waclaw 2015's 'quiescent core' model"""
        
        cell = self.cells.choose_random_item()
        #print(f'chose {cell}')
        gen = cell.gen
        nbrs = cell.get_empty_nbrs()
        
        if nbrs.shape[0] >0:
            nbr = nbrs[np.random.randint(0,nbrs.shape[0])]
            #print(f'chose nbr {nbr}')
            self.hit_bound = utils.check_bound(nbr, self.params['boundary'])
            br = cell.get_birth_rate()
            #print(br)
            if np.random.random() < br:
                new_cell = Cell(self.sim, ID = self.iter+1, gen = gen, pos = tuple(nbr))
                self.add_cell(new_cell)
            #print(f'cell pop is now: {self.cells}')
            #mutate daughter cell, sample to determine whether to kill parent cell. If no, mutate parent cell 
                new_cell.mutate(**self.params['mutate_params'])
            #if np.random.random() <cell.get_death_rate()/self.dmax: #comment out! 
            dr = cell.get_death_rate()
            #print(dr)
            if np.random.random() < dr:
                #kill cell
                #print(f'{cell} died')
                self.remove_cell(cell)
            else:
                cell.mutate(**self.params['mutate_params'])
        return

    def bdg_nonSpatialDeath(self):
        """growth where cells only give birth when there is empty space but die regardless. Note that the death rate function could kill cells dependent on spatial factors,
        but this function would be evaluated on all cells regardless of location or surrounding cells
         """

        cell = self.cells.choose_random_item()
        #print(f'chose {cell}')
        gen = cell.gen
        nbrs = cell.get_empty_nbrs()

        br = cell.get_birth_rate()
       
        #can_push = cell.get_push_rate()
        if nbrs.shape[0] >0:
            nbr = nbrs[np.random.randint(0,nbrs.shape[0])]
            #print(f'chose nbr {nbr}')
            self.hit_bound = utils.check_bound(nbr, self.params['boundary'])
            
            #print(br)
            if np.random.random() < br:
                new_cell = Cell(self.sim, ID = self.iter+1, gen = gen, pos = tuple(nbr))
                self.add_cell(new_cell)
            #print(f'cell pop is now: {self.cells}')
            #mutate daughter cell, sample to determine whether to kill parent cell. If no, mutate parent cell 
                new_cell.mutate(**self.params['mutate_params'])
            #if np.random.random() <cell.get_death_rate()/self.dmax: #comment out! 
       

        dr = cell.get_death_rate()
        if np.random.random() < dr:
            self.remove_cell(cell)
        return

    
    def nonSpatialGrowth(self,push_forward= False):
        """growth where cells are randomly pushed to make room for offspring at every timestep, not BDG"""
        cell = self.cells.choose_random_item()
        #print(f'chose {cell}')
        gen = cell.gen
        br = cell.get_birth_rate()
        
        if np.random.random() < br:
            if not push_forward:
                direction = utils.choose_direction(self.graph, cell.pos, self.sim.nbrs)
            else:
                #choose lattice direction closest to radial vector
                direction = utils.get_closest_vector(np.array(cell.pos) - np.array(self.center), self.sim.nbrs)
                
                
            birth_pos = cell.pos
            pushed = self.simple_pushing(cell = cell, direction = direction)
            if pushed:
                new_cell = Cell(self.sim, ID = self.iter+1, gen = gen, pos = birth_pos)
                self.add_cell(new_cell)
                new_cell.mutate(**self.params['mutate_params'])

        dr = cell.get_death_rate()

        if np.random.random() < dr:
            self.remove_cell(cell)
        return

    def simple_pushing(self, cell, direction):
        """simplest pushing scheme: shift cells along specified lattice direction, update their positions, starting position should now be empty, 
        direction is one of the possible directions specified by the model (6, 8, 26)
        """
        cur_id = cell.ID
        cur_pos = cell.pos
        traceback = []
        while cur_id > 0:
            traceback.append(cur_pos)
            cur_pos = tuple(np.array(cur_pos)+direction)
            
            try:
                cur_id = self.graph[cur_pos]
                
            except(IndexError):
                self.hit_bound = True
                return False
        empty_pos = cur_pos
        for pos in traceback[::-1]:
            #print(pos)
            cur_cell = self.cells.get_item(self.graph[pos])
            cur_cell.pos = empty_pos
            self.graph[empty_pos] = cur_cell.ID
            self.graph[pos] = 0
            empty_pos = pos
        return True

    
    def nbr_swap(self):
        """method to choose a random cell from the tumor, choose a random neighbor, swap locations
        input: none
        output: none 
        """
        cell = self.cells.choose_random_item()
        new_pos = cell.pos+self.sim.nbrs[np.random.randint(self.sim.nbrs.shape[0])]
        new_pos = tuple(new_pos)
        
        try:
            occupant_id = self.graph[new_pos]
        except(IndexError):
            self.hit_bound = True
            return

        if occupant_id>0:
                #swap
                other = self.cells.get_item(occupant_id)
                temp_pos = cell.pos
                cell.pos = other.pos
                other.pos = temp_pos
                self.graph[cell.pos] = cell.ID
                self.graph[other.pos] = other.ID
                pass
        else:
                self.graph[cell.pos] = 0 
                cell.pos = new_pos
                self.graph[cell.pos] = cell.ID
        return 
            
            #if unoccupied, move there, otherwise swap
            
    def add_cell(self, born_cell):
        """function to add cell to cell list, change relevant fields"""        
        self.graph[born_cell.pos] = born_cell.ID
        self.cells.add_item(born_cell)
        self.N+=1
        born_cell.gen.add()
    
    def remove_cell(self, doomed_cell):
        """function to remove cell from cell list change relevant fields"""
        self.graph[doomed_cell.pos]=0
        self.cells.remove_item(doomed_cell)
        self.N-=1
        doomed_cell.gen.remove()
        

    def add_genotype(self, gen):
        self.gens.add_item(gen)
        self.drivers = np.append(self.drivers, gen.drivers)
        self.mutators = np.append(self.mutators, gen.mutators)

    def __repr__(self):
        return str(self.graph)


class ListDict(object):
    """ListDict Class
    
    Description: Represents a data structure that combines the features of both lists and dictionaries, 
    designed to quickly add, remove, and randomly select items, while maintaining a mapping of items to 
    their positions in the list.
    
    Attributes:
    - item_to_position: Dictionary mapping items (by their IDs) to their positions in the list.
    - items: An array containing the items.
    
    Methods:
    - __init__(self): Constructor method that initializes an empty ListDict.
    - add_item(self, item): Adds an item to the ListDict.
    - remove_item(self, item): Removes the specified item from the ListDict.
    - choose_random_item(self): Returns a randomly chosen item from the ListDict.
    - get_item(self, index): Retrieves an item based on its ID.
    - len(self): Returns the number of items present in the ListDict.
    - __repr__(self): Returns a string representation of the ListDict."""

    def __init__(self):
        self.item_to_position = {}
        self.items = np.array([])
    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items = np.append(self.items, item)
        self.item_to_position[item.ID] = self.items.shape[0]-1
    def remove_item(self, item):
        position = self.item_to_position.pop(item.ID)
        last_item = self.items[-1]
        self.items = self.items[:-1]
        if position != self.items.shape[0]:
            self.items[position] = last_item
            self.item_to_position[last_item.ID] = position

    def choose_random_item(self):
        return np.random.choice(self.items)
    def get_item(self, index):
        return self.items[self.item_to_position[index]]
    def len(self):
        return self.items.shape[0]
    def __repr__(self):
        return f'{str([item.__repr__() for item in self.items])}'








   
    
   