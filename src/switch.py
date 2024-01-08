import numpy as np
import nkpack as nk
from time import time,sleep
import random
from itertools import combinations as comb
from itertools import combinations

import statistics
from scipy.stats import norm
import math
from matplotlib import pyplot as plt
import pandas as pd
import cvxpy as cp
import scipy.stats as stats

dormantStakeholder = 65535 # Stakeholders decimal strategy when they quite the process [1 1 1 1 1 1 1 1]


class Organization:
    def __init__(self, p, n, k, c, s, t, rho, num, env, org_pos, correlations, s_e_bs, s_e_ps, efforts, name):
        # Organization properties
        self.p = p  # Population = 2 (Organization and Environment)
        self.n = n  # Number of tasks per agent
        self.k = k  # Number of internally coupled bits
        self.c = c  # Number of externally coupled bits
        self.s = s  # Number of externally coupled agents
        self.t = t  # Lifespan of the organization
        self.rho = rho  # Correlation coefficient among Organization and Environment
        self.Num = num  # Number of stakeholders ( 'num' used twice intentionally)
        self.num = num  # Number of stakeholders ( 'num' used twice intentionally)
        self.name = name  # Name of the organization

        # Classes and references
        self.nature = None  # Reference to the Nature class
        self.shocked_nature = None

        # Lists to keep stakeholders and outcomes of open_strategy
        self.stakes = []  # Reference to the stakeholders
        self.proposals = []  # Stakeholders' proposals
        self.perf_of_proposals = []  # Performance of proposals
        self.percived_performance = []  # Perceived performance (Errored)

        # Locations of the environment and the organization
        self.env = env # The location of the environment agent on its landscape
        self.org_pos = org_pos # The location of the organization on its landscape
        self.perceived_env_stk = None # The perceived location of the environment by stakeholders
        self.perceived_env_org = None # The perceived location of the environment by environment
        self.stake_es_position = None # The consensus of the environment by stakeholders

        # Features of stakeholders
        self.correlations = correlations # Correlation values (Stakeholders objectives related to the organization)
        self.s_e_bs = s_e_bs # List of stakeholders' error about the location of the environment (Error about the environment)
        self.s_e_ps = s_e_ps # List of stakeholders' error about the performance of given strategy (additive error)
        self.efforts = efforts # List of stakeholders' effort in their local search around the neighborhood
        self.all_proposals = [] # All the proposals in time step t
        self.all_environments = [] # the trejectory of the environment movement
        self.current_state = np.array(np.hstack((self.org_pos, self.env))) # Pair of the strategy and environment (For measuring performance)
        self.strategy_line = np.zeros((1, self.t)) # Trajectory of the strategy and the environment over time
        self.strategy_line[0, 0] = nk.bin2dec(self.current_state) # The trajectory starts with the random current strategy and environment
        self.time_shock = None # The time in which landscapes alter (Replaced with correlated ones)
        self.annealing = None # simulated annealing
        self.removed = []
        self.T = None # number of time steps
        self.voting_activate = None # If stakeholders participate in voting without proposing

        # Switch to open strategy
        self.timeSwitch = None
        self.movements = 0
        self.net = None

        # Clustering of correlation values

        self.black_list = []
        self.number_of_participants = [num]
        self.current_participants = num
        self.not_improved = 0
        self.switch_per = False
        self.switch_num = False
        self.max_per = 0
    

    def switching(self,t):
            
            #performance treshhold
            current_strategy_dec = self.strategy_line[0, t]
            current_strategy_bin = self.dec_to_bin(current_strategy_dec)
            
            previous_strategy_dec = self.strategy_line[0, t-1]
            previous_strategy_bin = self.dec_to_bin(previous_strategy_dec)
    
            current_strategy_performance  = self.nature.phi(None, current_strategy_bin)[0]
            previous_strategy_performance = self.nature.phi(None, previous_strategy_bin)[0]
            
            
            if current_strategy_performance - previous_strategy_performance <=0:
                self.not_improved += 1
            if self.not_improved >= 5:
                self.switch_per = True
            
            # Number of stakeholders treshhold
            if self.current_participants < .7*self.num:
                self.switch_num = False
            else:
                self.switch_num = True
    
    
    def switching2(self,t):
            #performance treshholdtra
            line[0, t]
            current_strategy_bin = self.dec_to_bin(current_strategy_dec)
            
            previous_strategy_dec = self.strategy_line[0, t-1]
            previous_strategy_bin = self.dec_to_bin(previous_strategy_dec)
    
            current_strategy_performance  = self.nature.phi(None, current_strategy_bin)[0]
            previous_strategy_performance = self.nature.phi(None, previous_strategy_bin)[0]
            if current_strategy_performance - previous_strategy_performance <=0:
                self.not_improved += 1
            if self.not_improved >= 3:
                self.switch = True
            # Number of stakeholders treshhold
            if self.current_participants < .5*self.num:
                self.switch_num = False
            else:
                self.switch_num = True     
       
                
                 
        
    def define_tasks(self):
        """ Create the Nature with given parameters (Nature has all the landscapes for all agents)"""
        nature = Nature(p=self.p,n=self.n,k=self.k,c=self.c,s=self.s,t=self.t,rho=self.rho,correlations=self.correlations,num=self.num)
        self.nature = nature
        self.nature.set_interactions()
        self.nature.set_landscapes() # !!! processing heavy !!!

    def hire_stakeholders(self):
        """ Create stakeholders and stores them"""
        for i in range(self.Num):
            self.stakes.append(Stake(employer=self, myid=i, correlation=self.correlations[i], s_e_b = self.s_e_bs[i], s_e_p = self.s_e_ps[i], effort=self.efforts[i], frustration = 0 ,satis = 1))
        self.nature.stakes = self.stakes

    def idea_generation(self,time,ham):
        """ Each stakeholder generates one proposal exploring given hamming_distance (2) """
        self.all_proposals = [] # A list to keep stakeholders' proposals
        self.all_environments = [] # A list to keep perception of stakeholders of the environment (Due to their error)
        
        for stake in self.stakes:
           stake.perform_climb(time,ham) # Every stakeholder explores the neighborhood and propose a strategy
           stake.stake_report(time) # Every stakeholder reports it's proposal
        self.environment_estimation() # Estimation of the environment by use of stakeholders perceptions (Knowldege of crowds)
             
    def satisfaction(self):
        """
        Measures the average performance of a selected strategy 
        for all stakeholders.
        """
        
        all_performances = []
        
        # Extract strategies over time.
        strategies = [self.strategy_line[0, t] for t in range(self.t)]
    
        # Calculate performance for each strategy.
        for t, strategy in enumerate(strategies):
            strategy_bit = self.dec_to_bin(strategy)
            
            # Calculate stakeholder performance based on the current and post-shock scenarios.
            if t < self.time_shock:
                stake_performances = [self.nature.phi(stake.myid, strategy_bit)[0] for stake in self.stakes]
            else:
                stake_performances = [self.shocked_nature.phi_shock(stake.myid, strategy_bit)[0] for stake in self.stakes]
            
            # Calculate mean performance for the strategy. If no stakeholders, default to 0.
            mean_performance = sum(stake_performances) / len(stake_performances)
            all_performances.append(mean_performance)
        return all_performances

    def track_the_number(self):
        participants = 0
        for proposal in self.all_proposals:
            if proposal.all() != np.array([1] * self.n * self.p).all():
                participants = participants + 1
        self.current_participants = participants
        self.number_of_participants.append(participants)
                
                
        
        
        
    def shock(self, correlation):
        """
        Change landscapes with a correlated landscape with noise
        """
        
        # Create a new instance of shocked nature.
        self.shocked_nature = shocked_nature(
            p=self.p, n=self.n, k=self.k, c=self.c, s=self.s, 
            t=self.t, rho=self.rho, correlations=self.correlations, 
            num=self.num, inmat=self.nature.inmat
        )
        
        # Extract the original landscape.
        original_landscape = self.nature.landscape
        
        # Create a random matrix matching the shape of the original landscape.
        matrix_shape = np.shape(original_landscape)
        random_matrix = np.random.uniform(0, 1, matrix_shape)
        
        # Apply shock to the original landscape.
        self.shocked_nature.landscape_shock = original_landscape * correlation + (1 - correlation) * random_matrix
        
        # Apply shock to all landscapes.
        for landscape in self.nature.all_landscapes:
            shocked_landscape = landscape * correlation + (1 - correlation) * random_matrix
            self.shocked_nature.all_landscapes_shock.append(shocked_landscape)
            self.shocked_nature.globalmax_shock = self.nature.get_globalmax(self.nature.inmat, self.shocked_nature.landscape_shock, self.n, self.p)
    
  
    def openFirstPhase(self, org_craft_err, time):
        """
        Open up the first phase of strategy making

        Parameters:
        org_craft_err: Error rate
        time: time step

        Returns:
        Add a decimal strategy into strategy_line
        """
        proposals = []

        # Process proposals and attach consensus of stakeholders' environment (Consensus environment)
        for pro in self.all_proposals:
            p1 = np.split(pro, 2)[0]
            p2 = np.hstack([p1, self.stake_es_position])
            proposals.append(pro)
        
        per_of_proposals = [] # A list ti keep the performance of proposals
        
        # Measure the performance of proposals
        for pro in proposals:
            if time < self.time_shock:
                per_of_proposals.append(self.nature.phi(None, pro)[0])
            else:
                per_of_proposals.append(self.shocked_nature.phi_shock(None, pro)[0])
        
        counter = len(per_of_proposals)
        x = [nk.bin2dec(proposal) for proposal in proposals]  # Convert proposals to decimal form
        
        err = org_craft_err
        tmp = np.zeros((counter, 2))
        
        for j in range(2):
            for i in range(counter):
                if j == 0:
                    tmp[i, j] = x[i]  # First column: decimal strategies
                elif j == 1:
                    tmp[i, j] = per_of_proposals[i] + random.normalvariate(0, err)  # Second column: performance of strategies
        
        # Selection of the strategy with the best performance
        sort1 = tmp[:, 1].argsort()
        sort1 = np.flip(sort1)
        tmp = tmp[sort1]
        decStrategyWithConsensusEnv = tmp[0, 0]
        
        # Convert the best strategy to binary form
        binStrategyWithConsensusEnv = self.dec_to_bin(decStrategyWithConsensusEnv)
        strategyWithoutEnv = np.split(binStrategyWithConsensusEnv, 2)[0]
        binStrategyWithRealEnv = np.hstack((strategyWithoutEnv, self.env))
        
        maximum_per = tmp[0, 1]
        # Current situation (strategy + env) of the organization
        bit0 = np.hstack((self.org_pos, self.env))
        
        # Measure the performance of the current strategy
        if time < self.time_shock:
            phi0 = self.nature.phi(None, bit0)[0]
        else:
            phi0 = self.shocked_nature.phi_shock(None, bit0)[0]
        
        # Simulated Annealing

        if self.annealing == "off":
            if phi0 < maximum_per:
                self.strategy_line[0, time] = nk.bin2dec(binStrategyWithRealEnv)
            else:
                self.strategy_line[0, time] = nk.bin2dec(bit0)
        
        elif self.annealing == "on":
            if np.random.rand() < self.metropolis(time, 0, 0):
                strategy1 = np.array([1] * (self.n * self.p))
                while nk.bin2dec(strategy1) != dormantStakeholder:
                    random_strategy_id = np.random.choice(len(proposals))
                    strategy1 = self.all_proposals[random_strategy_id]
                just_strategy = np.split(strategy, 2)[0]
                strategy2 = np.hstack((just_strategy, self.env))
                self.strategy_line[0, time] = nk.bin2dec(strategy2)
            else:
                if phi0 < maximum_per:
                    self.strategy_line[0, time] = nk.bin2dec(strategyrealenv)
                else:
                    self.strategy_line[0, time] = nk.bin2dec(bit0)


    def openSecondPhaseTideman(self,vote_err,time):
        """
        Open up the first and second phases of strategy making

        Parameters:
        vote_err: Stakeholders' errors while voting
        time: time step

        Returns:
        Add a decimal strategy into strategy_line
        """        
        stake_craft_err = vote_err # Stakeholders error while voting
        proposals = []

        for pro in self.all_proposals:
            p1 = np.split(pro,2)[0]
            p2 = np.hstack([p1,self.stake_es_position]) # Stakeholders use knowledge of crowd to check the environment
            proposals.append(pro)

        num = self.num
        tmp = np.zeros((num,2)) #keep performance of each proposal for one stakeholder
        vote_box = []
        number_of_attenders = 0
        for stake in self.stakes: #myid
            if nk.bin2dec(stake.proposal) != dormantStakeholder or self.voting_activate == 'on':#remove stakeholders who havent proposed
                number_of_attenders = number_of_attenders + 1
                for j in range(num):#make a list of all strategies
                    tmp[j,0]=j #startegy code
                    
                    if self.net == "off":
                        if time < self.time_shock:
                            tmp[j,1] = self.nature.phi(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                        else:
                            tmp[j,1] = self.shocked_nature.phi_shock(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                        
                    elif self.net == "on":
                             if time < self.time_shock:
                                 tmp[j,1] = self.nature.phi_h(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                             else:
                                 tmp[j,1] = self.shocked_nature.phi_shock_h(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                             
                    else:
                             raise Exception("Network situation is not determined")
                
                sort = tmp[:,1].argsort()
                sort = np.flip(sort)
                check_list = tmp[sort]#Ballot for stakeholder with id=myid
                #print('tmp',tmp)
                #print('check_list',check_list)
                ballot = check_list[:,0].tolist()
                vote_box.append(ballot)#aggregation of all stakeholders votes in the vote_box
            
        vote_box_array = np.array(vote_box)
        #print(vote_box_array)
        #making pairs
        proposals_id = [i for i in range(len(proposals))]#there are Ids from 0 to number of proposals -1
        pairs = combinations(proposals_id, 2)#making pairs for mutual comparison
        
        #save pairs in a list to be usable for several time (comb()) output is iterable just one time 
        all_combanitions = []
        for p in pairs:
            all_combanitions.append(p)
        #print(all_combanitions)
        
        add_prefrences_all = np.zeros(len(all_combanitions))#to save perferences of each stakeholder for each pair
        for stk in range(number_of_attenders):
            pair_comparison = []
            ballot_for_this_stakeholder = vote_box_array[stk,:].tolist()# check ballots of stakeholders one by one, it is not important whose ballot is it!
            for p in all_combanitions:
                first_strategy_in_pair = p[0]
                second_strategy_in_pair = p[1]
               
                #measure the index for strategies less index better rank!               
                if ballot_for_this_stakeholder.index(first_strategy_in_pair) > ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(-1)
                elif ballot_for_this_stakeholder.index(first_strategy_in_pair) == ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(0)
                elif ballot_for_this_stakeholder.index(first_strategy_in_pair) < ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(+1)
            #print(pair_comparison)
            pair_comparison_add = []#we add all stakeholders prefernces 
            add_prefrences_all = np.add(add_prefrences_all,pair_comparison)
        #print("add_prefrences_all",add_prefrences_all)
        
        #for the pairs with negative preference we switch the location of two pairs(A) and set the preference positive(B)
        k = 0
        all_comb_positive = []
        #(A)
        for p in all_combanitions:
            if add_prefrences_all[k] >= 0:
                all_comb_positive.append(p)
            elif add_prefrences_all[k] < 0:
                all_comb_positive.append((p[1],p[0]))
            k = k+1
        add_prefrences_all_positive = abs(add_prefrences_all)#(B)
        #print(add_prefrences_all_positive)
        #print(all_comb_positive)
        
        
        
        #sort pairs by number of prefrences
        sort_pairs = np.argsort(add_prefrences_all_positive)
        sort_pairs = np.flip(sort_pairs)
        sorted_all_comb_positive = np.array(all_comb_positive)[sort_pairs]
        #print(add_prefrences_all_positive[sort_pairs])
        #print(sorted_all_comb_positive)
        
        
        
        
        lock = np.zeros((len(proposals),len(proposals)))#Matrix to record locks
        
        for l in range(len(sorted_all_comb_positive)-1):
            pairA = sorted_all_comb_positive[l]
            pairB = sorted_all_comb_positive[l+1]
            #print("pairA",pairA)
            #print("pairB",pairB)
            #lock pairs over each other
            for strategy_raw in range(len(proposals)):
                for strategy_column in range(len(proposals)):
                    if strategy_raw == pairA[0] and strategy_column == pairA[1] and lock[strategy_raw,strategy_column] ==0 and strategy_column != strategy_raw:
                            lock[strategy_raw,strategy_column] = 1
                            lock[strategy_column,strategy_raw] = -1
                            #print("1",'\n',lock)
                            if pairA[1] == pairB[0] and lock[pairA[0],pairB[1]] == 0 :
                                #print(pairA)
                                #print(pairB)
                                lock[int(pairA[0]),int(pairB[1])] = 1
                                lock[int(pairB[1]),int(pairA[0])] = -1
                                #print("2","\n",lock)
                              
         #calculate score for each strategy
        lock_arr = np.array(lock)
        lock_sum_arr = np.sum(lock_arr,axis=1)
        strategy_score = np.zeros((len(proposals),2))
        for i in range(len(proposals)):
             strategy_score[i,0] = i
             strategy_score[i,1] = lock_sum_arr[i]
        
        
        #print(sorted_all_comb_positive)
        #print(strategy_score)
        sort_final_acend = np.argsort(strategy_score[:,1])
        sort_final_descend = np.flip(sort_final_acend)
        #print(sort_final_descend)
        sorted_strategies = strategy_score[:,0][sort_final_descend]        
        idx = sorted_strategies[0]
        #print(idx)
        
        a1 = proposals[int(idx)]# this strategy is with consensus environment to return it you need to attach it to real environment
        a2 = np.split(a1,2)[0]
        strategy = np.hstack((a2,self.env))
        
        if self.annealing == "off":
            self.strategy_line[0,time] = nk.bin2dec(strategy)
            return strategy
    
        elif self.annealing == "on":
           if np.random.rand() < self.metropolis(time,0,0):
                strategy1 = np.array([1]*self.n*self.p)
                while(nk.bin2dec(strategy1) != dormantStakeholder):
                    random_strategy_id = np.random.choice(len(proposals))
                    strategy1 = self.all_proposals[random_strategy_id]
                just_strategy = np.split(strategy,2)[0]
                #print(just_strategy)
                strategy2 = np.hstack((just_strategy,self.env))
                self.strategy_line[0,time] = nk.bin2dec(strategy2)
           else:
               self.strategy_line[0,time] = nk.bin2dec(strategy)
               return strategy

    
        
    def openSecondPhaseTidemanF(self,vote_err,time):
        """
        Open up the first and second phases of strategy making (In second phase stakeholders vote on a shortlist of proposlas provided by the organization)

        Parameters:
        vote_err: Stakeholders' errors while voting
        time: time step

        Returns:
        Add a decimal strategy into strategy_line
        """    
        crf_err = 0
        stake_craft_err = 0
        proposals = []
        per_of_proposals = []
        for pro in self.all_proposals:
            p1 = np.split(pro,2)[0]
            p2 = np.hstack([p1,self.stake_es_position])
            proposals.append(pro)
           
            i = 0
        for proposal in proposals:
            if time < self.time_shock:
                per_of_proposals.append(self.nature.phi(None,proposal)[0] + random.normalvariate(0,crf_err))
            else:
                per_of_proposals.append(self.shocked_nature.phi_shock(None,proposal)[0] + random.normalvariate(0,crf_err))
            i = i+1
        
        mean_of_per = statistics.mean(per_of_proposals)#calculate the average of the performances
        counter_1 = 0
        
        list_of_perf = np.zeros((len(proposals),2))
        
        p = 0
        for performance in per_of_proposals:# make a list of performances
            
            list_of_perf[p,0] = p
            list_of_perf[p,1] = performance
            p = p+1
        #print(list_of_perf)  
        
        sort5 = np.argsort(list_of_perf[:,1]) #sort performances
        #print(sort5)
        sorted_list = list_of_perf[sort5]
        #print(sorted_list)
        #print(mean_of_per)
        short_list = []
        for perf_id in enumerate(per_of_proposals): # remove strategies with performance less than average
            if perf_id[1] < mean_of_per:
                cut_number = perf_id[0]
            else:
                short_list.append(perf_id[0])
        
        num = self.num
        proposals_v = [] # a list to keep filtered strategies
        for pro_id in short_list:
            proposals_v.append(proposals[pro_id])
        num = len(proposals_v)


        #print('proposals',self.all_proposals)
        proposals = []
        for pro in proposals_v:
            p1 = np.split(pro,2)[0]
            p2 = np.hstack([p1,self.stake_es_position])#stakeholders use knowledge of crowd to check the environment
            proposals.append(p2)
            
        
        
        tmp = np.zeros((num,2)) #keep performance of each proposal for one stakeholder
        vote_box = []
        number_of_voters = 0
        for stake in self.stakes: #myid
            if nk.bin2dec(stake.proposal) != dormantStakeholder or self.voting_activate == 'on':
                number_of_voters = number_of_voters + 1
                for j in range(num):
                    tmp[j,0]=j#startegy code
                    
                    if self.net == "off":
                        if time < self.time_shock:
                            tmp[j,1] = self.nature.phi(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                        else:
                            tmp[j,1] = self.shocked_nature.phi_shock(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                        
                    elif self.net == "on":
                             if time < self.time_shock:
                                 tmp[j,1] = self.nature.phi_h(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                             else:
                                 tmp[j,1] = self.shocked_nature.phi_shock_h(stake.myid,proposals[j])[0] + random.normalvariate(0,stake_craft_err)
                             
                    else:
                             raise Exception("Network situation is not determined")
                
                sort = tmp[:,1].argsort()
                sort = np.flip(sort)
                check_list = tmp[sort]#Ballot for stakeholder with id=myid
            #print('tmp',tmp)
            #print('check_list',check_list)
                ballot = check_list[:,0].tolist()
                vote_box.append(ballot)#aggregation of all stakeholders votes in the vote_box
        vote_box_array = np.array(vote_box)
        #print(vote_box_array)
        #making pairs
        proposals_id = [i for i in range(len(proposals))]#there are Ids from 0 to number of proposals -1
        pairs = combinations(proposals_id, 2)#making pairs for mutual comparison
        
        #save pairs in a list to be usable for several time (comb()) output is iterable just one time 
        all_combanitions = []
        for p in pairs:
            all_combanitions.append(p)
        #print(all_combanitions)
        
        add_prefrences_all = np.zeros(len(all_combanitions))#to save perferences of each stakeholder for each pair
        for stk in range(number_of_voters):
            pair_comparison = []
            ballot_for_this_stakeholder = vote_box_array[stk,:].tolist()
            for p in all_combanitions:
                first_strategy_in_pair = p[0]
                second_strategy_in_pair = p[1]
               
                #measure the index for strategies less index better rank!               
                if ballot_for_this_stakeholder.index(first_strategy_in_pair) > ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(-1)
                elif ballot_for_this_stakeholder.index(first_strategy_in_pair) == ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(0)
                elif ballot_for_this_stakeholder.index(first_strategy_in_pair) < ballot_for_this_stakeholder.index(second_strategy_in_pair):
                   pair_comparison.append(+1)
            #print(pair_comparison)
            pair_comparison_add = []#we add all stakeholders prefernces 
            add_prefrences_all = np.add(add_prefrences_all,pair_comparison)
        #print("add_prefrences_all",add_prefrences_all)
        
        #for the pairs with negative preference we switch the location of two pairs(A) and set the preference positive(B)
        k = 0
        all_comb_positive = []
        #(A)
        for p in all_combanitions:
            if add_prefrences_all[k] >= 0:
                all_comb_positive.append(p)
            elif add_prefrences_all[k] < 0:
                all_comb_positive.append((p[1],p[0]))
            k = k+1
        add_prefrences_all_positive = abs(add_prefrences_all)#(B)
        #print(add_prefrences_all_positive)
        #print(all_comb_positive)
        
        
        
        #sort pairs by number of prefrences
        sort_pairs = np.argsort(add_prefrences_all_positive)
        sort_pairs = np.flip(sort_pairs)
        sorted_all_comb_positive = np.array(all_comb_positive)[sort_pairs]
        #print(add_prefrences_all_positive[sort_pairs])
        #print(sorted_all_comb_positive)
        
        
        
        
        lock = np.zeros((len(proposals),len(proposals)))#Matrix to record locks
        
        for l in range(len(sorted_all_comb_positive)-1):
            pairA = sorted_all_comb_positive[l]
            pairB = sorted_all_comb_positive[l+1]
            #print("pairA",pairA)
            #print("pairB",pairB)
            #lock pairs over each other
            for strategy_raw in range(len(proposals)):
                for strategy_column in range(len(proposals)):
                    if strategy_raw == pairA[0] and strategy_column == pairA[1] and lock[strategy_raw,strategy_column] ==0 and strategy_column != strategy_raw:
                            lock[strategy_raw,strategy_column] = 1
                            lock[strategy_column,strategy_raw] = -1
                            #print("1",'\n',lock)
                            if pairA[1] == pairB[0] and lock[pairA[0],pairB[1]] == 0 :
                                #print(pairA)
                                #print(pairB)
                                lock[int(pairA[0]),int(pairB[1])] = 1
                                lock[int(pairB[1]),int(pairA[0])] = -1
                                #print("2","\n",lock)
                              
         #calculate score for each strategy
        lock_arr = np.array(lock)
        lock_sum_arr = np.sum(lock_arr,axis=1)
        strategy_score = np.zeros((len(proposals),2))
        for i in range(len(proposals)):
             strategy_score[i,0] = i
             strategy_score[i,1] = lock_sum_arr[i]
        
        
        #print(sorted_all_comb_positive)
        #print(strategy_score)
        sort_final_acend = np.argsort(strategy_score[:,1])
        sort_final_descend = np.flip(sort_final_acend)
        #print(sort_final_descend)
        sorted_strategies = strategy_score[:,0][sort_final_descend]        
        idx = sorted_strategies[0]
        #print(idx)
        a1 = proposals[int(idx)] # strategy attached to errored environment
        a2 = np.split(a1,2)[0] # seperate strategy from consensus environment
        
        strategy = np.hstack((a2,self.env))
               
        if self.annealing == "off":
            self.strategy_line[0,time] = nk.bin2dec(strategy)
            return strategy 

        
        elif self.annealing == "on":
           if np.random.rand() < self.metropolis(time,0,0):
                strategy1 = np.array([1]*self.n*self.p)
                while(nk.bin2dec(strategy1) != dormantStakeholder):
                    random_strategy_id = np.random.choice(len(proposals))
                    strategy1 = self.all_proposals[random_strategy_id]
                just_strategy = np.split(strategy1,2)[0]
                strategy2 = np.hstack((just_strategy,self.env))
                self.strategy_line[0,time] = nk.bin2dec(strategy2)
           else:
               self.strategy_line[0,time] = nk.bin2dec(strategy)
               return strategy      
    

    def closed(self,error_p,error_e,eff,ham,time):
        """
        Closed strategy making

        Parameters:
        error_p: Organizations' error about the performance
        error_e: Organizations' error about the status of the environment
        eff: Number of investigated strategies before selection
        ham: The hamming distance for local search
        time: time step

        Returns:
        Add a decimal strategy into strategy_line
        """    
        env = self.env
        stk = self.org_pos
        '''current position without Error'''
        bit0 = np.hstack((stk,env))
        '''perceive environment with error'''
        env_e = self.with_noise(env,pb= error_e)
        '''current position with error''' 
        bit0e = np.hstack((stk,env_e))
        '''a list to keep errored positions'''
        exploration_list_err = []
        exploration_list_real = []
        
        ''' selection of  other random positions for one stakeholder(effort)'''
        combs = []#A list to keep possible combinations
        x = [i for i in range(self.n)]#List of bits in the first landscape(organization)
        for i in range(1,ham+1): # make all the possible neighbors with hamming distance less than ham+1
            comb = combinations(x, i)
            for c in comb:
                combs.append(c)
        
        random_bits = random.sample([n for n in range(len(combs))],1) # choose one random location in hamming distance = ham      
        ex_list = [] # a list to keep selected  neighbors 
        for x in random_bits:# change the corresponding bits
            ex_list.append(combs[x])
            for tup in ex_list:
                y = bit0e.copy()
                for num in tup:
                    if y[num] == 0:
                        y[num]=1
                    elif y[num] == 1:
                        y[num]=0
 
                exploration_list_err.append(y)
                exploration_candidate_real = np.hstack(( np.split(y,2)[0],env))
                exploration_list_real.append(exploration_candidate_real)
                
               
        performances = []  
        performances_err = []
            
        # Calculate performance for candidate positions
        for position in exploration_list_err:
            if time < self.time_shock:
                performances_err.append(self.nature.phi(None,position)[0] + random.normalvariate(0,error_p))
                phi0 = self.nature.phi(None,bit0e)[0] 
                
            elif time >= self.time_shock:
                performances_err.append(self.shocked_nature.phi_shock(None,position)[0] + random.normalvariate(0,error_p))
                phi0 = self.shocked_nature.phi_shock(None,bit0e)[0]

           
        # find best errored performance
        maximum_per = max(performances_err)
        # find index of best  errored performance 
        idx = performances_err.index(maximum_per)
        # look for the real position with the same index
        real_position = exploration_list_real[idx]
        # look for the errored position with the same index 
        
        # Simulated annealing
        if self.annealing == "off":
            if maximum_per > phi0:
                bit1 = real_position 
                self.strategy_line[0,time] = nk.bin2dec(bit1)
            else:
                bit1 = bit0
                self.strategy_line[0,time] = nk.bin2dec(bit1)
        elif self.annealing == "on":
           if np.random.rand() < self.metropolis(time,phi0,maximum_per) or maximum_per > phi0:
               bit1 = real_position
               self.strategy_line[0,time] = nk.bin2dec(bit1)
           else:
                self.strategy_line[0,time] = nk.bin2dec(bit0)
                

    def implementation(self,time):
        """Updates the organization's position based on strategy at a given time."""
        strategy_at_time = self.strategy_line[0, time]
        binary_representation = self.dec_to_bin(strategy_at_time)
        self.org_pos = np.split(binary_representation, 2)[0]
    
        
    def implement_sat(self, timestamp):
        """Updates the organization's position based on strategy at a given time based on stakeholders average performance"""
        
        
        strategy_at_time = self.dec_to_bin(self.strategy_line[0, timestamp])
        intended_strategy = np.split(strategy_at_time, 2)[0]
    
        # Calculate average satisfaction and frustration among stakeholders
        avg_satisfaction = sum(stake.satis for stake in self.stakes) / len(self.stakes)
        avg_frustration = sum(stake.frustration for stake in self.stakes) / len(self.stakes)
    
        move_probability = avg_satisfaction / (avg_frustration + avg_satisfaction)
        
        # Determine if implementation is successful
        success = np.random.choice(2, p=[1 - move_probability, move_probability])
        
        if success:
            self.org_pos = intended_strategy  # Successfully update the organization position
            self.movements += 1
        else:
            # Apply noise to the strategy if the implementation is not successful
            self.org_pos = self.with_noise(intended_strategy, 1)


    def environment_change_random(self,speed,pb,time):
    
        """Update the environment randomly."""
        if np.random.choice(2, p=[1-pb, pb]) == 1:
            for _ in range(speed):
                self.env = self.with_noise(self.env, 1)
            self.current_state = np.hstack((self.org_pos, self.env))
    
      
    
    
    def dec_to_bin(self, dec_strategy):
        """Convert decimal strategy to binary representation."""
        
        output = nk.dec2bin(dec_strategy, 0).astype(np.int32)
        
        desired_length = self.n * self.p
        current_length = len(output)
        
        if current_length < desired_length:
            padding = desired_length - current_length
            output = np.hstack((np.zeros(padding, dtype=int), output))
        
        return output



    def stake_feedback(self,eff_change,time):
        """
        Stakeholders reaction to the selected strategy
        """
        for stake in self.stakes:
            timeSwitch = self.timeSwitch
            stake.modify_test(time)
            
              
            
   
    def environment_estimation(self):
        """Estimation of the environment based on stakeholders' errored perceptions (Knowledge of crowd)"""
        consensus = []

        num = len(self.all_environments)
    
        for i in range(self.n):  # Calculate consensus for each bit
            bit_sum = sum([env[i] for env in self.all_environments])
            consensus.append(bit_sum)
    
        tmp_arr = np.array([1 if bit >= num / 2 else 0 for bit in consensus])
    
        self.stake_es_position = tmp_arr
        return tmp_arr
        
        
     
    def with_noise(self,vec,pb):
        """
        Generate an errored version of a bit string

        Parameters: 
        vec: the bit string
        pb: the probability of error to happen
        
        Returns:
        An errored bit string with the probability of pb
        """
        tmp = vec.copy()
        nvec = len(vec)
        error_number = np.random.choice([1,2],p=[0.5,0.5])#Error gonna happen in distance = 1 or 2
        noise = np.random.choice(2,p=[1-pb, pb])
        #print(noise)
        if noise:
            rnds = np.random.choice((range(nvec)),error_number)
            for rnd in rnds:
                if tmp[rnd] == 0:
                    tmp[rnd] = 1
                else:
                    tmp[rnd] = 0

            output = tmp
            return output
        else:
           return vec
       
    def report_strategy_line(self,time):
          strategy_string = []
          strategy = self.strategy_line[0,time]
          strategy_string.append(self.dec_to_bin(strategy))
          #print(time)
          #print( self.dec_to_bin(strategy))
            
          if time == self.t:
              return self.strategy_string

    def metropolis(self,time,perf1,perf2):
        tmp = 9/(1+time)
        criterion = math.exp((-1)/tmp)
        return criterion
   

class Stake:
    ''' Decides on tasks, interacts with peers; aggregation relation with Organization class.'''
    def __init__(self,employer,myid,correlation,s_e_b,s_e_p,effort,frustration, satis):
        self.employer = employer
        self.myid = myid
        self.nature = employer.nature
        self.shocked_nature = employer.shocked_nature
        self.n = employer.n
        self.p = employer.p
        self.t = employer.t
        self.frustration = frustration
        self.satis = satis
        
        self.current_util = 0.0
        self.current_perf = 0.0
        '''features'''
        self.eff = effort
        self.current_state = None
        self.eff_float = effort
        '''Errors'''
        self.s_e_b = s_e_b
        self.s_e_p = s_e_p
        self.correlation = correlation
        '''history'''
        self.proposal = None
        self.stake_proposals = []
        self.probability = 0
        self.comeback = 0
        self.t_quit = None
        self.last_strategy = None
        '''network'''
        self.num = self.employer.num
        self.s_network = [1]*self.employer.num
        self.memory = []
        self.memory_sat = []
        self.memory_dis_sat = []
        self.leave = 1
        self.active = True
        
  
    def get_decay_weight(self, elapsed_time, decay_rate=.1):
        return math.exp(-decay_rate * elapsed_time)
    
    def get_weighted_mean(self, memory, current_time):
        # Assuming each memory is a tuple (time, experience)
        total_weight = 1
        total_experience = .5
        for mem in self.memory:
            elapsed_time = current_time - mem[0]
            weight = self.get_decay_weight(elapsed_time)
            total_weight += weight
            total_experience += weight * mem[1]
        return total_experience / total_weight if total_weight else 0

    def modify_test(self, t):
        time = t
        # Observing the selected strategy
        selected_strategy_dec = int(self.employer.strategy_line[0, t])
        selected_strategy_binary = self.employer.dec_to_bin(selected_strategy_dec)
    
        # Observing the  proposed strategy by the stakeholder
        stake_strategy = self.stake_proposals[t - 1]
    
        # Calculating performance of selected strategy and proposed strategy for the stakeholder and subtraction between them (before and after the time shock)
        if time < self.employer.time_shock:
            stake_strategy_performance = self.nature.phi(self.myid, stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p / 10)) # Performance of proposed strategy for the stakeholder
            selected_strategy_performance = self.nature.phi(self.myid, selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p / 10)) # Performance of the selected strategy for the stakeholder
        else:  # time >= self.employer.time_shock
            stake_strategy_performance = self.shocked_nature.phi_shock(self.myid, stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p / 10))
            selected_strategy_performance = self.shocked_nature.phi_shock(self.myid, selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p / 10))
    
        # Adjusting satisfaction or frustration based on strategy performance
        if stake_strategy_performance <= selected_strategy_performance:
            add_to_sat = selected_strategy_performance - stake_strategy_performance
            if nk.bin2dec(stake_strategy) == dormantStakeholder:
                add_to_sat = 0
            self.satis += add_to_sat
            self.eff_float += add_to_sat
            self.memory_sat.append((t, add_to_sat))
            self.memory_dis_sat.append((t, 0))
    
        elif stake_strategy_performance > selected_strategy_performance:
            add_to_frust = stake_strategy_performance - selected_strategy_performance
            if nk.bin2dec(stake_strategy) == dormantStakeholder:
                add_to_frust = 0
            self.frustration += add_to_frust
            self.eff_float -= add_to_frust
            self.memory_sat.append((t, 0))
            self.memory_dis_sat.append((t, add_to_frust))
    
        weighted_mean_dis_sat = self.get_weighted_mean(t, self.memory_dis_sat)
        weighted_mean_sat = self.get_weighted_mean(t, self.memory_sat)
    
        # Adjusting effort based on the float value and bounds
        x = self.eff_float - self.eff
        if abs(x) >= 0.5:
            self.eff += np.sign(x)
    
        if self.eff <= 0:
            self.eff = 0
            self.leave += 1
            if self.active:
                self.active = False
    
        # Decision for a stakeholder to come back
        p_in = weighted_mean_sat / (weighted_mean_sat + weighted_mean_dis_sat)
        p_out = 1 - p_in
        if self.eff >= 36:
            self.eff = 36
        comeback = np.random.choice(2, p=[1 - p_in, p_in])
        if self.eff <= 0 and comeback:
            self.eff_float = 0.5
            self.active = True

                
    def modify(self,eff_change,timeSwitch,t):
            time = t
            #observing the utility of the selected strategy for the stakeholder
            selected_strategy_dec = int(self.employer.strategy_line[0,t])
            selected_strategy_binary = self.employer.dec_to_bin(selected_strategy_dec)
            
            #observing the utility of the proposed strategy for the stakeholder
            stake_strategy = self.stake_proposals[t-1]
            
            
            if self.employer.net == "off":
            
                if time < self.employer.time_shock:                
                    stake_strategy_performance = self.nature.phi(self.myid,stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p/10))
                    selected_strategy_performance = self.nature.phi(self.myid,selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p/10))
                elif time >= self.employer.time_shock:
                    stake_strategy_performance = self.shocked_nature.phi_shock(self.myid,stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p/10))
                    selected_strategy_performance = self.shocked_nature.phi_shock(self.myid,selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p/10))
            
            elif self.employer.net == "on":
                if time < self.employer.time_shock:
                    stake_strategy_performance = self.nature.phi_h(self.myid,stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p/10))
                    selected_strategy_performance = self.nature.phi_h(self.myid,selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p/10))
                else:
                    stake_strategy_performance = self.shocked_nature.phi_shock_h(self.myid,stake_strategy)[0] + random.normalvariate(0, float(self.s_e_p/10))
                    selected_strategy_performance = self.nature.phi_shock_h(self.myid,selected_strategy_binary)[0] + random.normalvariate(0, float(self.s_e_p/10))
        

            else:
                raise Exception("The network situation is not determined")
            
            
            #network
            idx = 0
            for strategy in self.employer.all_proposals:
                if nk.bin2dec(strategy) == dormantStakeholder:
                    idx += 1
                    break
                elif all(strategy) == all(stake_strategy):
                    self.s_network[idx] = self.s_network[idx] + 1
                    idx += 1
                
            
            if stake_strategy_performance <= selected_strategy_performance:
                
                #effort
                add_to_sat = selected_strategy_performance - stake_strategy_performance
                if nk.bin2dec(stake_strategy) == dormantStakeholder:
                   add_to_sat = 0 
                self.satis = self.satis + math.erf(add_to_sat)
                #self.satis = self.satis + add_to_sat
                self.eff_float = self.eff_float + add_to_sat

            elif stake_strategy_performance > selected_strategy_performance:
                add_to_frust = stake_strategy_performance - selected_strategy_performance
                if nk.bin2dec(stake_strategy) == dormantStakeholder:
                   add_to_frust = 0 
                self.frustration = self.frustration + math.erf(add_to_frust)
                #self.frustration = self.frustration + add_to_frust
                self.eff_float = self.eff_float - add_to_frust
                
                #for stake in self.employer.stakes:
                    #if nk.bin2dec(stake.proposal) == dormantStakeholder:
                        #break
                    #if all(stake.proposal) == all(selected_strategy_binary):
                        
                        #self.s_network[stake.myid] -= dec
                        #if self.s_network[stake.myid] < 0:
                            #self.s_network[stake.myid] = 0
                            
                        
             
            # Add to the memory
            if stake_strategy_performance <= selected_strategy_performance:
                self.memory.append((t, add_to_sat))
            else:
                self.memory.append((t, -add_to_frust))
            
            # Calculate weighted mean
            #weighted_mean = self.get_weighted_mean(t)
            #self.eff_float += weighted_mean
            
            #round the effort
         
            x = self.eff_float - self.eff
            if abs(x) >= 0.5:
               self.eff = self.eff + np.sign(x)
               
            
            if self.eff <= 0 :
                self.eff = 0
                self.leave = self.leave + 1
                if self.active:
                    self.active = False

            
                
           #print(t, ":",self.eff)
                    
            p_in = (self.satis/ (self.satis + self.frustration))#/self.leave
            p_out = 1-p_in
            if self.eff >= 36:
                self.eff = 36
            comeback = np.random.choice(2,p=[1-p_in,p_in]) 
            if self.eff <= 0 and comeback:
                self.eff_float = 0.5
                self.active = True
            #if (t == 90, t==10 , t==30 , t==60) and self.employer.name == "firm2":
                #print("stakeholder",self.myid, self.s_network)
            #print(time,":", self.leave)
 
    def perform_climb(self, time, ham):
        """
        Stakeholder explores the neighborhood of the current strategy
        Parameters:
        time: time step
        ham: hamming distance
        Returns:
        A decimal strategy
        """
        if self.eff == 0:
            self.proposal = np.array([1] * self.n * self.p)
            return
        elif self.eff < 0:
            print("effort is negative", self.eff)
            return
    
        env = self.employer.env  # Environment location
        stk = self.employer.org_pos  # Organization location
        bit0 = np.hstack((stk, env))  # current position without Error
    
        env_e = self.with_noise(env, self.s_e_b)  # perceive environment with error
        self.employer.all_environments.append(env_e) # We save all the perceived environments by stakeholders in a list
        bit0e = np.hstack((stk, env_e))  # current position with error
    
        # Prepare combinations for exploration
        x = list(range(self.n))
        combs = [comb for i in range(1, ham + 1) for comb in combinations(x, i)]
        random_bits = random.sample(combs, int(self.eff))
    
        exploration_list_err, exploration_list_real = [], []
    
        for tup in random_bits:
            y = bit0e.copy()
            for num in tup:
                y[num] = 1 - y[num]  # Toggle bit
    
            exploration_list_err.append(y)
            exploration_candidate_real = np.hstack((np.split(y, 2)[0], env))
            exploration_list_real.append(exploration_candidate_real)
    
        performances_err = []
        stake_error = random.normalvariate(0, float(self.s_e_p/10))
    
        # Calculate performance for the strategies
        for position in exploration_list_err:
            if self.employer.net == "off":
                if time < self.employer.time_shock:
                    phi_func = self.nature.phi
                else:
                    phi_func = self.shocked_nature.phi_shock
            elif self.employer.net == "on":
                if time < self.employer.time_shock:
                    phi_func = self.nature.phi_h
                else:
                    phi_func = self.shocked_nature.phi_shock_h
            else:
                raise Exception(f"employer.net {self.employer.name} at time {time} is not determined")
    
            performances_err.append(phi_func(self.myid, position)[0] + stake_error)
    
        # Determine best performance and proposal
        maximum_per = max(performances_err)
        idx = performances_err.index(maximum_per)
        errored_position = exploration_list_err[idx]
        self.proposal = errored_position.copy()
       

    
    
    def stake_report(self,time):
          
        self.employer.all_proposals.append(self.proposal)
        self.stake_proposals.append(self.proposal)
        #self.employer.black_list.append(self.myid)
        
        
    def with_noise(self,vec,pb):
        """
        Parameters
        ----------
        vec : a bit string
            DESCRIPTION.
        pb : probability of change in the bit
            DESCRIPTION.

        Returns
        -------
        a bit string with the same length that one or two of its bits has/have changed
            DESCRIPTION.

        """
        tmp = vec.copy()
        nvec = len(vec)
        Error_number = np.random.choice([1,2],p=[0.5,.5])
        noise = np.random.choice(2,p=[1-pb, pb])
        #print(noise)
        if noise:
            rnds = np.random.choice((range(nvec)),Error_number)
            for rnd in rnds:
                if tmp[rnd] == 0:
                    tmp[rnd] = 1
                else:
                    tmp[rnd] = 0

            output = tmp
            return output
        else:
           return vec
    

    

class Nature:
    '''Defines the performances, inputs state, outputs performance; a hidden class.'''
    def __init__(self,p,n,k,c,s,t,rho,correlations,num):
        self.p = p
        self.n = n
        self.k = k
        self.c = c
        self.s = s
        self.t = t
        self.num = num
        self.correlations = correlations
        self.inmat = None
        self.landscape = None
        self.landscape_shock = None
        self.globalmax = None
        self.globalmax_shock = None
        self.current_state = np.zeros(n*p,dtype=np.int8)
        self.all_landscapes = []
        self.all_landscapes_shock = []
        self.stakes = None

        
    def set_interactions(self):
        '''sets interaction matrices'''
        p = self.p
        n = self.n
        k = self.k
        c = self.c
        s = self.s
        tmp = np.zeros((n*p, n*p),dtype=np.int8)
        if s>(p-1):
            return
        couples = nk.generate_couples(p,s)
        
        # the idea is to have similar interaction for rho=1 to work
        internal = nk.interaction_matrix(n,k,"random")
        external = nk.random_binary_matrix(n,c)
        # internal coupling
        for i in range(p):
            tmp[i*n:(i+1)*n, i*n:(i+1)*n] = internal

        # external coupling
        for i,qples in zip(range(p),couples):
            for qple in qples:
                tmp[i*n:(i+1)*n, qple*n:(qple+1)*n] = external
        self.inmat = tmp
        
    def set_landscapes(self):
        '''sets landscapes; set gmax=False to skip calculating global maximum'''
        p = self.p
        n = self.n
        k = self.k
        c = self.c
        s = self.s
        contrib = self.contrib_define(p,n,k,c,s)
        self.landscape = contrib
        self.globalmax = self.get_globalmax(self.inmat, contrib, n, p)
        
        
        
    def phi_h(self, myid, x):
        # None id id for the organization
        if myid is None:
            return self.phi(myid, x)
        else:
            # DormantStakeholder is the case in that stakeholder doesnt propose strategy
            if nk.bin2dec(x) == dormantStakeholder:
                return np.array([0, 0])
            # Load the reciprocal matrix for the stakeholder with given id
            s_network = self.stakes[myid].s_network
            # How much stakeholders give weight to other stakeholders while calculating the performance 
            herd = 0.5
    
            
            # Check for the impact of the other stakeholders by checking reciprocal matrix
            impacts = []
            all_nodes = 0
            
            '''
            for index, node in enumerate(s_network):
                if index == myid:
                    node = 0
                impacts.append(math.exp(node) * self.phi(index, x)[0])
                all_nodes += math.exp(node)
                
            '''
            for index, node in enumerate(s_network):
                # The recprocal effect of each stakeholder on itself is zero.
                # Index denotes the id and node denotes the weight of effect
                if index == myid:
                    node = 0
                impacts.append(node * self.phi(index, x)[0])
                all_nodes += node
            
            
    
            
            total_impact = sum(impacts)
            total_impact /= all_nodes
            
            its_phi = self.phi(myid, x)[0]
            performance = (1 - herd) * its_phi  + herd * total_impact
     
    
            return np.array([performance, self.phi(myid, x)[1]])

     
    def phi(self,myid,x):
        '''inputs bitstring, outputs performance; set gmax=False to skip calculating global maximum'''
        if nk.bin2dec(x) == dormantStakeholder:
            #print("x1",x)
            return np.array([0,0])
            
        n = self.n
        p = self.p
        n_p = n*p
        imat = self.inmat
        #print(x)
        if myid == None:
            cmat = self.landscape
            globalmax = self.globalmax
        else:
            cmat = self.all_landscapes[myid]
            globalmax = np.array([1,1])
            
        if len(x)  != n_p :
            print("Error: Please enter the full bitstring")
            return
   
        tmp = np.array(self.contrib_solve(x,imat,cmat,n,p))/globalmax
        output = tmp 
        
        return output

    #First approach: makeing correlated NK landscapes
    def contrib_define2(self,p,n,k,c,s):
        if  1:#self.cluster_marker == False:
            all_correlations = []
            for corr in self.correlations:
                all_correlations.append(corr)
            
            all_correlations.insert(0,1)
            # put all_correlations inside sin(pi/6* dsjfkjdklfj)
            corrmat = np.zeros((self.num+1,self.num+1)) + (1) * np.eye(self.num+1)
            corrmat[0,:] = all_correlations
            corrmat[:,0] = all_correlations
            #print(all_correlations)
            #print(corrmat)
            x = len(all_correlations)
            cm = np.zeros((x,x))
            for i in range(x):
                for j in range(x):
                    if j != i and j != 0 and i != 0:
                        corrmat[i,j] = self.correlations[0]
            #print(corrmat)
        #else:
            #corrmat = self.clusterd_matrix
       
     
        cm = 2*np.sin((np.pi / 6 )*corrmat)
        #print(cm)
        cm2 = self.create_spd_matrix(cm)
        #print(cm2)
        #cm2 = self.create_spd_matrix2(cm)
        tmp1 = np.random.multivariate_normal(mean=[0]*(self.num+1), cov=cm2, size=(n*2**(1+k+c*s)),check_valid='warn')
        out1 = norm.cdf(tmp1)[:,0]
        out1 = np.reshape(out1.T, (n, (2**(1+k+c*s)))).T
        
        tmp2 = np.random.multivariate_normal(mean=[0]*(self.num+1), cov=cm2, size=(n*2**(1+k+c*s)),check_valid='warn')
        out2 = norm.cdf(tmp2)[:,0]
        out2 = np.reshape(out2.T, (n, (2**(1+k+c*s)))).T       
        tmp3 = np.column_stack((out1,out2))
        output = tmp3
        
        
        
        
        for i in range(1,self.num+1):
            t1 = norm.cdf(tmp1)[:,i]           
            
            
            out1_l = np.reshape(t1.T, (n, (2**(1+k+c*s)))).T
            
            t2 = norm.cdf(tmp2)[:,i]
            out2_l = np.reshape(t2.T, (n, (2**(1+k+c*s)))).T
            corr_land = np.column_stack((out1_l,out2_l))
            
            self.all_landscapes.append(corr_land)
        
        reshaped_data = np.reshape(self.all_landscapes, (10, -1))
        
        # Create a DataFrame from the reshaped data
        df = pd.DataFrame(reshaped_data)
        
        # Save the DataFrame to Excel
        df.to_excel('allLandscapes.xlsx', index=False)
        return output
    
 



    def create_spd_matrix(self,A):
        n = A.shape[0]
        X = cp.Variable((n, n), symmetric=True)
        
        # Constraints
        constraints = [X >> 0]  # Ensure positive semidefiniteness
        
        mask = np.ones((n, n)) - np.eye(n)
        constraints.extend([cp.multiply(mask, X) <= 1, cp.multiply(mask, X) >= -1])
        
        # Objective with adjusted weights
        objective = cp.Minimize(
            100 * cp.norm(X - np.eye(n), 'fro') +  # heavily prioritize diagonal elements being 1
            50 * cp.norm(X[0, :] - A[0, :], 2) +  # heavily prioritize first row matching
            50 * cp.norm(X[:, 0] - A[:, 0], 2)    # heavily prioritize first column matching
        )
        
        # Solve
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-9, verbose=False)
        
        if X.value is None:
            raise ValueError("Solution does not exist!")
        return X.value
    

        
    def contrib_define(self,p,n,k,c,s):
        """
        

        Parameters
        ----------
        p : int
            always 2.
        n : int
            number of bits in the strategy.
        k : int
            number of internal interactions.
        c : int
            number of external interactions.
        s : int
            always 1: denotes existance of the environment.

        Returns
        -------
        original_landscape : np array
        correlated landscapes: a list of np array
        """
        
        #make a list of all correlation values
        all_correlations = []
        for corr in self.correlations:
            all_correlations.append(corr)
        
        rho = 0
        corrmat = np.repeat(rho,p*p).reshape(p,p) + (1-rho) * np.eye(p)
        corrmat = 2*np.sin((np.pi / 6 ) * corrmat)
        tmp = np.random.multivariate_normal(mean=[0]*p, cov=corrmat, size=(n*2**(1+k+c*s)))
        tmp = norm.cdf(tmp)
        tmp = np.reshape(tmp.T, (n*p, (2**(1+k+c*s)))).T
        original_landscape = tmp
  
        
        for i in range(0,self.num):
            crrelation_value = all_correlations[i]
            correlated_landscape = self.generate_correlated_NK_landscape(original_landscape,  crrelation_value)
            self.all_landscapes.append(correlated_landscape)
        return original_landscape


    def generate_correlated_NK_landscape(self,original_landscape, corr):
        """
        Generate a correlated NK landscape based on a given original landscape using Gaussian Copula.
        
        Parameters:
        - original_landscape: The original NK landscape.
        - corr: The desired correlation coefficient.
        
        Returns:
        - correlated_landscape: The correlated NK landscape.
        """

        # Helper function to compute empirical CDF
        def empirical_cdf(data, value):
            return np.mean(data <= value)

        # Helper function to compute empirical inverse CDF
        def empirical_icdf(data, p):
            sorted_data = np.sort(data)
            size = len(sorted_data)
            if p <= 0:
                return sorted_data[0]
            elif p >= 1:
                return sorted_data[-1]
            idx = int(p * size)
            return sorted_data[idx]

        # 1. Flatten original landscape and transform to uniform marginals
        flattened_original = original_landscape.flatten()
        uniform_marginals = np.array([empirical_cdf(flattened_original, v) for v in flattened_original])

        # 2. Convert to Gaussian marginals
        gaussian_marginals = stats.norm.ppf(uniform_marginals)

        # 3. Induce desired correlation
        # Generate standard normal values
        standard_normals = np.random.normal(size=len(gaussian_marginals))
        # Scale and shift them to have the desired correlation with the gaussian_marginals
        correlated_gaussians = corr * gaussian_marginals + np.sqrt(1 - corr**2) * standard_normals

        # 4. Convert back to uniform marginals
        correlated_uniform = stats.norm.cdf(correlated_gaussians)

        # Transform back to the original distribution
        correlated_landscape_flattened = np.array([empirical_icdf(flattened_original, p) for p in correlated_uniform])
        correlated_landscape = correlated_landscape_flattened.reshape(original_landscape.shape)
        
        return correlated_landscape


    
    
    
    def get_globalmax(self,imat,cmat,n,p):
            n_p = n*p
            output = None
            perfmax = [0.0]*p#np.zeros(p,dtype=float)
            for i in range(2**n_p):
                bval = self.contrib_solve(nk.dec2bin(i,n_p),imat,cmat,n,p)
                if bval[0] > perfmax[0]:
                    perfmax = bval
    
            output = np.array([perfmax[0],1])
            return output
    
    def contrib_solve(self,x,imat,cmat,n,p):
        """Computes a performance for vector x from given contribution matrix and interaction matrix

        Notes:
            Uses Numba's njit compiler.

        Args:
            x: An input vector
            imat (numpy.ndarray): Interaction matrix
            cmat (numpy.ndarray): Contribution matrix
            n (int): Number of tasks per landscape
            p (int): Number of landscapes (population size)

        Returns:
            float: A mean performance of an input vector x given cmat and imat.
        """
        
        n_p = n*p
        phi = np.zeros(n_p)
        for i in range(n_p):
            tmp = x[np.where(imat[:,i]>0)] # coupled bits
            tmp_loc = nk.bin2dec(tmp) # convert to integer
            phi[i] = cmat[tmp_loc,i]

        output = [0.0]*p
        for i in range(p):
            output[i] = phi[i*n : (i+1)*n].mean()
        return output
    
    
class shocked_nature:
    '''Defines the performances, inputs state, outputs performance; a hidden class.'''
    def __init__(self,p,n,k,c,s,t,rho,correlations,num,inmat):
        self.p = p
        self.n = n
        self.k = k
        self.c = c
        self.s = s
        self.t = t
        self.num = num
        self.correlations = correlations
        self.landscape_shock = None
        self.globalmax_shock = None
        self.current_state = np.zeros(n*p,dtype=np.int8)
        self.all_landscapes_shock = []
        self.imat = inmat
    

    
    def phi_shock(self,myid,x):
        '''inputs bitstring, outputs performance; set gmax=False to skip calculating global maximum'''
        
        if nk.bin2dec(x) == nk.bin2dec(np.array([1]*self.n*self.p)):
            return np.array([0,0])
        n = self.n
        p = self.p
        n_p = n*p
        imat = self.imat
        if myid == None:
            cmat = self.landscape_shock
            globalmax = self.globalmax_shock
        else:
            cmat = self.all_landscapes_shock[myid]
            globalmax = np.array([1,1])
        if len(x) != n_p:
            print("Error: Please enter the full bitstring")
            return
        tmp = np.array(self.contrib_solve(x,imat,cmat,n,p))/globalmax
        output = tmp 
        return output


    def contrib_solve(self, x,imat,cmat,n,p):
        """Computes a performance for vector x from given contribution matrix and interaction matrix

        Notes:
            Uses Numba's njit compiler.

        Args:
            x: An input vector
            imat (numpy.ndarray): Interaction matrix
            cmat (numpy.ndarray): Contribution matrix
            n (int): Number of tasks per landscape
            p (int): Number of landscapes (population size)

        Returns:
            float: A mean performance of an input vector x given cmat and imat.
        """
        
        n_p = n*p
        phi = np.zeros(n_p)
        for i in range(n_p):
            tmp = x[np.where(imat[:,i]>0)] # coupled bits
            tmp_loc = nk.bin2dec(tmp) # convert to integer
            phi[i] = cmat[tmp_loc,i]

        output = [0.0]*p
        for i in range(p):
            output[i] = phi[i*n : (i+1)*n].mean()
        return output
