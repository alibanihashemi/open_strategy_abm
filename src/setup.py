   
    
# Standard library imports
import os
import random
import time
from datetime import date, datetime
from math import sqrt
import math
from multiprocessing import Pool
import multiprocessing

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import skewnorm
from tqdm import tqdm
# Local application/library specific imports
import nkpack as nk
from switch import Organization, Stake


# Simulatin setups load
simulationDataSetup = pd.read_excel("simulations.xlsx")
print(simulationDataSetup)

listOfSimulations = [simulationCounter for simulationCounter in range(len(simulationDataSetup))]
simulationIds = []

def my_function(simulationCounter):
    # Modeler setup
    recip = "off"  # Reciprocity
    T = 100  # Time steps to observe in an organization
    MC = 100  # Number of simulation runs
    cnf = 1.96  # Confidence level
    RHO = 0  # Correlation between strategy landscape and the environment landscape
   
    # Simualted annealing
    annealing = 'off' # Switch for simulated annealing (Organization)

    # Time setup
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    today = date.today()
    
    # Retrieve current simulation ID and append to list
    currentID = simulationDataSetup.iloc[simulationCounter].at["ID"]
    simulationIds.append(currentID)
    
    # Landscapes setup
    P, N, S = 2, 8, 1  # Landscapes, bits per landscape, number of coupled peers
    K = int(simulationDataSetup.iloc[simulationCounter].at["K"])  # Internal interdependencies
    C = int(simulationDataSetup.iloc[simulationCounter].at["C"])  # Coupled bits
    
    # Stakeholders setup
    eff, ham = 1, 2  # Level of effort, hamming distance (While exploring the strategies around the current strategy)
    Num = int(simulationDataSetup.iloc[simulationCounter].at["NumStake"])  # Number of stakeholders
    Efforts = [eff] * Num # The replication of effort (initial) level for all stakeholders
    skew = simulationDataSetup.iloc[simulationCounter].at["Skewness"]  # Skewness of correlation (self-interest) distribution
    
    # Stakeholders' errors
    stake_err_env = float(simulationDataSetup.iloc[simulationCounter].at["err_env_stk"])  # Error about environment location
    S_e_bs = [stake_err_env] * Num  # Replication of errors about environment
    stake_err_per = float(simulationDataSetup.iloc[simulationCounter].at["err_per_stk"])  # Error about performance of strategy
    S_e_ps = [stake_err_per] * Num  # Replication of errors about performance
    Vote_err = stake_err_per # Error about performance of strategy while voting (It is the same as the error about performance in idea generation phase)
    react = 'on'  # Stakeholders' adaptive behavior (If stakeholder show reaction to selected strategy)
    voting_activate = 'off'  # Condition for stakeholders'participation in voting (second phase) when they havent proposed strategy in idea generation pahse
    
    # Shocks (Sudden changes in the landscapes)
    shock = simulationDataSetup.iloc[simulationCounter].at["Shock"]
    sw_shock = "off"  # Switch for shocks
    time_shock = simulationDataSetup.iloc[simulationCounter].at["ShockTime"] # The time step in which shock happens

    # Implementation phase
    sat_implement = "off"  # Consideration for stakeholders' overall satisfaction in success rate of implementing selected strategy
    # Log implementation settings
    print("sat implement=", sat_implement)
    
    # Environment settings
    environment_change_rate = float(simulationDataSetup.iloc[simulationCounter].at["EnvCngRate"] / 100)
    print('environment_change_rate: ', environment_change_rate) # Log environment change level
    
    # Make a comment to briefly describe setups
    comment = f'K={K}C={C}Skew={skew}Num={Num}Err={stake_err_per}Rate={environment_change_rate}'
    env_meth = "random"  # Method for environment changes
    
    # Organization's errors
    Crf_err = float(simulationDataSetup.iloc[simulationCounter].at["err_per_org"])  # Error in evaluating performance
    Env_err = float(simulationDataSetup.iloc[simulationCounter].at["err_env_org"])  # Error in environment understanding
    closed_per_err = Crf_err  # Error in closed performance evaluation

    
    
    environment_positions = [] # A list of MC number of random initial positions for Environment  (MC is the number of simulations)
    for i in range(MC):
        environment = np.random.choice(2, N)
        environment_positions.append(environment)

        
    organization_positions = [] # A list of random initial positions for Organization and stakeholders (MC is the number of simulations)
    for j in range(MC):
        pos = np.random.choice(2, N)
        organization_positions.append(pos)

    Correlations = [] # A list of correlation values for generating stakholders' landscapes (according to the skwness value)
    for _ in range(MC):
        random_numbers = skewnorm.rvs(skew, size=int(Num) + 2)
        random_numbers = 2 * ((random_numbers - np.min(random_numbers)) /
                                (np.max(random_numbers) - np.min(random_numbers))) - 1

        random_numbers_filtered = random_numbers[(random_numbers != 1) & (random_numbers != -1)]
        if len(random_numbers_filtered) != Num:
            for _ in range(Num - len(random_numbers_filtered)):
                random_numbers_filtered.append(0)
        Correlations.append(random_numbers_filtered.tolist())
        
      
       
    def single_iteration(Env_iteration,Org_pos_iteration,correlations): # One complete simulation for T time steps
        
        # Define different organizations (every one starts from a same position and environment is same for all of them)
        firm1 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name='firm1')        
        firm2 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm2')
        firm3 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm3')
        firm4 = Organization(p=P, n=N, k=K, c=C, s=S, t=T, rho=RHO, num=Num, env=Env_iteration, org_pos=Org_pos_iteration, correlations = correlations, s_e_bs = S_e_bs ,s_e_ps=S_e_ps, efforts=Efforts, name = 'firm4')                 
        
        
        # Reciprocal behavior
        firm1.net = recip
        firm2.net = recip
        firm3.net = recip
        firm4.net = recip

        
        # define landscapes for one of them and copy it for the rest
        firm1.define_tasks() 
        firm2.nature = firm1.nature
        firm3.nature = firm1.nature 
        firm4.nature = firm1.nature
        
        #Time steps
        firm1.T = T 
        firm2.T = T
        firm3.T = T
        firm4.T = T
        
        # Remove stakeholders who havent proposed
        firm1.voting_activate = voting_activate
        firm2.voting_activate = voting_activate
        firm3.voting_activate = voting_activate
        firm4.voting_activate = voting_activate
        
        # Switching on/off shocks
        if sw_shock == "on": 
             firm1.time_shock = time_shock
             firm2.time_shock = time_shock
             firm3.time_shock = time_shock
             firm4.time_shock = time_shock
             
             # Execution of shocks
             firm1.shock(shock)
             firm2.shocked_nature = firm1.shocked_nature
             firm3.shocked_nature = firm1.shocked_nature
             firm4.shocked_nature = firm1.shocked_nature
       
        else: # Skips the shock if sw is off
            firm1.time_shock = T+50
            firm2.time_shock = T+50
            firm3.time_shock = T+50
            firm4.time_shock = T+50
       
        # Including stakeholders
        firm1.hire_stakeholders()
        firm2.hire_stakeholders()
        firm3.hire_stakeholders()
        firm4.hire_stakeholders()
        
        # Hamming distance for exploration of new strategies in the neighborhood (Idea generation phase)
        firm1.ham = ham
        firm2.ham = ham
        firm3.ham = ham
        firm4.ham = ham
        
        # Annealing simulation status
        firm1.annealing = annealing
        firm2.annealing = annealing
        firm3.annealing = annealing
        firm4.annealing = annealing

        # time represents time_step
        for time in range(1,T):  
            
            #Phase1: Idea generation
            firm1.idea_generation(time,ham)
            firm2.idea_generation(time,ham)
            firm3.idea_generation(time,ham)
            firm4.idea_generation(time,ham)
            
            # Track how many participants have participated in first phase
            firm1.track_the_number()
            firm2.track_the_number()
            firm3.track_the_number()
            firm4.track_the_number()

            # Phase2: Strategy selection
            org_craft_err = 0 # organization's Error while choosing strategies in the second phase
            
            # Allocate strategy methodology to the organiations (Open / Closed)
            # firm1.openFirstPhase1(Crf_err,time)
            
            firm1.openFirstPhase(Crf_err,time) # Method for firm 1 (First phase is open).
            firm2.openSecondPhaseTideman(Vote_err,time)# Method for firm 2 (First & second phases are open).
            firm3.openSecondPhaseTidemanF(Vote_err,time)# Method for firm 3 (First phase and second phases (voting on short listed proposals) are open).
            firm4. closed(0,0,1,2,time)# Method for firm 4 closed strategy making.
            
            # phase3: strategy implemetation
            if sat_implement == "on": 
                firm1.implement_sat(time)
                firm2.implement_sat(time)
                firm3.implement_sat(time)
                firm4.implement_sat(time)

            elif sat_implement == "off":
                firm1.implementation(time)
                firm2.implementation(time)
                firm3.implementation(time)
                firm4.implementation(time)
             

            
            # Stakeholders' reaction (Stakeholders' adaptive behavior)
            if react == 'on':
                firm1.stake_feedback(1, time)
                firm2.stake_feedback(1, time)
                firm3.stake_feedback(1, time)
                firm4.stake_feedback(1, time)

            # Environment change (Environment is the same for all of the organisations)
            speed = 1 # Number of flipped bits in the case the environment changes
            firm1.environment_change_random(speed,environment_change_rate,time)
            firm2.env = firm1.env
            firm3.env = firm1.env
            firm4.env = firm1.env
     
                
        #Elicit results for firm 1
        
        bit_states1 = []
        for dec_state in firm1.strategy_line[0, :]:
            bit_states1.append(firm1.dec_to_bin(dec_state))
        performances1 = []
        t1 = 0
        for bit_state in bit_states1:
            if t1 < firm1.time_shock:
                performance1 = firm1.nature.phi(None, bit_state)[0]
            else:
                performance1 = firm1.shocked_nature.phi_shock(None, bit_state)[0]
            performances1.append(performance1)
            t1 += 1
        
        # Elicit results for firm 2
        bit_states2 = []
        for dec_state in firm2.strategy_line[0, :]:
            bit_states2.append(firm2.dec_to_bin(dec_state))
        performances2 = []
        t2 = 0
        for bit_state in bit_states2:
            if t2 < firm2.time_shock:
                performance2 = firm2.nature.phi(None, bit_state)[0]
            else:
                performance2 = firm2.shocked_nature.phi_shock(None, bit_state)[0]
            performances2.append(performance2)
            t2 += 1
            
        # Elicit results for firm 3
        bit_states3 = []
        for dec_state in firm3.strategy_line[0, :]:
            bit_states3.append(firm3.dec_to_bin(dec_state))
        performances3 = []
        t3 = 0
        for bit_state in bit_states3:
            if t3 < firm3.time_shock:
                performance3 = firm3.nature.phi(None, bit_state)[0]
            else:
                performance3 = firm3.shocked_nature.phi_shock(None, bit_state)[0]
            performances3.append(performance3)
            t3 += 1


        # Elicit results for firm 4
        bit_states4 = []
        for dec_state in firm4.strategy_line[0, :]:
            bit_states4.append(firm4.dec_to_bin(dec_state))
        performances4 = []
        t4 = 0
        for bit_state in bit_states4:
            if t4 < firm4.time_shock:
                performance4 = firm4.nature.phi(None, bit_state)[0]
            else:
                performance4 = firm4.shocked_nature.phi_shock(None, bit_state)[0]
            performances4.append(performance4)
            t4 += 1



        # Average of performance for all satkeholders
        sat1 = firm1.satisfaction()
        sat2 = firm2.satisfaction()
        sat3 = firm3.satisfaction()
        sat4 = firm4.satisfaction()
        
        # Receive the number of participants as a list
        number_of_participants1 = firm1.number_of_participants
        number_of_participants2 = firm2.number_of_participants
        number_of_participants3 = firm3.number_of_participants
        number_of_participants4 = firm4.number_of_participants
        
        # Return Performnace, Satisfaction, and the number of participants
        return performances1, performances2, performances3, performances4, sat1, sat2, sat3, sat4, number_of_participants1, number_of_participants2, number_of_participants3, number_of_participants4

            
    # Lists for keeping the outcome for all simulations

    results1, results2, results3, results4  = [], [] ,[] ,[] # Keep the performance
    SAT1, SAT2, SAT3, SAT4 = [], [], [], [] # Keep the average of performance for all the stakeholders
    number1, number2, number3, number4 = [], [], [], [] # Keep the number of participants
    
    # Run model for MC times
    for mc in range(MC):
         result1, result2, result3, result4, sat1, sat2, sat3, sat4, n1, n2, n3, n4 = single_iteration(environment_positions[mc], organization_positions[mc], Correlations[mc])


         results1.append(result1)
         results2.append(result2)
         results3.append(result3)
         results4.append(result4)
                 
         SAT1.append(sat1)
         SAT2.append(sat2)
         SAT3.append(sat3)
         SAT4.append(sat4)
         
         number1.append(n1)
         number2.append(n2)
         number3.append(n3)
         number4.append(n4)
         
         percentage = int((mc + 1) / MC * 100)
         # Print only unique percentages to avoid clutter
         if mc == 0 or percentage > int(mc / MC * 100):
            print(f"Progress: {percentage}%")

    # Convert the results into numpy arrays (Performance) 
    arr_results1 = np.array(results1)
    arr_results2 = np.array(results2)
    arr_results3 = np.array(results3)
    arr_results4 = np.array(results4)
    
    # Convert the results into numpy arrays (The average of performance for all the stakeholders)
    arr_SAT1 = np.array(SAT1)
    arr_SAT2 = np.array(SAT2)
    arr_SAT3 = np.array(SAT3)
    arr_SAT4 = np.array(SAT4)
    
    # Convert the results into numpy arrays (The number of participants in each time step)    
    arr_num1 = np.array(number1)
    arr_num2 = np.array(number2)
    arr_num3 = np.array(number3)
    arr_num4 = np.array(number4)
    
    
    # Reshape the arrays (Each raw represents performances for one simulation in T time steps)
    output1 = np.reshape(arr_results1,(MC,T))
    output2 = np.reshape(arr_results2,(MC,T))
    output3 = np.reshape(arr_results3,(MC,T))
    output4 = np.reshape(arr_results4,(MC,T))
    # Calculate the average of performance of all simulations in each time step.
    average_performance1 = np.mean(output1,0)
    average_performance2 = np.mean(output2,0)
    average_performance3 = np.mean(output3,0)
    average_performance4 = np.mean(output4,0)
    # Calculate the confidence level for performnace
    s1 = np.std(output1,0)
    confidence_value1 = cnf*(s1/math.sqrt(MC))
    
    s2 = np.std(output2,0)
    confidence_value2 = cnf*(s2/math.sqrt(MC))
    
    s3 = np.std(output3,0)
    confidence_value3 = cnf*(s3/math.sqrt(MC))

    s4 = np.std(output4,0)
    confidence_value4 = cnf*(s4/math.sqrt(MC))

    # Convert the results into numpy arrays (Satisfaction) 

    outputSAT1 = np.reshape(arr_SAT1,(MC,T))
    average_sat1 = np.mean(outputSAT1,0)

    t10 = np.std(outputSAT1,0)
    confidence_value10 = cnf*(t10/math.sqrt(MC))
    #

    outputSAT2 = np.reshape(arr_SAT2,(MC,T))
    average_sat2 = np.mean(outputSAT2,0)

    t20 = np.std(outputSAT2,0)
    confidence_value20 = cnf*(t20/math.sqrt(MC))
    
    #

    outputSAT3 = np.reshape(arr_SAT3,(MC,T))
    average_sat3 = np.mean(outputSAT3,0)

    t30 = np.std(outputSAT3,0)
    confidence_value30 = cnf*(t30/math.sqrt(MC))
    
    #

    outputSAT4 = np.reshape(arr_SAT4,(MC,T))
    average_sat4 = np.mean(outputSAT4,0)

    t40 = np.std(outputSAT4,0)
    confidence_value40 = cnf*(t40/math.sqrt(MC))
    
    # Convert the results into numpy arrays (Number of participants)
    output_num1 = np.reshape(arr_num1,(MC,T))
    average_num1 = np.mean(output_num1,0)

    t100 = np.std(average_num1,0)
    confidence_value100 = cnf*(t100/math.sqrt(MC))
    #

    output_num2 = np.reshape(arr_num2,(MC,T))
    average_num2 = np.mean(output_num2,0)

    t200 = np.std(average_num2,0)
    confidence_value200 = cnf*(t200/math.sqrt(MC))
    #
    output_num3 = np.reshape(arr_num3,(MC,T))
    average_num3 = np.mean(output_num3,0)

    t300 = np.std(average_num3,0)
    confidence_value300 = cnf*(t300/math.sqrt(MC))
    
    output_num4 = np.reshape(arr_num4,(MC,T))
    average_num4 = np.mean(output_num4,0)

    t400 = np.std(average_num4,0)
    confidence_value400 = cnf*(t400/math.sqrt(MC))
    
    # Capture time
    today_str = str(today)
    current_time_str = str(current_time)
    current_time_str = current_time_str.replace(":", "&")
    
    # Path to save
    parent_dir = r"C:\Users\sabanihashen\Desktop\TEST"
    comment = f'K={K}C={C}Skew={skew}Num={Num}Err={stake_err_per}Rate={environment_change_rate}'
    path = os.path.join(parent_dir, comment, today_str, current_time_str)
    os.makedirs(path, exist_ok=True)
    
    # Saving files using proper path formatting
    np.save(os.path.join(path, '1_per.npy'), output1)
    np.save(os.path.join(path, '2_per.npy'), output2)
    np.save(os.path.join(path, '2f_per.npy'), output3)
    np.save(os.path.join(path, 'close_per.npy'), output4)
    
    np.save(os.path.join(path, '1_num.npy'), output_num1)
    np.save(os.path.join(path, '2_num.npy'), output_num2)
    np.save(os.path.join(path, '2f_num.npy'), output_num3)
    np.save(os.path.join(path, 'close_num.npy'), output_num4)
    
    # Set a consistent style for clarity
    plt.style.use('classic') 

    
# Font settings suitable for APA
    font = {'family': 'Times New Roman',
            'color':  'black',
            'weight': 'normal',
            'size': 12,
            }

    time = [t for t in range(0, T)]

    # Create a function for plotting to reduce repetitive code
    def plot_data(time, avg_performance, confidence_value, color, linestyle, label, marker, marker_spacing):
        plt.plot(time, avg_performance, color=color, linestyle=linestyle, label=label, marker=marker, markersize=5, markevery=marker_spacing)
        plt.fill_between(time, avg_performance - confidence_value, avg_performance + confidence_value, alpha=0.2, color=color)
    comment = f'K={K}C={C}Skew={skew}Num={Num}Err={stake_err_per} rate={environment_change_rate}'
    # Plot data using different colors, line styles, and markers for differentiation
    plt.figure()
    plot_data(time, average_performance1, confidence_value1, "blue", '-', "Open Strategy (First pahse is open)", "o", 10)
    plot_data(time, average_performance2, confidence_value2, "red", '--', "Open Strategy (Second phase is open)", "s", 10)
    plot_data(time, average_performance3, confidence_value3, "green", '-.', "Optimization based on number of participants", "^", 10)
    plot_data(time, average_performance4, confidence_value4, "black", ':', "Optimization based on performance", "d", 10)
    
    # Customize labels and legend
    font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 12}
    plt.xlabel("Time", fontdict=font)
    plt.ylabel("Performance", fontdict=font)
    plt.legend(loc="lower right", fontsize=10)
    plt.ylim(.45, .9)
    # Customize grid
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Add a title
    # Add a title
    plt.title(f'{comment}', fontsize=14)
    
    # Customize tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # Show the plot
    plt.savefig(f'{path}\\{comment}.pdf', dpi=300, bbox_inches="tight")
    plt.savefig(f'{path}\\{comment}.jpg', dpi=300)
    
    #######
    #Plot the number of stakeholders
    #######

    comment = f'K={K}C={C}Skew={skew}Num={Num}Err={stake_err_per} rate={environment_change_rate} Number'
    # Plot data using different colors, line styles, and markers for differentiation
    plt.figure()
    plot_data(time, average_num1, confidence_value100, "blue", "-","Open Strategy (First Phase is open)", "o", 50)
    plot_data(time, average_num2, confidence_value200, "red","--","Open Strategy (Second phase is open)", "s", 50)
    plot_data(time, average_num3, confidence_value300, "green", "-","Open Strategy (Second phase is open filtered)", "^", 50)
    plot_data(time, average_num4, confidence_value400, "black", "-","Close", "o", 50)
    # Customize labels and legend
    font = {'family': 'serif', 'color':  'darkred', 'weight': 'normal', 'size': 12}
    plt.xlabel("Time", fontdict=font)
    plt.ylabel("Number", fontdict=font)
    plt.ylim(0, 40)
    plt.legend(loc="upper right", fontsize=10)

    # Customize grid
    plt.grid(True, linestyle="--", alpha=0.5)

    # Add a title
    plt.title(f'{comment}', fontsize=14)

    # Customize tick labels
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig(f'{path}\\{comment}Number.pdf', dpi=300, bbox_inches="tight")
    plt.savefig(f'{path}\\{comment}Number.jpg', dpi=300)
    
    
    return





# Creating a list of simulation counters
listOfSimulations = [simulationCounter for simulationCounter in range(len(simulationDataSetup))]
simulationIds = []

if __name__ == '__main__':
    # Create a pool of workers
    with multiprocessing.Pool() as pool:
        # Call my_function with multiple arguments in parallel and show progress bar
        results = list(tqdm(pool.imap(my_function, listOfSimulations), total=len(listOfSimulations)))
