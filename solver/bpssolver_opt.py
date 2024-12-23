import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd
import numpy as np
import time
import csv
import torch
import random
import coptpy as cp
from coptpy import COPT
import time
import pyscipopt
import matplotlib.pyplot as plt

# if not os.path.exists('Data_Train'):
#     os.makedirs('Data_Train')

#parameters
units = ['heater1','heater2','reactor1', 'reactor2', 'reactor3', 'reactor4', 'still1', 'still2']
tasks = ['task0', 'task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7', 'task8', 'task9', 'task10', 'task11', 'task12', 'task13', 'task14', 'task15']
#task0-heating in heater1, task1-heating in heater2, task2-reaction1 in reactor1, task3-reaction1 in reactor2, task4-reaction1 in reactor3, task5-reacion1 in reactor4,
#task6-reaction2 in reactor1, task7-reaction2 in reactor2, task8-reaction2 in reactor3, task9-reaction2 in reactor4,
#task10-reaction3 in reactor1, task11-reaction3 in reactor2, task12-reaction3 in reactor3, task13-reaction3 in reactor4,
#task14-separation in still1, task15-separation in still2
states = ['feedA','feedB','feedC','hotA','IntAB','IntBC','impureE','Prod1','Prod2']
events = ['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8']

capacity = {'heater1': 100, 'heater2': 120, 'reactor1': 70, 'reactor2': 80, 'reactor3': 70, 'reactor4': 120, 'still1': 200, 'still2': 150}

suitability = {('heater1','task0'):1, ('heater2','task1'): 1, ('reactor1','task2'): 1,('reactor1','task6'): 1,('reactor1','task10'): 1,
               ('reactor2','task3'): 1, ('reactor2','task7'): 1, ('reactor2','task11'): 1, ('reactor3','task4'): 1,
               ('reactor3','task8'): 1,('reactor3','task12'): 1,('reactor4','task5'): 1,('reactor4','task9'): 1,
               ('reactor4','task13'): 1, ('still1','task14'): 1, ('still2','task15'): 1
              }

#time: to be updated

proportions_consumed = {('feedA','task0'): -1, ('feedA','task1'): -1,
                        ('hotA','task6'): -0.4, ('hotA','task7'): -0.4, ('hotA','task8'): -0.4, ('hotA','task9'): -0.4, 
                        ('IntBC','task6'): -0.6, ('IntBC','task7'): -0.6,('IntBC','task8'): -0.6,('IntBC','task9'): -0.6,
                        ('feedB','task2'): -0.5, ('feedB','task3'): -0.5, ('feedB','task4'): -0.5,('feedB','task5'): -0.5,
                        ('feedC','task2'): -0.5,('feedC','task3'): -0.5,('feedC','task4'): -0.5,('feedC','task5'): -0.5,
                        ('feedC','task10'): -0.2, ('feedC','task11'): -0.2,('feedC','task12'): -0.2,('feedC','task13'): -0.2,
                        ('IntAB','task10'): -0.8,('IntAB','task11'): -0.8,('IntAB','task12'): -0.8,('IntAB','task13'): -0.8,
                        ('impureE','task14'): -1, ('impureE', 'task15'): -1}
proportions_produced = {('hotA','task0'): 1, ('hotA','task1'): 1, ('Prod1','task6'): 0.4, ('Prod1','task7'): 0.4, 
                        ('Prod1','task8'): 0.4, ('Prod1','task9'): 0.4,
                        ('IntAB','task6'): 0.6, ('IntAB','task7'): 0.6,('IntAB','task8'): 0.6,('IntAB','task9'): 0.6,
                        ('IntBC','task2'): 1, ('IntBC','task3'): 1,('IntBC','task4'): 1,('IntBC','task5'): 1,
                        ('impureE','task10'): 1,('impureE','task11'): 1,('impureE','task12'): 1,('impureE','task13'): 1,
                        ('Prod2','task14'): 0.9, ('IntAB','task14'): 0.05, ('Prod2','task15'): 0.9, ('IntAB', 'task15'): 0.05}

storage_capacity = {'feedA': 10000, 'feedB': 10000, 'feedC': 10000,'hotA': 100, 'IntAB': 200, 'IntBC': 150, 'impureE': 200, 'Prod1': 10000, 'Prod2': 10000}
initial_amount = {'feedA': 1000, 'feedB': 800, 'feedC': 800, 'hotA': 0, 'IntAB': 0, 'IntBC': 0, 'impureE': 0, 'Prod1': 0, 'Prod2': 0}
price = {'feedA': 0, 'feedB':0, 'feedC': 0, 'hotA': 0, 'IntAB': 0, 'IntBC': 0, 'impureE': 0, 'Prod1': 25, 'Prod2': 30}
H = 48

for i in tasks:
    for s in states:
        if (s, i) in proportions_consumed:
            continue
        else:
            proportions_consumed[s, i] = 0

for i in tasks:
    for s in states:
        if (s, i) in proportions_produced:
            continue
        else:
            proportions_produced[s, i] = 0

def Solver(t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, rnd):
    # env = gp.Env(logfilename = f"result/log/gurobi_{rnd}_opt.log")
    env = gp.Env()
    env.setParam("IGNORENAMES", 1)
    
    model = gp.Model("newmodel", env)
    
    time = {
        'task0': t0,
        'task1': t1,
        'task2': t2,
        'task3': t3,
        'task4': t4,
        'task5': t5,
        'task6': t6,
        'task7': t7,
        'task8': t8,
        'task9': t9,
        'task10': t10,
        'task11': t11,
        'task12': t12,
        'task13': t13,
        'task14': t14,
        'task15': t15
    }
    
    #variables
    #the beginning of task i at event point n; binary variable
    make = model.addVars(tasks, events, vtype = GRB.BINARY, name = 'Make')

    #the utilization of unit j at event point n; binary variable
    utilization = model.addVars(units, events, vtype = GRB.BINARY, name = 'Utilization')

    #the amount of material undertaking task i in unit j at event point n
    processing = model.addVars(tasks, units, events, lb = 0, name = 'Processing')

    #the amount of state s being delivered to the market at event point n
    sold = model.addVars(states, events, lb = 0, name = 'Sold')

    #the amount of state s at event point n
    state_amount = model.addVars(states, events, lb = 0, name = 'State_amount')
    
    #time that task i starts in unit j at event point n
    t_start = model.addVars(tasks, units, events, lb = 0, name = 'T_start')

    #time that task i finishes in unit j while it starts at event point n
    t_finish = model.addVars(tasks, units, events, lb = 0, name = 'T_finish')
    
    #supplementary variable
    h = model.addVars(20, 1, vtype = GRB.CONTINUOUS, name = 'Supplementary_variable')
    for i in range(0, 20):
        h[i, 0].start = np.random.normal(0, 20)
    
    #print(h)
    
    #add constraints
    
    #initial constraints
    #initial amount
    model.addConstrs((state_amount[s, 'e0'] == initial_amount[s] for s in states), name = 'Initial Amount')
    #add other constraints
    #allocation
    model.addConstrs((gp.quicksum(make[i, n] for i in tasks if suitability.get((j, i), 0) == 1) == utilization[j, n] for j in units for n in events), name = 'Allocation')
    #capacity
    model.addConstrs((processing[i, j, n] <= capacity[j] * make[i, n] for i in tasks for j in units if suitability.get((j,i),0) == 1 for n in events), name = 'Capacity')
    #initial processing
    model.addConstrs((processing[i, j, n]  <= state_amount[s, n] for i in tasks for j in units if suitability.get((j, i), 0) == 1 for s in states if proportions_consumed[s, i] < 0 for n in events), name = 'Processing Constraints')
    #storage
    model.addConstrs((state_amount[s, n] <= storage_capacity[s] for s in states for n in events), name = 'Storage')
    #material balance
    model.addConstrs((state_amount[s, events[k]] == state_amount[s, events[k-1]] - sold[s, events[k]] + 
                      gp.quicksum(proportions_produced[s, i] * gp.quicksum(processing[i, j, events[k-1]] for j in units if suitability.get((j, i), 0) == 1) for i in tasks) +
                      gp.quicksum(proportions_consumed[s, i] * gp.quicksum(processing[i, j, events[k-1]] for j in units if suitability.get((j, i), 0) == 1) for i in tasks) for s in states for k in range(1, 9)), 
                      name = 'Material Balance')
    #sold
    model.addConstrs((sold[s, n] <= state_amount[s, n] for s in states for n in events), name = 'Sold')
    #duration
    model.addConstrs((t_finish[i, j, n] == t_start[i, j, n] + 0.67 * time [i] * make[i, n] + 0.67 * time[i] / capacity[j] * processing[i, j, n]
                      for i in tasks for j in units if suitability.get((j,i),0) == 1 for n in events), name = 'Duration')
    #sequence - same task in the same unit
    model.addConstrs((t_start[i, j, events[k+1]] >= t_finish[i, j, events[k]] - H * (2 - make[i, events[k]] - utilization[j, events[k]]) for i in tasks for j in units if suitability.get((j,i),0) == 1 for k in range(0,8)), name = 'Sequence_same')
    model.addConstrs((t_start[i, j, events[k+1]] >= t_start[i, j, events[k]] for i in tasks for j in units if suitability.get((j,i),0) == 1 for k in range(0,8)), name = 'Sequence_same_1')
    model.addConstrs((t_finish[i, j, events[k+1]] >= t_finish[i, j, events[k]] for i in tasks for j in units if suitability.get((j,i),0) == 1 for k in range(0,8)), name = 'Sequence_same_2')
    #sequence - different tasks in the same unit
    model.addConstrs((t_start[i, j, events[k+1]] >= t_finish[i_1, j, events[k]] - H * (2 - make[i_1, events[k]] - utilization[j, events[k]]) for j in units for i in tasks if suitability.get((j,i),0) == 1 
                      for i_1 in tasks if suitability.get((j,i_1),0) == 1 and i != i_1 for k in range(0,8)), name = 'Sequence_diff')
    #sequence - different tasks in different units
    model.addConstrs((t_start[i, j, events[k+1]] >= t_finish[i_1, j_1, events[k]] - H * (2 - make[i_1, events[k]] - utilization[j_1, events[k]]) for i in tasks for i_1 in tasks for j in units for j_1 in units 
                      if suitability.get((j,i),0) == 1 and suitability.get((j_1,i_1),0) == 1 and i != i_1 for k in range(0,6)), name = 'Sequence_diffdiff')
    #sequence - completion of previous tasks
    model.addConstrs((t_start[i, j, events[k+1]] >= gp.quicksum(gp.quicksum(t_finish[i_1, j, events[t]] - t_start[i_1, j, events[t]] 
                                                                            for i_1 in tasks if suitability.get((j,i_1),0) == 1) for t in range(0, k+1)) for i in tasks for j in units if suitability.get((j,i),0) == 1 for k in range(0,8)), name = 'Sequence_previous')
    #time horizon
    model.addConstrs((t_finish[i, j, n] <= H for i in tasks for j in units if suitability.get((j,i),0) == 1 for n in events), name = 'Time Horizon_finish')
    model.addConstrs((t_start[i, j, n] <= H for i in tasks for j in units if suitability.get((j,i),0) == 1 for n in events), name = 'Time Horizon_start')
    model.addConstrs((make[i, n] <= gp.quicksum(processing[i, j, n] for j in units if suitability.get((j, i),0) == 1) for i in tasks for n in events), name = 'making')
    
    #additional constraints
    
    W = torch.load('Model/216to20_W_default.pt')
    a = torch.load('Model/216to20_a_default.pt')
    
    W = np.array(W.cpu().detach())
    a = np.array(a.cpu().detach()).reshape([-1, 1])
    W = W.T
    
    #print(W.shape)
    #print(a.shape)

    #model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - make[tasks[t], events[s]]) * (-541) for t in range(8, 16) for s in range(4, 9)), name = 'Constr1')
    #model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - utilization[units[t], events[s]]) * (-541) for t in range(4, 8) for s in range(4, 9)), name = 'Constr2')
    #model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) <= make[tasks[t], events[s]] * (460) for t in range(8, 16) for s in range(4, 9)), name = 'Constr3')
    #model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) <= utilization[units[t], events[s]] * (460) for t in range(4, 8) for s in range(4, 9)), name = 'Constr4')
    
    model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - make[tasks[t], events[s]]) * (-632) for t in range(0, 16) for s in range(0, 9)), name = 'Constr1')
    model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) >= (1 - utilization[units[t], events[s]]) * (-632) for t in range(0, 8) for s in range(0, 9)), name = 'Constr2')
    model.addConstrs((gp.quicksum(W[t * 9 + s, j] * h[j, 0] + a[j] for j in range(0, 10)) <= make[tasks[t], events[s]] * (520) for t in range(0, 16) for s in range(0, 9)), name = 'Constr3')
    model.addConstrs((gp.quicksum(W[t * 9 + s + 144, j] * h[j, 0] + a[j] for j in range(0, 10)) <= utilization[units[t], events[s]] * (520) for t in range(0, 8) for s in range(0, 9)), name = 'Constr4')
    
    #objective function
    obj = gp.quicksum(gp.quicksum(price[s] * sold[s, n] for s in states) for n in events)
    model.setObjective(obj, GRB.MAXIMIZE)
    
    model.optimize()
    #print(h)
    model.write(f"data/Data_Problem_5pt_op/{(rnd):0>4}.lp")
    #save the result
    # model.write(f'data/Data_Optimized_0925/{(rnd):0>4}.sol')
    
    return model.ObjVal, model.runtime

    #plot
    '''fig, ax = plt.subplots()
    ax.set_xlim(0, 48)
    ax.set_ylim(0, 9)
    labels = ['heater1', 'heater2','reactor1','reactor2','reactor3','reactor4','separ1', 'separ2']
    y = [1, 2, 3, 4, 5, 6, 7, 8]
    plt.yticks(y, labels)

    cnt = 0
    colors = ['tab:orange','tab:green','tab:red','tab:blue','tab:pink', 'tab:purple', 'tab:orange', 'tab:green', 'tab:red']

    for j in units:
        for i in tasks:
            if suitability.get((j, i), 0) == 1:
                for k in range(0,9):
                    ts = t_start[i, j, events[k]].x
                    tf = t_finish[i, j, events[k]].x - ts
                    ax.broken_barh([(ts, tf)], (0.7 + cnt, 0.6), facecolors = colors[k])
        cnt += 1
    
    #plt.savefig(f'Data_Train/{(rnd):0>4}.png')'''
def solve_else(solver, mpspath, logflag = 0, saveflag = 0, mipgap = 0.0, thread = 1):
    """ set Param """
    timelimit = 60000
    mipgap = mipgap
    poolgap = 0
    # self.model.write(f"BPS_data/temp/{self.title}.mps")
    
    """ load and solve model """
    if solver == "copt":
        env = cp.Envr()
        model = env.createModel()
        model.read(mpspath)
        model.setParam(COPT.Param.TimeLimit, timelimit)
        model.setParam(COPT.Param.RelGap, mipgap)
        model.setParam(COPT.Param.TuneOutputLevel, 3)
        model.setLogFile(f"result/log/{solver}_{mpspath.split('.')[0].split('/')[-1]}_opt.log")
        # model.setParam(COPT.Param.Threads, thread)
        if logflag:
            model.setParam(COPT.Param.Logging, 1)
        else:
            model.setParam(COPT.Param.Logging, 0)
        model.solve()

    if solver == "scip":
        model = pyscipopt.Model()
        model.readProblem(mpspath)
        model.setParam("limits/time", timelimit)
        model.setParam("limits/gap", mipgap)
        # model.setParam("parallel/maxnthreads", thread)
        if logflag:
            model.setParam("display/verblevel", 4)
        else:
            model.setParam("display/verblevel", 3)
        model.optimize()
    
    # os.remove(f"BPS_data/temp/{self.title}.mps")
    
    """ write sol return obj and time """
    if solver == "copt":
        # if saveflag:
            # self.save_sol(model, param = "std_copt")
        
        if model.getAttr("MipStatus") != COPT.OPTIMAL:
            return -1, model.getAttr("SolvingTime")
        else:
            return model.getAttr("BestObj"), model.getAttr("SolvingTime")
    if solver == "scip":
        # if saveflag:
            # model.writeBestSol(f'aed_test/bps/sol/{self.title}_std_scip.sol')
        if model.getGap() >= (mipgap + 1e-6):
            return -1, model.getSolvingTime()
        else:
            return model.getObjVal(), model.getSolvingTime()

if __name__ == "__main__":

    for i in range(1000, 1200):
        time_record = []
        opt_record = []
        initial_data_time = np.load(f'data/Data_Generated_5pt/{(i):0>4}.npz')
        parameters = []
        
        for key, arr in initial_data_time.items():
            parameters.append(arr[0])
        
        # s = time.time()
        o, t = Solver(parameters[0], parameters[1], parameters[2], parameters[3], parameters[4], parameters[5], parameters[6], parameters[7], parameters[8], parameters[9], parameters[10], parameters[11], parameters[12], parameters[13], parameters[14], parameters[15], i)
        """ other solver """
        # o, t = solve_else("copt", f'data/Data_Problem_0925_op/{(i):0>4}.lp')
        time_record.append(t)
        opt_record.append(o)
        
        with open('result/value_optimized_0912_20_gurobi.csv', mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow(opt_record)
        with open('result/time_optimized_0912_20_gurobi.csv', mode='a', newline='') as file:
           writer = csv.writer(file)
           writer.writerow(time_record)