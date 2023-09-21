import sys
sys.path.append('C:\\Users\oscar\OneDrive\Dokumenter\Høst 2023\TET4565 Spesialiseringsemne\Hydro_optimization') #OSCAR path
sys.path.append('C:\\Users\\benny\\Documents\\Hydro_optimization')  #BENJAMIN path
sys.path.append('C:\\Users\\Epsilon Delta\\OneDrive - NTNU\\Energi og Miljø NTNU\\Høst2023\\TET4565 Fordypningsemne\\Hydro_optimization') #ESPEN path


import pyomo.environ as pyo
import numpy as np
from pyomo.environ import ConcreteModel,Set,RangeSet,Param,Suffix,Reals,NonNegativeReals,NonPositiveReals,Binary,Objective,minimize,maximize,value
from pyomo.core import Constraint,Var,Block,ConstraintList
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from calculations.datahandling import*
from calculations.data_processor import* 




#--Data handling--
#Read of parameters for portfolio
def InputData(data_file):
    inputdata = pd.read_excel(data_file)
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()
    data = {}
    data['hydro'] = inputdata[['Ci', 'yi', 'P_min', 'P_max', 'H_max']].drop('Solar')
    data['solar']=inputdata[['Ci', 'yi', 'P_min', 'P_max']].drop('Hydro1').drop('Hydro2')
    return data


#Solar production based on forecast 
def read_solar_data(irrad_file, spec_file, start_date, end_date):
    input_data_PV = read_excel_data(spec_file)
    input_data_Irr = read_irr_data(irrad_file)
    PV_power = pv_power_estimated(input_data_PV,input_data_Irr)
    Solar_1=convert_to_dict(PV_power, start_date, end_date, 'H')
    Solar_p=scale_dict(Solar_1, 10)
    return Solar_p

#Start and end dates of the optimization
start_date='2018-05-28 00:00'
end_date='2018-05-28 23:00'

#Original forecast for solar power production
Solar_p=read_solar_data('data/Data_solar_irr_NOR.csv', 'data/PV_spec.xlsx', start_date, end_date)

#Scenarios for solar forecast 
S_high=scale_dict(Solar_p, 1.5)
S_avg=scale_dict(Solar_p, 1)
S_low=scale_dict(Solar_p, 0.5)

#Load 
L= {1:30, 2:20, 3:20, 4:30, 5:50, 6:80, 7:50, 8:90, 9:110, 10:150, 11:120, 12:80, 13:70, 14:80, 15:90, 16:160, 17:170, 18:150, 19:120, 20:100, 21:70, 22:60, 23:50, 24:40} 

#--Constants--
Constants= {
    'Load_penalty':100, 
    'Hydro_cap':3000,
    'Scenarios':['S_high', 'S_avg', 'S_low'], 
    'probs':{'S_high':1/3, 'S_avg':1/3, 'S_low':1/3}     
}
#Production bounds for hydro plants in stage 1
def p_bounds(model,i,j):
    return (model.Pmin[i],model.Pmax[i])
#Production constraint for hydro1 in stage 2
def hydro1_bounds(model,s,j):
    return model.p_s1[s,j]<=40
#Production constraint for hydro2 in stage 2

def hydro2_bounds(model,s,j):
    return model.p_s2[s,j]<=100
#Solar production in stage 1 must be equal to original forecast
def Solar_rule(model,j):
    return  model.phi['Solar',j] == Solar_p[j]
#Solar production in stage 2 must be equal to scenario for high production
def Solar_high(model,j):
    if model.probs['S_high']>0:
        return model.phi_s['S_high',j]==S_high[j]
    else:
        return model.phi_s['S_high',j]==0
#Solar production in stage 2 must be equal to scenario for average production
def Solar_avg(model,j):
    if model.probs['S_avg']>0:
        return  model.phi_s['S_avg',j] == S_avg[j]
    else:
        return  model.phi_s['S_avg',j] == 0
#Solar production in stage 2 must be equal to scenario for low production
def Solar_low(model,j):
    if model.probs['S_low']>0:
        return  model.phi_s['S_low',j] == S_low[j]
    else:
        return  model.phi_s['S_low',j] == 0
    
#Sum of hydro production in stage 1 must be lower than available generation
def Hydro_firststage(model,i):
    return sum(model.p['Hydro1',j] for j in model.periods)<=model.Hmax['Hydro1']
    
#Sum of hydro production in stage 2 must be lower than capacity and the already used power in stage 1
def hydro1_scenario_bounds(model, s, j):
    return sum(model.p_s1[s, j] for j in model.periods )<= model.Hmax['Hydro1'] - sum(model.p['Hydro1', j] for j in model.periods)

# Production constraint for hydro2 in stage 2
def hydro2_scenario_bounds(model, s, j):
    return sum(model.p_s2[s, j] for j in model.periods) <= model.Hmax['Hydro2'] - sum(model.p['Hydro2', j] for j in model.periods)

#def Hydro_secondstage(model,s,i):
    #return sum(model.p_s1[s,j] +model.p_s2[s,j] for j in model.periods)<=model.Hmax[i]-sum(model.p['Hydro1',j] + model.p['Hydro2',j] for j in model.periods)
    
#Total power generation in stage 1 must be equal to the hourly set load
def load_rule_FirstStage(model, j):
    return model.p['Hydro1', j] + model.p['Hydro2', j] + model.phi['Solar', j] + model.L_p['Load_penalty', j] == L[j]
    
#Total power generation in stage 2 must be equal to the hourly set load
def load_rule_TwoStage(model, s, j):
    return model.p_s1[s,j] +model.p_s2[s,j]+ model.phi_s[s, j] + model.L_p_s[s, j] == L[j]



#Objective function 
def ObjRule(model):
    #Sum of production costs in stage 1 
    first_stage_obj = sum(model.yi[i] * model.p[i, j] for i in model.plants for j in model.periods) \
        + sum(model.ki[s] * model.phi[s, j] for s in model.solar for j in model.periods) \
        + sum(model.Li[n] * model.L_p[n, j] for n in model.penalty for j in model.periods)

    # Second stage objective (scenario-dependent), sum of all production costs
    second_stage_obj = sum(model.probs[l] * (
        model.yi[i]*(model.p_s1[l,j]+model.p_s2[l,j]) + model.ki[s] * model.phi_s[l, j] + model.Li[n] * model.L_p_s[l, j]
        ) for i in model.plants for s in model.solar for l in model.scenarios for n in model.penalty for j in model.periods)

    # Total objective is the sum of first and second stage objectives
    return first_stage_obj + second_stage_obj


def model_setup(Constants, data):
    #--Defining sets--
    model = pyo.ConcreteModel()
    #Defining the set of plants
    model.plants = pyo.Set(initialize=['Hydro1','Hydro2'])
    model.solar=pyo.Set(initialize=['Solar'])
    #Defining the load penalty
    model.penalty=pyo.Set(initialize=['Load_penalty']) 
    #Defining the set of periods
    periods=set()
    for i in range (1,25):
        periods.add(i) 
    model.periods = pyo.Set(initialize=periods)
    #Defining scenarios for two-stage problem 
    model.scenarios=pyo.Set(initialize=Constants['Scenarios'])

    #--Defining parameters--
    #Load penalty 
    model.Li=pyo.Param(model.penalty, initialize=Constants['Load_penalty'])
    #Initial costs for plants
    model.Ci=pyo.Param(model.plants, initialize=data['hydro']['Ci'])
    model.Si=pyo.Param(model.solar, initialize=data['solar']['Ci'])
    #Variable costs for plants 
    model.yi=pyo.Param(model.plants, initialize=data['hydro']['yi'])
    model.ki=pyo.Param(model.solar, initialize=data['solar']['yi'])
    #Production bounds for plants
    model.Pmin=pyo.Param(model.plants, initialize=data['hydro']['P_min'])
    model.Pmax=pyo.Param(model.plants, initialize=data['hydro']['P_max'])
    model.Phi_min=pyo.Param(model.solar, initialize=data['solar']['P_min'])
    model.Phi_max=pyo.Param(model.solar, initialize=data['solar']['P_max'])
    #probabilities for each scenario 
    model.probs=pyo.Param(model.scenarios, initialize=Constants['probs'])
    #Maximum hydro production available for the two plant for 24 hours
    model.Hmax=pyo.Param(model.plants, initialize=data['hydro']['H_max'])

    #--Defining variables and bounds-- 
    #Production from hydro plants stage 1
    model.p = pyo.Var(model.plants,model.periods, bounds=p_bounds)
    #Scenario and stage 2 based hydro power
    model.p_s1=pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)
    model.p_s2=pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)
    #Production from solar installation base case
    model.phi = pyo.Var(model.solar,model.periods, within=NonNegativeReals)
    #Production from solar installation scenario based
    model.phi_s=pyo.Var(model.scenarios,model.periods, within=NonNegativeReals)
    #Load penalty base case
    model.L_p = pyo.Var(model.penalty, model.periods, within=NonNegativeReals)
    #Load penalty during scenarios
    model.L_p_s=pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)

    #--Defining constraints-- all based on constraint functions defined above
    model.solar_cons = pyo.Constraint(model.periods, rule=Solar_rule)
    
    model.high_cons = pyo.Constraint(model.periods, rule=Solar_high)

    model.avg_cons = pyo.Constraint(model.periods, rule=Solar_avg)

    model.low_cons = pyo.Constraint(model.periods, rule=Solar_low)

    model.hydro1_cons=pyo.Constraint(model.scenarios, model.periods, rule=hydro1_bounds)

    model.hydro2_cons=pyo.Constraint(model.scenarios, model.periods, rule=hydro2_bounds)

    model.hydro_cons=pyo.Constraint(model.plants, rule=Hydro_firststage)

    #model.hydro_scenario_cons=pyo.Constraint(model.scenarios, model.plants, rule=Hydro_secondstage)
    model.hydro1_scenario_cons=pyo.Constraint(model.scenarios, model.periods, rule=hydro1_scenario_bounds)
    
    model.hydro2_scenario_cons=pyo.Constraint(model.scenarios, model.periods,  rule=hydro2_scenario_bounds)
    
    model.load_cons_FirstStage = pyo.Constraint(model.periods, rule=load_rule_FirstStage)

    model.load_cons_TwoStage = pyo.Constraint(model.scenarios, model.periods, rule=load_rule_TwoStage)
    
    #Objective rule 
    model.obj= pyo.Objective(rule=ObjRule, sense=pyo.minimize)
    return model
    
def solve(model):
    #solver 
    opt = SolverFactory('gurobi', solver_io="python") 
    #defining dual 
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model,tee=True) 
    return results, model

def displayresults(model):
    return print(model.display()) #model.dual.display())

def plotting(model):
    prod={}
    solar={}
    penalty={}
    penalty_scenarios={}    
    scenarios={}        
    hydro1_scenarios={}
    hydro2_scenarios={}    
    for i in model.plants:
        prod[i]=[value(model.p[i, j]) for j in model.periods]
    for s in model.solar:
        solar[s]=[value(model.phi[s, j]) for j in model.periods]
    for n in model.penalty:
        penalty[n]=[value(model.L_p[n,j]) for j in model.periods]
    for h in model.scenarios:
        hydro1_scenarios[h]=[value(model.p_s1[h,j]) for j in model.periods]
        hydro2_scenarios[h]=[value(model.p_s2[h,j]) for j in model.periods]
        scenarios[h]=[value(model.phi_s[h,j]) for j in model.periods]
        penalty_scenarios[h]=[value(model.L_p_s[h,j]) for j in model.periods]
        
    #Plotting stacked plot for production plan    
    plt.figure(figsize=(10, 6))
    bottom = [0] * len(model.periods)

    for plant in model.plants:
        plt.bar(model.periods, prod[plant], label=plant, bottom=bottom)
        bottom = [bottom[i] + prod[plant][i] for i in range(len(model.periods))]
        
    plt.bar(model.periods, solar['Solar'], label='Solar', bottom=bottom)
    bottom = [bottom[i] + solar['Solar'][i] for i in range(len(model.periods))]
    
    for pen in model.penalty:
        plt.bar(model.periods, penalty[pen], label='Load Penalty', bottom=bottom)
        bottom = [bottom[i] + penalty[pen][i] for i in range(len(model.periods))]
        
    plt.xlabel("Period [h]")
    plt.ylabel("Production [MW]")
    plt.title("Optimal production plan first stage")
    plt.legend()
    plt.show()
    
    for s in model.scenarios:
        plt.figure(figsize=(10, 6))
        bottom = [0] * len(model.periods)
        if model.probs[s]>0:
            plt.bar(model.periods, hydro1_scenarios[s], label='Hydro1', bottom=bottom)
            bottom = [bottom[i] + hydro1_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, hydro2_scenarios[s], label='Hydro2', bottom=bottom)
            bottom = [bottom[i] + hydro2_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, scenarios[s], label='Solar', bottom=bottom)
            bottom = [bottom[i] + scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, penalty_scenarios[s], label='Load Penalty', bottom=bottom)
            bottom = [bottom[i] + penalty_scenarios[s][i] for i in range(len(model.periods))]
            plt.xlabel("Period [h]")
            plt.ylabel("Production [MW]")
            plt.title('Optimal production plan second stage: {}'.format(s))
            plt.legend()
            plt.show()
    
    prod_total = {plant: sum(values) for plant, values in prod.items()}

    print(prod_total)

    scenario_to_sum = 'S_avg'
    
    scenario_high='S_high'
    
    scenario_low='S_low'

    hydro1_total_scenario = sum(hydro1_scenarios[scenario_to_sum])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario = sum(hydro2_scenarios[scenario_to_sum])
    
    hydro1_total_scenario2 = sum(hydro1_scenarios[scenario_high])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario2 = sum(hydro2_scenarios[scenario_high])
    
    hydro1_total_scenario3 = sum(hydro1_scenarios[scenario_low])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario3= sum(hydro2_scenarios[scenario_low])
    
    
    print('hydro1 S_high:', hydro1_total_scenario2)
    print('hydro2 S_high:',hydro2_total_scenario2)
    
    print('hydro1 S_avg:', hydro1_total_scenario)
    print('hydro2 S_avg:',hydro2_total_scenario)

    print('hydro1 S_low:', hydro1_total_scenario3)
    print('hydro2 S_low:',hydro2_total_scenario3)

  
  
