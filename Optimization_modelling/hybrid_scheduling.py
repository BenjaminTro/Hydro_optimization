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


model = pyo.ConcreteModel()

#--Data handling--
#Read of parameters for portfolio
def InputData(data_file):
    inputdata = pd.read_excel(data_file)
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()
    data = {}
    data['hydro'] = inputdata[['Ci', 'yi', 'P_min', 'P_max']].drop('Solar')
    data['solar']=inputdata[['Ci', 'yi', 'P_min', 'P_max']].drop('Hydro1').drop('Hydro2')
    return data
data=InputData('data/Parameters.xlsx')

#defining start and end of optimization
start_date='2018-05-28 00:00'
end_date='2018-05-28 23:00'

#extract average market price
input_data_market = read_csv_data('data/Market_price.csv')
market_prices_h=convert_to_dict(input_data_market, start_date, end_date, 'H')
avg_market_price=average_value(market_prices_h)

#Solar production based on forecast (should come from irradiance data)
input_data_PV = read_excel_data('data/PV_spec.xlsx')
input_data_Irr = read_irr_data('data/Data_solar_irr_NOR.csv')
PV_power = pv_power_estimated(input_data_PV,input_data_Irr)
Solar_1=convert_to_dict(PV_power, start_date, end_date, 'H')
Solar_p=scale_dict(Solar_1, 10)

#--Constants--
Constants= {
    'Market_penalty':100, 
}

#--Defining sets--
#Defining the set of plants
model.plants = pyo.Set(initialize=['Hydro1','Hydro2'])
model.solar=pyo.Set(initialize=['Solar'])

#Defining the market
model.market=pyo.Set(initialize=['Market']) 

#Defining the set of periods
periods=set()
for i in range (1,25):
    periods.add(i) 
model.periods = pyo.Set(initialize=periods)

#--Defining parameters--
#Inital cost of buying from market
model.Fi=pyo.Param(model.market, initialize=Constants['Market_penalty'])

#Averaged NO3 market price for chosen simulation day
model.Mi=pyo.Param(model.market, initialize=avg_market_price)

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


#Load (should be imported from dataset)
L= {1:30, 2:20, 3:20, 4:30, 5:50, 6:80, 7:50, 8:90, 9:110, 10:150, 11:120, 12:80, 13:70, 14:80, 15:90, 16:160, 17:170, 18:150, 19:120, 20:100, 21:70, 22:60, 23:50, 24:40} 


#--Defining variables and bounds-- 
#Production from hydro plants
def p_bounds(model,i,j):
    return (model.Pmin[i],model.Pmax[i])
model.p = pyo.Var(model.plants,model.periods, bounds=p_bounds)

#Production from solar plants
def phi_bounds(model,s,j):
    return (model.Phi_min[s],model.Phi_max[s])
model.phi = pyo.Var(model.solar,model.periods, bounds=phi_bounds)

#Buying from market 
model.m = pyo.Var(model.market, model.periods, within=NonNegativeReals)


#--Defining constraints--

def Solar_rule(model,j):
    return  model.phi['Solar',j] == Solar_p[j]
model.solar_cons = pyo.Constraint(model.periods, rule=Solar_rule)

def load_rule(model,j):
    return model.p['Hydro1',j] + model.p['Hydro2',j] + model.phi['Solar',j] + model.m['Market',j]== L[j]
model.load_cons = pyo.Constraint(model.periods, rule=load_rule)


#Objective function 
def ObjRule(model):
    return sum(model.Ci[i]+model.yi[i]*model.p[i,j] for i in model.plants for j in model.periods) +sum(model.Si[s] +model.ki[s]*model.phi[s,j] for s in model.solar for j in model.periods )+sum(model.Fi[n]+model.Mi[n]*model.m[n,j] for n in model.market for j in model.periods)
model.obj= pyo.Objective(rule=ObjRule, sense=pyo.minimize)

#defining dual 
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

#solver 
opt = SolverFactory('gurobi', solver_io="python") 
results = opt.solve(model,tee=True) 

print('\n')

#printing solver
for v in model.component_data_objects(pyo.Var):
  print('%s:   %s'%(str(v), v.value))
  
  
model.display()
model.dual.display()

#appending production values in dictionary
prod={}
for i in model.plants:
    prod[i]=[value(model.p[i, j]) for j in model.periods]
    
solar={}
for s in model.solar:
    solar[s]=[value(model.phi[s, j]) for j in model.periods]

market={}    
for n in model.market:
    market[n]=[value(model.m[n,j]) for j in model.periods]
    


#Plotting stacked plot for production plan     
plt.figure(figsize=(10, 6))
bottom = [0] * len(model.periods)

for plant in model.plants:
    plt.bar(model.periods, prod[plant], label=plant, bottom=bottom)
    bottom = [bottom[i] + prod[plant][i] for i in range(len(model.periods))]

for sol in model.solar:
    plt.bar(model.periods, solar[sol], label='Solar', bottom=bottom)
    bottom = [bottom[i] + solar[sol][i] for i in range(len(model.periods))]
    
for mark in model.market:
    plt.bar(model.periods, market[mark], label='Market', bottom=bottom)
    bottom = [bottom[i] + market[mark][i] for i in range(len(model.periods))]

plt.xlabel("Period [h]")
plt.ylabel("Production [MW]")
plt.title("Optimal production plan")
plt.legend()
plt.show()

print(Solar_p)


