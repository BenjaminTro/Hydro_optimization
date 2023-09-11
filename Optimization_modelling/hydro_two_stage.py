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
    data['prod'] = inputdata[['Ci', 'yi', 'P_min', 'P_max']]
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
Solar_p=convert_to_dict(PV_power, start_date, end_date, 'H')

#--Constants--

#--Defining sets--
#Defining the set of plants
model.plants = pyo.Set(initialize=['Hydro1','Hydro2', 'Solar'])

#Defining the market
model.market=pyo.Set(initialize=['Market']) 

#Defining the set of periods
periods=set()
for i in range (1,25):
    periods.add(i) 
model.periods = pyo.Set(initialize=[1,2,3,4,5,6,7, 8 , 9 , 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

#defining scenarios for two-stage problem 
#S_high=scale_dict(Solar_p, 120)
#S_avg=scale_dict(Solar_p, 100)
#S_low=scale_dict(Solar_p, 80)
#model.scenarios=pyo.Set(initialize=[S_high,S_avg, S_low])
#probabilities for each scenario 
#probs={S_high:'1/3',S_avg:'1/3', S_low:'1/3'}
#model.probs=pyo.Param(model.scenarios, initialize=[probs])
#Load demand [MW]

#--Defining parameters--
#Inital cost of buying from market
Fi=1000
model.Fi=pyo.Param(model.market, initialize=Fi)

#Market price
model.Mi=pyo.Param(model.market, initialize=avg_market_price)

#Initial costs for plants
model.Ci=pyo.Param(model.plants, initialize=data['prod']['Ci'])

#Variable costs for plants 
model.yi=pyo.Param(model.plants, initialize=data['prod']['yi'])

#Production bounds for plants
model.Pmin=pyo.Param(model.plants, initialize=data['prod']['P_min'])
model.Pmax=pyo.Param(model.plants, initialize=data['prod']['P_max'])
#Load
L= {1:30, 2:20, 3:20, 4:30, 5:50, 6:80, 7:50, 8:90, 9:110, 10:150, 11:120, 12:80, 13:70, 14:80, 15:90, 16:160, 17:170, 18:150, 19:120, 20:100, 21:70, 22:60, 23:50, 24:40} 


#--Defining variables and bounds-- 
#Production from plants
def p_bounds(model,i,j):
    return (model.Pmin[i],model.Pmax[i])
model.p = pyo.Var(model.plants,model.periods, bounds=p_bounds)
#Buying from market 
model.m=pyo.Var(model.market, model.periods, within=NonNegativeReals)


#--Defining constraints--

def Solar_rule(model,j):
    return  model.p['Solar',j] == Solar_p[j]
model.solar_cons = pyo.Constraint(model.periods, rule=Solar_rule)

def load_rule(model,j):
    return model.p['Hydro1',j] + model.p['Hydro2',j] + model.p['Solar',j] + model.m['Market',j]== L[j]
model.load_cons = pyo.Constraint(model.periods, rule=load_rule)


#Objective function 
def ObjRule(model):
    return sum(model.Ci[i]+model.yi[i]*model.p[i,j] for i in model.plants for j in model.periods)+sum(model.Fi[n]+model.Mi[n]*model.m[n,j] for n in model.market for j in model.periods)
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
#prod={}
#for i in model.plants:
    #prod[i]=[value(model.p[i, j]) for j in model.periods]

#Plotting stacked plot for production plan     
#plt.figure(figsize=(10, 6))
#bottom = [0] * len(model.periods)

#for plant in model.plants:
    #plt.bar(model.periods, prod[plant], label=plant, bottom=bottom)
    #bottom = [bottom[i] + prod[plant][i] for i in range(len(model.periods))]

#plt.xlabel("Period [h]")
#plt.ylabel("Production [MW]")
#plt.title("Optimal production plan")
#plt.legend()
#plt.show()


