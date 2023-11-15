import sys

sys.path.append('C:\\Users\oscar\OneDrive\Dokumenter\Høst 2023\TET4565 Spesialiseringsemne\Hydro_optimization')  # OSCAR path
sys.path.append('C:\\Users\\benny\\Documents\\Hydro_optimization')  # BENJAMIN path
sys.path.append('C:\\Users\\Epsilon Delta\\OneDrive - NTNU\\Energi og Miljø NTNU\\Høst2023\\TET4565 Fordypningsemne\\Hydro_optimization')  # ESPEN path

import pyomo.environ as pyo
import numpy as np
from pyomo.environ import ConcreteModel, Set, RangeSet, Param, Suffix, Reals, NonNegativeReals, NonPositiveReals, \
    Binary, Objective, minimize, maximize, value
from pyomo.core import Constraint, Var, Block, ConstraintList
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import matplotlib.pyplot as plt
from calculations.datahandling import *
from calculations.data_processor import *


# --Data handling--
# Read of parameters for portfolio
def InputData(data_file):
    inputdata = pd.read_excel(data_file)
    inputdata = inputdata.set_index('Parameter', drop=True)
    inputdata = inputdata.transpose()
    data = {}
    data['hydro'] = inputdata[['yi', 'P_min', 'P_max', 'H_max']].drop('Solar')
    data['solar'] = inputdata[['yi', 'P_min', 'P_max']].drop('Hydro1').drop('Hydro2')
    return data


# Solar production based on forecast
def read_solar_data(irrad_file, spec_file, start_date, end_date):
    input_data_PV = read_excel_data(spec_file)
    input_data_Irr = read_irr_data(irrad_file)
    PV_power = pv_power_estimated(input_data_PV, input_data_Irr)
    Solar_1 = convert_to_dict(PV_power, start_date, end_date, 'H')
    Solar_p = scale_dict(Solar_1, 1)
    return Solar_p


# Start and end dates of the optimization
start_date = '2018-05-28 13:00'
end_date = '2018-05-28 13:00'

# Original forecast for solar power production
Solar_p = read_solar_data('data/Data_solar_irr_NOR.csv', 'data/PV_spec.xlsx', start_date, end_date)

Solar_p = 15
Market = 60

# Scenarios for solar forecast
S_high = Solar_p*2
S_avg = Solar_p*1
S_low = Solar_p*0

# Load
L = 150

# --Constants--
Constants = {
    'Load_penalty': 200,
    'Hydro_cap': 90,
    "Load": 150,
    'Market': 60,
    'Scenarios': ['S_high', 'S_avg', 'S_low'],
    'probs': {'S_high': 1 / 3, 'S_avg': 1 / 3, 'S_low': 1 / 3},
    'yi_s1': {'S_high': 25, 'S_avg': 25, 'S_low': 25},
    'yi_s2': {'S_high': 35, 'S_avg': 35, 'S_low': 35}
}


# Production bounds for hydro plants in stage 1
def p_bounds(model, i, j):
    return (model.Pmin[i], model.Pmax[i])


# Production constraint for hydro1 in stage 2
def hydro1_bounds(model, s, j):
    return model.p_s1[s, j] <= 60


# Production constraint for hydro2 in stage 2

def hydro2_bounds(model, s, j):
    return model.p_s2[s, j] <= 100


# Solar production in stage 1 must be equal to original forecast
def Solar_rule(model):
    return model.phi['Solar', 1] == 0


# Solar production in stage 2 must be equal to scenario for high production
def Solar_high(model, j):
    if model.probs['S_high'] > 0:
        return model.phi_s['S_high', 2] == S_high
    else:
        return model.phi_s['S_high', 2] == 0


# Solar production in stage 2 must be equal to scenario for average production
def Solar_avg(model, j):
    if model.probs['S_avg'] > 0:
        return model.phi_s['S_avg', 2] == S_avg
    else:
        return model.phi_s['S_avg', 2] == 0


# Solar production in stage 2 must be equal to scenario for low production
def Solar_low(model, j):
    if model.probs['S_low'] > 0:
        return model.phi_s['S_low', 2] == S_low
    else:
        return model.phi_s['S_low', 2] == 0


# Sum of hydro production in stage 1 must be lower than available generation
def Hydro_firststage(model, i,j):
    return (model.p[i, 1]) <= model.Hmax


# Sum of hydro production in stage 2 must be lower than capacity and the already used power in stage 1
def hydro1_scenario_bounds(model, s, j):
    return (model.p_s1[s, 2]) <= model.Hmax - (model.p['Hydro1', 1])


# Production constraint for hydro2 in stage 2
def hydro2_scenario_bounds(model, s, j):
    return (model.p_s2[s, 2]) <= model.Hmax - (model.p['Hydro2', 1])


# Total power generation in stage 1 must be equal to the hourly set load
#def load_rule_FirstStage(model, j):
#    return model.p['Hydro1', j] + model.p['Hydro2', j] + model.phi['Solar', j] + model.L_p['Load_penalty', j] == L[j]


# Total power generation in stage 2 must be equal to the hourly set load
def load_rule_TwoStage(model, s, j):
    return model.p_s1[s, 2] + model.p_s2[s, 2] + model.phi_s[s, 2] + model.L_p_s[s, 2] == L


# Objective function
def ObjRule(model):
    # Sum of production costs in stage 1
    first_stage_obj = sum(model.yi[i] * model.p[i, 1] for i in model.plants) - sum(model.Mi[i]*model.p[i, 1] for i in model.plants) #minimize(power produced - power sold to market)


    # Second stage objective (scenario-dependent), sum of all production costs
    second_stage_obj = sum(model.probs[l] * (
            model.yi_s1[l] * model.p_s1[l, 2] + model.yi_s2[l] * model.p_s2[l, 2] + model.Li[n] * model.L_p_s[l, 2]
    ) for l in model.scenarios for n in model.penalty)

    # Total objective is the sum of first and second stage objectives
    return first_stage_obj + second_stage_obj


def model_setup(Constants, data):
    # --Defining sets--
    model = pyo.ConcreteModel()
    # Defining the set of plants
    model.plants = pyo.Set(initialize=['Hydro1', 'Hydro2'])
    model.solar = pyo.Set(initialize=['Solar'])
    # Defining the load penalty
    model.penalty = pyo.Set(initialize=['Load_penalty'])
    # Defining the set of periods
    periods = set()
    for i in range(1, 3):
        periods.add(i)
    model.periods = pyo.Set(initialize=periods)
    # Defining scenarios for two-stage problem
    model.scenarios = pyo.Set(initialize=Constants['Scenarios'])

    # --Defining parameters--
    # Load penalty
    model.Li = pyo.Param(model.penalty, initialize=Constants['Load_penalty'])
    # Variable costs for plants
    model.yi = pyo.Param(model.plants, initialize=data['hydro']['yi'])
    model.ki = pyo.Param(model.solar, initialize=data['solar']['yi'])
    model.yi_s1 = pyo.Param(model.scenarios, initialize=Constants['yi_s1'])
    model.yi_s2 = pyo.Param(model.scenarios, initialize=Constants['yi_s2'])
    # Production bounds for plants
    model.Pmin = pyo.Param(model.plants, initialize=data['hydro']['P_min'])
    model.Pmax = pyo.Param(model.plants, initialize=data['hydro']['P_max'])
    model.Phi_min = pyo.Param(model.solar, initialize=data['solar']['P_min'])
    model.Phi_max = pyo.Param(model.solar, initialize=data['solar']['P_max'])
    # probabilities for each scenario
    model.probs = pyo.Param(model.scenarios, initialize=Constants['probs'])
    # Maximum hydro production available for the two plant for 24 hours
    model.Hmax = pyo.Param(initialize=Constants['Hydro_cap'])
    model.Mi = pyo.Param(model.plants, initialize=Constants['Market'])

    # --Defining variables and bounds--
    # Production from hydro plants stage 1
    model.p = pyo.Var(model.plants, model.periods, bounds=p_bounds)
    # Scenario and stage 2 based hydro power
    model.p_s1 = pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)
    model.p_s2 = pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)
    # Production from solar installation base case
    model.phi = pyo.Var(model.solar, model.periods, within=NonNegativeReals)
    # Production from solar installation scenario based
    model.phi_s = pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)
    # Load penalty during scenarios
    model.L_p_s = pyo.Var(model.scenarios, model.periods, within=NonNegativeReals)

    # --Defining constraints-- all based on constraint functions defined above
    model.solar_cons = pyo.Constraint(rule=Solar_rule)

    model.high_cons = pyo.Constraint(model.periods, rule=Solar_high)

    model.avg_cons = pyo.Constraint(model.periods, rule=Solar_avg)

    model.low_cons = pyo.Constraint(model.periods, rule=Solar_low)

    model.hydro1_cons = pyo.Constraint(model.scenarios, model.periods, rule=hydro1_bounds)

    model.hydro2_cons = pyo.Constraint(model.scenarios, model.periods, rule=hydro2_bounds)

    model.hydro_cons = pyo.Constraint(model.plants, model.periods, rule=Hydro_firststage)

    model.hydro1_scenario_cons = pyo.Constraint(model.scenarios, model.periods, rule=hydro1_scenario_bounds)

    model.hydro2_scenario_cons = pyo.Constraint(model.scenarios, model.periods, rule=hydro2_scenario_bounds)

    model.load_cons_TwoStage = pyo.Constraint(model.scenarios, model.periods, rule=load_rule_TwoStage)

    # Objective rule
    model.obj = pyo.Objective(rule=ObjRule, sense=pyo.minimize)
    return model


def solve(model):
    # solver
    opt = SolverFactory('gurobi', solver_io="python")
    # defining dual
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
    results = opt.solve(model, tee=True)
    return results, model


def displayresults(model):
    return print(model.display())  # model.dual.display())


def plotting(model):
    prod = {}
    penalty_scenarios = {}
    scenarios = {}
    hydro1_scenarios = {}
    hydro2_scenarios = {}
    for i in model.plants:
        prod[i] = [value(model.p[i, 1]), 0]
    for h in model.scenarios:
        hydro1_scenarios[h] = [0, value(model.p_s1[h, 2])]
        hydro2_scenarios[h] = [0, value(model.p_s2[h, 2])]
        scenarios[h] = [0, value(model.phi_s[h, 2])]
        penalty_scenarios[h] = [0, value(model.L_p_s[h, 2])]

    # Plotting stacked plot for production plan
    plt.figure(figsize=(10, 6))
    bottom = [0]*len(model.periods)

    for plant in model.plants:
        plt.bar(model.periods, prod[plant], label=plant, bottom=bottom)
        bottom = [bottom[i] + prod[plant][i] for i in range(len(model.periods))]
        
    for s in model.scenarios:
        #plt.figure(figsize=(10, 6))
        #bottom = [0]*len(model.periods)
        if model.probs[s] > 0:
            plt.bar(model.periods, hydro1_scenarios[s], bottom=bottom, color="C0")
            bottom = [bottom[i] + hydro1_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, hydro2_scenarios[s], bottom=bottom, color="C1")
            bottom = [bottom[i] + hydro2_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, scenarios[s], label='Solar', bottom=bottom, color="C2")
            bottom = [bottom[i] + scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, penalty_scenarios[s], label='Load Penalty', bottom=bottom, color="C3")
            bottom = [bottom[i] + penalty_scenarios[s][i] for i in range(len(model.periods))]
            Title="Optimal production plan, individual scenario:  {}".format(s)
    plt.xlabel("Stage")
    plt.ylabel("Production [MW]")
    plt.title(Title)
    plt.legend()
    plt.show()



    prod_total = {plant: sum(values) for plant, values in prod.items()}

    print(prod_total)

    scenario_to_sum = 'S_avg'

    scenario_high = 'S_high'

    scenario_low = 'S_low'

    hydro1_total_scenario = sum(hydro1_scenarios[scenario_to_sum])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario = sum(hydro2_scenarios[scenario_to_sum])

    hydro1_total_scenario2 = sum(hydro1_scenarios[scenario_high])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario2 = sum(hydro2_scenarios[scenario_high])

    hydro1_total_scenario3 = sum(hydro1_scenarios[scenario_low])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario3 = sum(hydro2_scenarios[scenario_low])

    print('hydro1 S_high:', hydro1_total_scenario2)
    print('hydro2 S_high:', hydro2_total_scenario2)

    print('hydro1 S_avg:', hydro1_total_scenario)
    print('hydro2 S_avg:', hydro2_total_scenario)

    print('hydro1 S_low:', hydro1_total_scenario3)
    print('hydro2 S_low:', hydro2_total_scenario3)


def plotting_stoch(model):
    prod = {}
    penalty_scenarios = {}
    scenarios = {}
    hydro1_scenarios = {}
    hydro2_scenarios = {}
    for i in model.plants:
        prod[i] = [value(model.p[i, 1]), 0]
    for h in model.scenarios:
        hydro1_scenarios[h] = [0, value(model.p_s1[h, 2])]
        hydro2_scenarios[h] = [0, value(model.p_s2[h, 2])]
        scenarios[h] = [0, value(model.phi_s[h, 2])]
        penalty_scenarios[h] = [0, value(model.L_p_s[h, 2])]
    for s in model.scenarios:
        
        # Plotting stacked plot for production plan
        plt.figure(figsize=(10, 6))
        bottom = [0]*len(model.periods)

        for plant in model.plants:
            plt.bar(model.periods, prod[plant], label=plant, bottom=bottom)
            bottom = [bottom[i] + prod[plant][i] for i in range(len(model.periods))]
        if model.probs[s] > 0:
            plt.bar(model.periods, hydro1_scenarios[s], bottom=bottom, color="C0")
            bottom = [bottom[i] + hydro1_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, hydro2_scenarios[s], bottom=bottom, color="C1")
            bottom = [bottom[i] + hydro2_scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, scenarios[s], label='Solar', bottom=bottom, color="C2")
            bottom = [bottom[i] + scenarios[s][i] for i in range(len(model.periods))]
            plt.bar(model.periods, penalty_scenarios[s], label='Load Penalty', bottom=bottom, color="C3")
            bottom = [bottom[i] + penalty_scenarios[s][i] for i in range(len(model.periods))]
            plt.xlabel("Stage")
            plt.ylabel("Production [MW]")
            plt.title("Stochastic production plan for scenario:  {}".format(s))
            plt.legend()
            plt.show()



    prod_total = {plant: sum(values) for plant, values in prod.items()}

    print(prod_total)

    scenario_to_sum = 'S_avg'

    scenario_high = 'S_high'

    scenario_low = 'S_low'

    hydro1_total_scenario = sum(hydro1_scenarios[scenario_to_sum])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario = sum(hydro2_scenarios[scenario_to_sum])

    hydro1_total_scenario2 = sum(hydro1_scenarios[scenario_high])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario2 = sum(hydro2_scenarios[scenario_high])

    hydro1_total_scenario3 = sum(hydro1_scenarios[scenario_low])

    # Summing up production values for 'hydro2_scenarios' for the specified scenario
    hydro2_total_scenario3 = sum(hydro2_scenarios[scenario_low])

    print('hydro1 S_high:', hydro1_total_scenario2)
    print('hydro2 S_high:', hydro2_total_scenario2)

    print('hydro1 S_avg:', hydro1_total_scenario)
    print('hydro2 S_avg:', hydro2_total_scenario)

    print('hydro1 S_low:', hydro1_total_scenario3)
    print('hydro2 S_low:', hydro2_total_scenario3)