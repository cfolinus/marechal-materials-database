#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 08:44:19 2024

@author: cfolinus
"""
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import least_squares
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import os
from tqdm import tqdm

# Database-specific imports (classes)
from Hyperelastic import Hyperelastic
from HyperelasticStats import HyperelasticStats




# Function to calculate the residuals.
# The fitting function holds the parameter values
def objectiveFun_Callback(parameters, exp_strain, exp_stress):
    theo_stress = hyperelastic.ConsitutiveModel(parameters, exp_strain)   
    # The cost function for Levenberg-Marquardt and Trust Constraint algorithms are not expressed the same way ! Check Scipy documentation 
    if hyperelastic.fitting_method == 'lm':
        residuals = theo_stress - exp_stress
    elif hyperelastic.fitting_method == 'trust-constr':
        residuals = np.sqrt(sum((theo_stress-exp_stress)**2.0))        
    else:
        print("Error, please chose either 'lm' or 'trust-constr' as fitting method")
    
    return residuals

def fitModelParameters (hyperelastic, objective_function):
     # The least_squares package calls the Levenberg-Marquandt algorithm
     # best-fit parameters are kept within optim_result.x
     if hyperelastic.fitting_method == 'trust-constr':   
         if hyperelastic.model == 'Ogden':
             # Non Linear Conditions for the Ogden model : mu0*alpha0 > 0, mu1*alpha1 > 0, mu2*alpha2 > 0,
             const = NonlinearConstraint(hyperelastic.NonlinearConstraintFunction, 
                                         0.0, 
                                         np.inf,
                                         jac=hyperelastic.NonlinearConstraintJacobian)#, hess='2-point')
             
         elif hyperelastic.model == 'Mooney Rivlin':
             # Linear Conditions for the Mooney Rivlin model : C10 + C01 > 0
             const = LinearConstraint([[1.0, 1.0, 0.0][0:hyperelastic.order], [0.0, 0.0, 0.0][0:hyperelastic.order]], 0.0, np.inf)
         else:
             const=()
             
         # The ogden and Mooney Rivlin models need constraint optimisation which cannot be done with the Levenberg-Marquandt algorithm
         optim_result = minimize(objective_function, hyperelastic.initialGuessParam, args=(exp_strain, exp_stress), method='trust-constr', constraints=const, tol=1e-12)    
     elif hyperelastic.fitting_method == 'lm':
         # The least_squares package calls the Levenberg-Marquandt algorithm.
         # best-fit paramters are kept within optim_result.x
         optim_result = least_squares(objective_function, hyperelastic.initialGuessParam, method ='lm', gtol=1e-12, args=(exp_strain, exp_stress))   
     else:
         print("Error in fitting method")
    
     return optim_result.x

# Define all models to run
# [csv name, engineering/true, model name]
cases_to_run = [
     ['Ecoflex 00-30.csv', 'True', 'Neo Hookean'],
     ['Ecoflex 00-30.csv', 'True', 'Mooney Rivlin'],
     ['Ecoflex 00-30.csv', 'True', 'Yeoh'],
     ['Ecoflex 00-30.csv', 'True', 'Ogden'],
     ['Ecoflex 00-30.csv', 'True', 'Veronda Westmann'],
     ['Ecoflex 00-30.csv', 'True', 'Humphrey'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Neo Hookean'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Mooney Rivlin'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Yeoh'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Ogden'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Veronda Westmann'],
     ['Dragon Skin 10 MEDIUM.csv', 'True', 'Humphrey'],
     ['Dragon Skin 30.csv', 'True', 'Neo Hookean'],
     ['Dragon Skin 30.csv', 'True', 'Mooney Rivlin'],
     ['Dragon Skin 30.csv', 'True', 'Yeoh'],
     ['Dragon Skin 30.csv', 'True', 'Ogden'],
     ['Dragon Skin 30.csv', 'True', 'Veronda Westmann'],
     ['Dragon Skin 30.csv', 'True', 'Humphrey']
     ]
num_cases = len(cases_to_run)

# Define location of data (path to CSV files)
data_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '.', 'Tensile-Tests-Data'))
output_xlsx_filename = os.path.join(data_directory_path, 'Combined model outputs.xlsx')
# output_xlsx_writer = pd.ExcelWriter(output_xlsx_filename, mode='w')


for case_index in range(num_cases):
          
     # Extract information for this model configuration
     temp_case_information = cases_to_run[case_index]
     input_csv_filename = temp_case_information[0]
     input_data_type = temp_case_information[1]
     model_name = temp_case_information[2]

     # Import and format data from CSV file
     input_csv_filepath = os.path.join(data_directory_path, input_csv_filename)
     data = pd.read_csv(input_csv_filepath, 
                        delimiter = ';',
                        skiprows= 18,
                        names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)'])
     exp_strain = data['True Strain'].values        # converts panda series to numpy array
     exp_stress = data['True Stress (MPa)'].values

     # Instanciate a Hyperelastic object
     hyperelastic = Hyperelastic(model_name, np.array([0]), order=3, data_type=input_data_type)

     # Fit parameters for this constitutive model
     optim_parameters = fitModelParameters(hyperelastic, objectiveFun_Callback)

     # Compute the true stress from the Ogden model with optimized parameters   
     theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)

     # Compute statistics about this model/fitting
     hyperelastic_stats = HyperelasticStats(exp_stress, theo_stress, hyperelastic.nbparam)
     # print(hyperelastic_stats.aic())

     # Plot experimental and predicted data on the same graph
     plt.figure(figsize = (4,3), dpi = 720)
     plt.plot(exp_strain,exp_stress,'k',linewidth=2)
     plt.plot(exp_strain,theo_stress,'r--', linewidth=2)
     plt.xlabel('True Strain ' + r'$ \epsilon$')
     plt.ylabel('True Stress ' +  r'$ \sigma$' + ' (MPa)')
     plt.legend(['Experimental Data', 'Ogden model'],loc=2)

     # Format data for saving to excel spreadsheet
     model_information = [
          ['CSV filename', input_csv_filename],
          ['Data type', input_data_type],
          ['Model name', model_name]
          ]
     model_results = [
          [hyperelastic.param_names[i], hyperelastic.parameters[i]] for i in range(hyperelastic.nbparam)]
     model_statistics =[
          ['S', hyperelastic_stats.S()],
          ['AIC', hyperelastic_stats.aic()],
          ['adj_r_squared', hyperelastic_stats.adj_r_squared()],
          ['NRMSE', hyperelastic_stats.nrmse()],
          ['MAPD', hyperelastic_stats.mapd()]
          ]
     export_data = pd.DataFrame(model_information + model_results + model_statistics)
     
     # Export data to Excel sheet
     with pd.ExcelWriter(output_xlsx_filename, mode='a', if_sheet_exists='replace') as writer:
     
          temp_sheet_name = 'Sheet{sheet_number}'.format(sheet_number = case_index + 1)
          export_data.to_excel(writer, sheet_name = temp_sheet_name, index = False)


