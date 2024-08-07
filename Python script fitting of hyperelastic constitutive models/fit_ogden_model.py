#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:25:58 2024

@author: cfolinus
"""
# Pandas and table
import pandas as pd
from scipy.optimize import least_squares
import numpy as np
import matplotlib.pylab as plt
import os




def OgdenModel(trueStrain, parameters, order):
    """Ogden hyperelastic model (incompressible material under uniaxial tension)
    Uses true strain and true stress data"""
                
    # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
    muVec = parameters.reshape(2, order)[0]
    alphaVec = parameters.reshape(2, order)[1]
    lambd = np.exp(trueStrain)
    # broadcasting method to speed up computation
    lambd = lambd[np.newaxis, :]
    muVec = muVec[:order, np.newaxis]
    alphaVec = alphaVec[:order, np.newaxis]
        
    trueStress = np.sum(2*muVec*(lambd**(alphaVec - 1) - lambd**(-((1/2)*alphaVec + 1))), axis=0)
    return trueStress




class HyperelasticStats:
    
    def __init__(self, Yexp, Ymodel, p):
        self.target = Yexp
        self.model = Ymodel
        self.p = p
        self._n = len(Yexp)   


    def sse(self):
        '''returns sum of squared errors (model vs actual)'''
        squared_errors = (self.target - self.model) ** 2
        return np.sum(squared_errors)
        
    def sst(self):
        '''returns total sum of squared errors (actual vs avg(actual))'''
        avg_Yexp = np.mean(self.target)
        squared_errors = (self.target - avg_Yexp) ** 2
        return np.sum(squared_errors)
    
    def rmse(self):
        '''returns rmse'''
        squared_errors = (self.target - self.model) ** 2
        return np.sqrt(np.mean(squared_errors))

    def r_squared(self):
        '''returns calculated value of r^2'''
        return 1 - self.sse()/self.sst()
    
    def adj_r_squared(self):
        '''returns calculated value of adjusted r^2
        R¯² = 1 - (n-1)(R²-1)/(n-p-1)'''
        adj_squared_errors =   1 - ((self._n - 1)*(self.r_squared() - 1)/(self._n - self.p - 1))     
        return adj_squared_errors
    
    def aic(self):
       '''returns the Akaike Information Criterion'''   
       return self._n *np.log(self.sse()/self._n) + 2*self.p
   
    def S(self):
       '''returns the Residual Standard Error (S)'''   
       return np.sqrt(self.sse()/(self._n - self.p - 1))
   
    def mapd(self):
       '''returns the mean absolute percentage deviation (MAPD)'''
       relativeError = np.zeros(np.min([len(self.model),len(self.target)]))
       for i in range (0,np.min([len(self.model),len(self.target)])):
           relativeError[i] = np.absolute((self.target[i]-self.model[i])/self.model[i])
       return (100/self._n)*np.sum(relativeError)
  
     
  

# cost function to calculate the residuals. The fitting function holds the parameter values.  
def objectiveFun_Callback(parameters, exp_strain, exp_stress):  
    theo_stress = OgdenModel(exp_strain, parameters, order)   
    residuals = theo_stress - exp_stress 
    return residuals


# Define parameters for which material/model/datatype to use
input_csv_filename = 'Dragon Skin 30.csv'
input_data_type = 'True'
model_name = 'Ogden'

# read data from file
data_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tensile-Tests-Data'))
input_csv_filepath = os.path.join(data_directory_path, input_csv_filename)
data = pd.read_csv(input_csv_filepath, 
                   delimiter = ';',
                   skiprows= 18,
                   names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)'])

# Convert data to appropriate format
exp_strain = data['True Strain'].values        # converts panda series to numpy array
exp_stress = data['True Stress (MPa)'].values



# Set up for least squares optimization
order = 3
initialGuessMu = np.array([1]*order)     # ["µ1","µ2","µ3"]
initialGuessAlpha = np.array([1]*order)  # ["α1","α2","α3"]
initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
nbparam = order*2
param_names = ["µ1","µ2","µ3","α1","α2","α3"]

# The least_squares package calls the Levenberg-Marquandt algorithm.
# best-fit paramters are kept within optim_result.x
optim_result = least_squares(objectiveFun_Callback, initialGuessParam, method ='lm', args=(exp_strain, exp_stress))
optim_parameters = optim_result.x
theo_stress = OgdenModel(exp_strain, optim_parameters, order)



print('optimised parameters:')
print('µ1=' + str(optim_parameters[0]))
print('µ2=' + str(optim_parameters[1]))
print('µ3=' + str(optim_parameters[2]))
print('α1=' + str(optim_parameters[3]))
print('α2=' + str(optim_parameters[4]))
print('α3=' + str(optim_parameters[5]))


# Plotting the data
plt.plot(exp_strain,exp_stress,'k', marker="o", markersize=2, linestyle = 'None')
plt.plot(exp_strain,theo_stress,'r', linewidth=1)
#plt.title('Least-squares fit to data')
plt.xlabel('True Strain ' + r'$ \epsilon$')
plt.ylabel('True Stress ' +  r'$ \sigma$' + ' (MPa)')
plt.legend(['Experimental Data', 'Ogden model'],loc=2)
plt.grid(visible=None, which='both', axis='both')


hyperelastic_stats = HyperelasticStats(exp_stress, theo_stress, nbparam)
