import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from scipy.optimize import least_squares
from scipy.optimize import NonlinearConstraint
from scipy.optimize import LinearConstraint
from scipy.optimize import minimize
import os


class Hyperelastic:

    def __init__(self, model, parameters, order, data_type):
        self.model = model                # model = "Ogden" or "Mooney Rivlin" or ...
        self.order = order                # order = 1 or 2 or 3
        self.parameters = parameters
        self.param_names = []
        self.data_type = data_type        # data_type = 'True' or 'Engineering'
        self.fitting_method = 'lm'        # fitting_method = 'lm' or 'trust-constr'
        
        if model == 'Ogden':
            initialGuessMu = np.array([0.1]*self.order)
            initialGuessAlpha = np.array([0.2]*self.order)
            self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
            self.nbparam = self.order*2
            muVec_names = ["µ1","µ2","µ3"][0:self.order]
            alphaVec_names = ["α1","α2","α3"][0:self.order]
            self.param_names = np.append(muVec_names,alphaVec_names)
            self.fitting_method = 'trust-constr'
        elif model == 'Neo Hookean':
            self.initialGuessParam = np.array([0.1])
            self.nbparam = 1            
            self.param_names = ["µ"]
            self.fitting_method = 'lm'
        elif model == 'Yeoh':
            self.initialGuessParam = np.array([0.1]*self.order)
            self.nbparam = self.order
            self.param_names = ["C1","C2","C3"][0:self.order]
            self.fitting_method = 'lm'
        elif model == 'Mooney Rivlin':
            self.initialGuessParam = np.array([0.1]*self.order)
            self.nbparam = self.order
            self.param_names = ["C10","C01","C20"][0:self.order]
            self.fitting_method = 'trust-constr'
        elif model == 'Gent':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["µ","Jm"]
            self.fitting_method = 'lm'
        elif model == 'Veronda Westmann':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["C1","C2"]
            self.fitting_method = 'lm'
        elif model == 'Humphrey':
            self.initialGuessParam = np.array([0.1]*2)
            self.nbparam = 2
            self.order=2
            self.param_names = ["C1","C2"]
            self.fitting_method = 'lm'
        else:
            print("Error. Wrong name of model in Hyperelastic")



    def YeohModel(self, cVec, Strain):
        """Yeoh hyperelastic model (incompressible material under uniaxial tension)"""
    
        if self.data_type == 'True':
            lambd = np.exp(Strain)
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        I1 = lambd**2 + 2/lambd

        Stress = np.zeros((self.order,len(Strain)))

        for i in range (0,self.order):
            if self.data_type == 'True':
                Stress[i,:] = 2*(lambd**2 - 1/lambd)*(i+1)*cVec[i]*((I1-3)**(i)) # true
            elif self.data_type == 'Engineering':
                Stress[i,:] = 2*(lambd - 1/(lambd**2))*(i+1)*cVec[i]*(I1-3)**(i) # eng
            else:
                print("Data type error. Data is neither 'True' or 'Engineering'. ")

            Stress_sum = np.sum(Stress, axis=0)
        return Stress_sum



    def NeoHookeanModel(self, mu, Strain):
        """Neo-Hookean hyperelastic model (incompressible material under uniaxial tension)"""

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
            Stress = mu*(lambd**2 - 1/lambd)
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
            Stress = mu*(lambd - 1/(lambd**2))
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        return Stress



    def OgdenModel(self, parameters, Strain):
        """Ogden hyperelastic model (incompressible material under uniaxial tension)"""
                
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
        muVec = parameters[0:self.order]
        alphaVec = parameters[self.order:]
        
        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ") 
        
        # broadcasting method to speed up computation
        lambd = lambd[np.newaxis, :]
        muVec = muVec[:self.order, np.newaxis]
        alphaVec = alphaVec[:self.order, np.newaxis]

        if self.data_type == 'True':
            Stress = np.sum(2*muVec*(lambd**(alphaVec - 1) - lambd**(-((1/2)*alphaVec + 1))), axis=0)
        elif self.data_type == 'Engineering':
            Stress = np.sum((muVec*(lambd**alphaVec - 1/(lambd**(alphaVec/2)))/lambd), axis=0)
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ") 

        return Stress



    def MooneyRivlinModel(self, cVec, Strain):
        """Mooney Rivlin hyperelastic model (incompressible material under uniaxial tension)"""
        
        cVec = np.append(cVec, np.zeros(3-self.order) ) #To ensure CXX is zero if unsed
        C10 = cVec[0]
        C01 = cVec[1]
        C20 = cVec[2]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
            Stress = 2*(lambd**2 - 1/lambd)*(C10 + C01/lambd + 2*C20*(lambd**2 + 2/lambd -3))
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
            Stress = (2*(lambd**2 - 1/lambd)*(C10 + C01/lambd + 2*C20*(lambd**2 + 2/lambd -3)))/lambd
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        return Stress



    def GentModel(self, parameters, Strain):
        """Gent hyperelastic model (incompressible material under uniaxial tension)"""
       
        mu = parameters[0]
        Jm = parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")   

        I1 = lambd**2 + 2/lambd

        if self.data_type == 'True':
            Stress = (lambd**2 - 1/lambd)*(mu*Jm / (Jm - I1 + 3))
        elif self.data_type == 'Engineering':
            Stress = (lambd - 1/lambd**2)*(mu*Jm / (Jm - I1 + 3))

        return Stress
 
    
    
    def VerondaWestmannModel(self, parameters, Strain):
        """Veronda-Westmann hyperelastic model (incompressible material under uniaxial tension)"""  
        
        C1=parameters[0]
        C2=parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")   

        I1 = lambd**2 + 2/lambd

        if self.data_type == 'True':
            Stress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3) - 1/(2*lambd)))
        elif self.data_type == 'Engineering':
            Stress = (2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3) - 1/(2*lambd))))/lambd                
        
        return Stress
   
 
    def HumphreyModel(self, parameters, Strain):
        """Humphrey hyperelastic model (incompressible material under uniaxial tension)"""   
        
        C1=parameters[0]
        C2=parameters[1]

        if self.data_type == 'True':
            lambd = np.exp(Strain)       # lambd i.e lambda
        elif self.data_type == 'Engineering':
            lambd = 1 + Strain
        else:
            print("Data type error. Data is neither 'True' or 'Engineering'. ")

        I1 = lambd**2 + 2/lambd        
        
        if self.data_type == 'True':
            Stress = 2*(lambd**2 - 1/lambd) * C1*C2*(np.exp(C2*(I1-3)))
        elif self.data_type == 'Engineering':
            Stress = 2*(lambd - 1/lambd**2) * C1*C2*(np.exp(C2*(I1-3)))

        return Stress        
    
    
    def ConsitutiveModel(self, parameters, Strain):
        """ Constitutive Model"""      
        
        self.parameters = parameters # update parameters attribute      
        
        if self.model == 'Ogden':
            Stress = self.OgdenModel(self.parameters, Strain)
        elif self.model == 'Neo Hookean':
            Stress = self.NeoHookeanModel(self.parameters, Strain)      
        elif self.model == 'Yeoh':
            Stress = self.YeohModel(self.parameters, Strain)            
        elif self.model == 'Mooney Rivlin':
            Stress = self.MooneyRivlinModel(self.parameters, Strain)            
        elif self.model == 'Gent':
            Stress = self.GentModel(self.parameters, Strain) 
        elif self.model == 'Veronda Westmann':
            Stress = self.VerondaWestmannModel(self.parameters, Strain) 
        elif self.model == 'Humphrey':
            Stress = self.HumphreyModel(self.parameters, Strain) 
        else:
            print("Error")
            
        return Stress



    def NonlinearConstraintFunction(self, parameters):
        """ Constraints function for 'trust-constr' optimisation algorithm"""      
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan]
        self.parameters = parameters # update parameters attribute      
        
        if self.model == 'Ogden':
            if self.order == 3:
                constraints_function = [self.parameters[0]*self.parameters[3], self.parameters[1]*self.parameters[4], self.parameters[2]*self.parameters[5]]
            elif self.order == 2:
                constraints_function = [self.parameters[0]*self.parameters[2], self.parameters[1]*self.parameters[3]]
            elif self.order == 1:
                constraints_function = [self.parameters[0]*self.parameters[1]]
            else:
                print("Error in OGDEN Hyperelastic.ConstraintsFunction")
        else:
            constraints_function = []
            print("Error in Hyperelastic.ConstraintsFunction")
           
        return constraints_function



    def NonlinearConstraintJacobian(self, parameters):
        """ Constraints function for 'trust-constr' optimisation algorithm"""      
        # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan]
        self.parameters = parameters # update parameters attribute      
        
        if self.model == 'Ogden':
            if self.order == 3:
                constraints_jacobian = [[self.parameters[3], 0, 0, self.parameters[0], 0, 0], [0, self.parameters[4], 0, 0, self.parameters[1], 0], [0, 0, self.parameters[5], 0, 0, self.parameters[2]]]
            elif self.order == 2:
                constraints_jacobian = [[self.parameters[2], 0, self.parameters[0], 0], [0, self.parameters[3], 0, self.parameters[1]]]
            elif self.order == 1:
                constraints_jacobian = [self.parameters[1], self.parameters[0]]
            else:
                print("Error in OGDEN Hyperelastic.ConstraintsFunction")
        else:
            constraints_jacobian = []
            print("Error in Hyperelastic.ConstraintsFunction")
           
        return constraints_jacobian


# class Hyperelastic:
#     def __init__(self, model, parameters, order):
#         self.model = model
#         self.order = order
#         self.parameters = parameters
#         self.param_names = []
#         self.fitting_method = 'lm'

#         # Initialization of the Ogden model
#         if model == 'Ogden':
#             initialGuessMu = np.array([1.0]*self.order)
#             initialGuessAlpha = np.array([1.0]*self.order)
#             self.initialGuessParam = np.append(initialGuessMu,initialGuessAlpha)
#             self.nbparam = self.order*2
#             muVec_names = ["µ1","µ2","µ3"][0:self.order]
#             alphaVec_names = ["α1","α2","α3"][0:self.order]
#             self.param_names = np.append(muVec_names,alphaVec_names)
#             self.fitting_method = 'trust-constr'
#         elif model == 'Neo Hookean':
#             self.initialGuessParam = np.array([0.1])
#             self.nbparam = 1            
#             self.param_names = ["µ"]
#             self.fitting_method = 'lm'
#         elif model == 'Mooney Rivlin':
#             self.initialGuessParam = np.array([0.1]*self.order)
#             self.nbparam = self.order
#             self.param_names = ["C10","C01","C20"][0:self.order]
#             self.fitting_method = 'trust-constr'    
#         #elif ... Initialization of other models (not detailed here but the Hyperelastic class is avalailbe on the Github repo)
#         else:
#             print("Error. Wrong name of model in Hyperelastic")


#     def OgdenModel(self, parameters, trueStrain):
#         """Ogden hyperelastic model (incompressible material under uniaxial tension)
#         Uses true strain and true stress data"""
                
#         # parameter is a 1D array : [mu0,mu1,...,mun,alpha0,alpha1,...,alphan] 
#         muVec = parameters[0:self.order]              # [mu0,mu1,...,mun]
#         alphaVec = parameters[self.order:]            # [alpha0,alpha1,...,alphan]
#         lambd = np.exp(trueStrain)
 
#         # broadcasting method to speed up computation
#         lambd = lambd[np.newaxis, :]
#         muVec = muVec[:self.order, np.newaxis]
#         alphaVec = alphaVec[:self.order, np.newaxis]
        
#         trueStress = np.sum(muVec*(lambd**alphaVec - 1/(lambd**(alphaVec/2))), axis=0)
#         return trueStress 


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


# Define parameters for which material/model/datatype to use
input_csv_filename = 'Dragon Skin 10 MEDIUM.csv'
input_data_type = 'True'
model_name = 'Ogden'



# read data from file
data_directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Tensile-Tests-Data'))
input_csv_filepath = os.path.join(data_directory_path, input_csv_filename)
data = pd.read_csv(input_csv_filepath, 
                   delimiter = ';',
                   skiprows= 18,
                   names = ['Time (s)','True Strain','True Stress (MPa)','Engineering Strain','Engineering Stress (MPa)'])
exp_strain = data['True Strain'].values        # converts panda series to numpy array
exp_stress = data['True Stress (MPa)'].values



# Instanciate a Hyperelastic object
hyperelastic = Hyperelastic(model_name, np.array([0]), order=3, data_type=input_data_type)
# hyperelastic.fitting_method = 'lm'


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
    optim_result = minimize(objectiveFun_Callback, hyperelastic.initialGuessParam, args=(exp_strain, exp_stress), method='trust-constr', constraints=const, tol=1e-12)    
elif hyperelastic.fitting_method == 'lm':
    # The least_squares package calls the Levenberg-Marquandt algorithm.
    # best-fit paramters are kept within optim_result.x
    optim_result = least_squares(objectiveFun_Callback, hyperelastic.initialGuessParam, method ='lm', gtol=1e-12, args=(exp_strain, exp_stress))   
else:
    print("Error in fitting method")

optim_parameters = optim_result.x
print(optim_parameters)


# Compute the true stress from the Ogden model with optimized parameters   
theo_stress = hyperelastic.ConsitutiveModel(optim_parameters, exp_strain)


# Plot experimental and predicted data on the same graph
plt.plot(exp_strain,exp_stress,'k',linewidth=2)
plt.plot(exp_strain,theo_stress,'r--', linewidth=2)
plt.xlabel('True Strain ' + r'$ \epsilon$')
plt.ylabel('True Stress ' +  r'$ \sigma$' + ' (MPa)')
plt.legend(['Experimental Data', 'Ogden model'],loc=2)


hyperelastic_stats = HyperelasticStats(exp_stress, theo_stress, hyperelastic.nbparam)
print(hyperelastic_stats.aic())
