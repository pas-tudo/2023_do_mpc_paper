import numpy as np
from casadi import *
import do_mpc

class CSTR_Cascade:
    def __init__(self,n_reac=3):
        # Parameters
        self.n_reac=n_reac
        self.nx = 5*n_reac # 4 concentrations and 1 temperature each Reactor
        self.nu = 3*n_reac # Input of A+B+T_jacket
        self.np = 4*n_reac # 2 reaction rates and two heat of reactions each reactor, modelled independently due to e.g. fouling

        self.V_R=15
        self.V_i=self.V_R/n_reac
        self.V_out=1
        self.k1_0=2
        self.k2_0=2
        self.delH1_0=-100
        self.delH2_0=-50
        self.Tr_in=60
        self.kA=2
        self.rho=1
        self.cp=0.5
        self.Ea1=500
        self.Ea2=600
        self.R_gas=8.3145

        # Initialize model
        self.model = do_mpc.model.Model('continuous')
        self.cA = self.model.set_variable(var_type='_x', var_name='cA', shape=(n_reac,1))
        self.cB = self.model.set_variable(var_type='_x', var_name='cB', shape=(n_reac,1))
        self.cR = self.model.set_variable(var_type='_x', var_name='cR', shape=(n_reac,1))
        self.cS = self.model.set_variable(var_type='_x', var_name='cS', shape=(n_reac,1))
        self.Tr = self.model.set_variable(var_type='_x', var_name='Tr', shape=(n_reac,1))

        self.uA = self.model.set_variable(var_type='_u', var_name='uA',shape=(n_reac,1))
        self.uB = self.model.set_variable(var_type='_u', var_name='uB',shape=(n_reac,1))
        self.Tj = self.model.set_variable(var_type='_u', var_name='Tj',shape=(n_reac,1))

        self.k1_mean=self.model.set_variable(var_type='_tvp', var_name='k1_mean',shape=(n_reac,1))
        self.k2_mean=self.model.set_variable(var_type='_tvp', var_name='k2_mean',shape=(n_reac,1))
        self.delH1_mean=self.model.set_variable(var_type='_tvp', var_name='delH1_mean',shape=(n_reac,1))
        self.delH2_mean=self.model.set_variable(var_type='_tvp', var_name='delH2_mean',shape=(n_reac,1))

        self.k1_var=self.model.set_variable(var_type='_p', var_name='k1_var',shape=(n_reac,1))
        self.k2_var=self.model.set_variable(var_type='_p', var_name='k2_var',shape=(n_reac,1))
        self.delH1_var=self.model.set_variable(var_type='_p', var_name='delH1_var',shape=(n_reac,1))
        self.delH2_var=self.model.set_variable(var_type='_p', var_name='delH2_var',shape=(n_reac,1))


    def get_model(self, process_noise=False):
        # Prepare intermediates
        rhs_cA=[]
        rhs_cB=[]
        rhs_cR=[]
        rhs_cS=[]
        rhs_Tr=[]

        rhs_cA.append(-self.V_out/self.V_i*self.cA[0]-self.k1_mean[0]*self.k1_var[0]*exp(-self.Ea1/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cB[0]-2*self.k2_mean[0]*self.k2_var[0]*exp(-self.Ea2/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]**2+self.uA[0]/self.V_i)
        rhs_cB.append(-self.V_out/self.V_i*self.cB[0]-self.k1_mean[0]*self.k1_var[0]*exp(-self.Ea1/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cB[0]+self.uB[0]/self.V_i)
        rhs_cR.append(-self.V_out/self.V_i*self.cR[0]+self.k1_mean[0]*self.k1_var[0]*exp(-self.Ea1/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cB[0])
        rhs_cS.append(-self.V_out/self.V_i*self.cS[0]+self.k2_mean[0]*self.k2_var[0]*exp(-self.Ea2/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cA[0])
        rhs_Tr.append(self.V_out/self.V_i*(self.Tr_in-self.Tr[0])+self.kA/(self.rho*self.cp*self.V_i)*(self.Tj[0]-self.Tr[0])-self.delH1_mean[0]*self.delH1_var[0]/(self.rho*self.cp)*self.k1_mean[0]*self.k1_var[0]*exp(-self.Ea1/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cB[0]-self.delH2_mean[0]*self.delH2_var[0]/(self.rho*self.cp)*self.k2_mean[0]*self.k2_var[0]*exp(-self.Ea2/(self.R_gas*(self.Tr[0]+273.15)))*self.cA[0]*self.cA[0])

        for i in range(1,self.n_reac):
            rhs_cA.append(self.V_out/self.V_i*(self.cA[i-1]-self.cA[i])-self.k1_mean[i]*self.k1_var[i]*exp(-self.Ea1/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cB[i]-2*self.k2_mean[i]*self.k2_var[i]*exp(-self.Ea2/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]**2+self.uA[i]/self.V_i)
            rhs_cB.append(self.V_out/self.V_i*(self.cB[i-1]-self.cB[i])-self.k1_mean[i]*self.k1_var[i]*exp(-self.Ea1/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cB[i]+self.uB[i]/self.V_i)
            rhs_cR.append(self.V_out/self.V_i*(self.cR[i-1]-self.cR[i])+self.k1_mean[i]*self.k1_var[i]*exp(-self.Ea1/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cB[i])
            rhs_cS.append(self.V_out/self.V_i*(self.cS[i-1]-self.cS[i])+self.k2_mean[i]*self.k2_var[i]*exp(-self.Ea2/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cA[i])
            rhs_Tr.append(self.V_out/self.V_i*(self.Tr[i-1]-self.Tr[i])+self.kA/(self.rho*self.cp*self.V_i)*(self.Tj[i]-self.Tr[i])-self.delH1_mean[i]*self.delH1_var[i]/(self.rho*self.cp)*self.k1_mean[i]*self.k1_var[i]*exp(-self.Ea1/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cB[i]-self.delH2_mean[i]*self.delH2_var[i]/(self.rho*self.cp)*self.k2_mean[i]*self.k2_var[i]*exp(-self.Ea2/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cA[i])
        
        self.model.set_rhs('cA',vertcat(*rhs_cA))
        self.model.set_rhs('cB', vertcat(*rhs_cB))
        self.model.set_rhs('cR',vertcat(*rhs_cR))
        self.model.set_rhs('cS', vertcat(*rhs_cS))
        self.model.set_rhs('Tr', vertcat(*rhs_Tr))

        self.model.setup()







def print_progress(k,N, bar_len = 50):
    k = int(max(min(k,N),0))
    percent_done = round(100*(k)/(N-1))

    done = round(percent_done/(100/bar_len))
    togo = bar_len-done
    done_str = '█'*done
    togo_str = '░'*togo

    msg = f"\t Progress: [{done_str}{togo_str}] {percent_done}% done"
    print(msg, end='\r')