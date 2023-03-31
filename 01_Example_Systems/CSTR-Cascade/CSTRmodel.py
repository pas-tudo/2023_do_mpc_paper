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


    def get_model(self):

        rhs_cA=[]
        rhs_cB=[]
        rhs_cR=[]
        rhs_cS=[]
        rhs_Tr=[]

        for i in range(self.n_reac):
            r1 = self.k1_mean[i]*self.k1_var[i]*exp(-self.Ea1/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]*self.cB[i]
            r2 = self.k2_mean[i]*self.k2_var[i]*exp(-self.Ea2/(self.R_gas*(self.Tr[i]+273.15)))*self.cA[i]**2

            if i == 0:
                delCA = -self.cA[i]
                delCB = -self.cB[i]
                delCR = -self.cR[i]
                delCS = -self.cS[i]
                delTR = self.Tr_in - self.Tr[i]
            else:
                delCA = self.cA[i-1] - self.cA[i]
                delCB = self.cB[i-1] - self.cB[i]
                delCR = self.cR[i-1] - self.cR[i]
                delCS = self.cS[i-1] - self.cS[i]
                delTR = self.Tr[i-1] - self.Tr[i]

            rhs_cA.append(self.V_out/self.V_i*delCA - r1 - 2*r2 + self.uA[i]/self.V_i)
            rhs_cB.append(self.V_out/self.V_i*delCB - r1 + self.uB[i]/self.V_i)
            rhs_cR.append(self.V_out/self.V_i*delCR + r1)
            rhs_cS.append(self.V_out/self.V_i*delCS + r2)
            rhs_Tr.append(
                self.V_out/self.V_i*delTR+self.kA/(self.rho*self.cp*self.V_i)*(self.Tj[i]-self.Tr[i])
                -self.delH1_mean[i]*self.delH1_var[i]/(self.rho*self.cp)*r1-self.delH2_mean[i]*self.delH2_var[i]/(self.rho*self.cp)*r2
            )


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