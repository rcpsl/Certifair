from random import random,seed,choice
from time import time
import pickle
import math
from gurobipy import * 
import gurobipy as gp
import sys
import os
from NeuralNetwork import *
from copy import copy,deepcopy
import re
# import cdd
from utils.sample_network import *
#from volestipy import *
from functools import cmp_to_key
eps = 1E-5
np.seterr(all='raise')


class Solver():

    def __init__(self, network = None, target = -1,maxIter = 100000,property_check=None, samples = None,check_prop_samples = None, INSTR=False,convex_calls =0,MAX_DEPTH=30):
        self.maxNumberOfIterations = maxIter
        self.nn        = deepcopy(network)
        self.orig_net = deepcopy(self.nn)

        #TODO: self.__parse_network() #compute the dims of input and hidden nodes
        self.__input_dim    = self.nn.image_size
        self.__hidden_units = self.nn.num_hidden_neurons
        self.__output_dim   = self.nn.output_size
        self.num_layers     = self.nn.num_layers #including the input/output layers
        self.check_potential_CE = property_check
        
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)

        env.start()
        # Build model m here
        self.model = Model(env = env)
        self.model.params.OutputFlag = 0
        self.model.params.DualReductions = 0
        
        #Add variables
        self.state_vars         = self.model.addVars(self.__input_dim,name = "x", lb  = -1*GRB.INFINITY)  
        self.relu_vars           = self.model.addVars(self.__input_dim,name = "y", lb = -1*GRB.INFINITY)      
        self.relu_vars.update(self.model.addVars([self.__input_dim + i for i in range(self.__hidden_units)],name = "y", lb = 0))
        self.net_vars        = self.model.addVars(self.__input_dim,name = "n",lb = -1* GRB.INFINITY)      
        self.net_vars.update(self.model.addVars([self.__input_dim + i for i in range(self.__hidden_units)],name = "n", lb = -1* GRB.INFINITY) )
        self.slack_vars         = self.model.addVars(self.__input_dim + self.__hidden_units,name = "s",lb = 0)
        self.out_vars           = self.model.addVars(self.__output_dim,name = "u", lb = -1* GRB.INFINITY)
        #Variable names

        self.in_vars_names       = ['x[%d]'%i for i in range(self.__input_dim)]
        self.relu_vars_names     = ['y[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.net_vars_names      = ['n[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.slack_vars_names    = ['s[%d]'%i for i in range(self.__input_dim + self.__hidden_units)]
        self.out_vars_names      = ['u[%d]'%i for i in range(self.__output_dim)]

        self.abs2d              = [[0,i] for i in range(self.__input_dim)]
        self._2dabs              = {}
        self.fixed_relus = set()
        self.MAX_DEPTH = MAX_DEPTH
        self.samples = samples
        self.check_prop_samples = check_prop_samples
        # self.phases,self.samples_outs = self.nn.get_phases(self.samples)
        self.convex_calls = convex_calls
        self.INSTRUMENT = INSTR
        self.target = target

        #Layer index 
        self.model.update()
        self.layer_start_idx = [0] * len(self.nn.layers)
        self.layer_stats = np.zeros((self.nn.num_layers-1,2))

        idx = self.__input_dim
        for layer_idx, layer in enumerate(self.nn.layers):
            if(layer_idx == 0):
                continue
            # self.layer_stats[layer_idx] = {'undecided':0, 'infeasible':0}
            self._2dabs[layer_idx] = {}
            self.layer_start_idx[layer_idx] = self.layer_start_idx[layer_idx-1] + self.nn.layers[layer_idx-1]['num_nodes']
            for neuron_idx in range(layer['num_nodes']):
                self.abs2d += [[layer_idx,neuron_idx]]
                self._2dabs[layer_idx][neuron_idx] = idx
                idx+=1
        self.linear_constraints = []

        # if(target != -1):
        #     outs = self.out_vars.values()
        #     decision_var = self.model.addVar(name = 'd')
        #     self.model.addConstr(decision_var == max_(outs[:target] + outs[target+1:]))
        #     self.model.addConstr(decision_var >= 0)

    def add_linear_constraints(self, A, x, b, sense = GRB.LESS_EQUAL):
        #Senses are GRB.LESS_EQUAL, GRB.EQUAL, or GRB.GREATER_EQUAL
        for row in range(len(b)):
            # linear_expression = LinExpr(A[row],x)
            constraint = {'A' : A[row], 'x' : x, 'sense': sense,'rhs': b[row]} 
            self.linear_constraints.append(constraint)

    def __add_NN_constraints(self,model, nn):
        fixed_relus = 0
        #First layer of network is assumed to be the input to the network
        layer_idx = 0
        num_neurons = nn.layers[layer_idx]['num_nodes']
        layer_start_idx = self.layer_start_idx[layer_idx]
        state_vars = [model.getVarByName(var_name) for var_name in self.in_vars_names]
        out_vars = [model.getVarByName(var_name) for var_name in self.out_vars_names]
        relu_vars  = [model.getVarByName(var_name) for var_name in self.relu_vars_names]
        net_vars   = [model.getVarByName(var_name) for var_name in self.net_vars_names]
        slack_vars = [model.getVarByName(var_name) for var_name in self.slack_vars_names]
        for neuron_idx in range(num_neurons):
            neuron_abs_idx = layer_start_idx + neuron_idx
            model.addConstr(relu_vars[neuron_abs_idx] == state_vars[neuron_abs_idx])
            model.addConstr(net_vars[neuron_abs_idx]  == state_vars[neuron_abs_idx])
        for layer_idx in range(1,nn.num_layers): #exclude input
            num_neurons = nn.layers[layer_idx]['num_nodes']
            layer_start_idx = self.layer_start_idx[layer_idx]
            prev_layer_start_idx = self.layer_start_idx[layer_idx - 1]
            W = nn.layers[layer_idx]['weights']
            b = nn.layers[layer_idx]['bias']
            lb = nn.layers[layer_idx]['conc_lb']
            ub = nn.layers[layer_idx]['conc_ub']
            in_lb = nn.layers[layer_idx]['in_lb']
            in_ub = nn.layers[layer_idx]['in_ub']

            prev_layer_size = nn.layers_sizes[layer_idx -1]
            prev_relus = [relu_vars[prev_layer_start_idx + input_idx] for input_idx in range(prev_layer_size)]
            for neuron_idx in range(num_neurons):
                #add - constraints
                neuron_abs_idx = layer_start_idx + neuron_idx
                net_expr = LinExpr(W[neuron_idx], prev_relus)
                if(nn.layers[layer_idx]['type'] != 'output'):
                    model.addConstr(net_vars[neuron_abs_idx] == (net_expr + b[neuron_idx]))
                    model.addConstr(slack_vars[neuron_abs_idx] == relu_vars[neuron_abs_idx] - net_vars[neuron_abs_idx])

                    if(ub[neuron_idx] <= 0):
                        model.addConstr(relu_vars[neuron_abs_idx] == 0, name= "%d_inactive"%neuron_abs_idx)
                        fixed_relus +=1
                    elif(in_lb[neuron_idx] > 0):
                        model.addConstr(slack_vars[neuron_abs_idx] == 0, name= "%d_active"%neuron_abs_idx)
                        fixed_relus +=1
                    else:
                        factor = (in_ub[neuron_idx]/ (in_ub[neuron_idx]-in_lb[neuron_idx]))[0]
                        model.addConstr(relu_vars[neuron_abs_idx] <= factor * (net_vars[neuron_abs_idx]- in_lb[neuron_idx]),name="%d_relaxed"%neuron_abs_idx)
                        A_up = nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                        model.addConstr(LinExpr(A_up[:-1],state_vars)  + A_up[-1]  >= relu_vars[neuron_abs_idx],name= "%d_sym_UB"%neuron_abs_idx)
            
                else:
                    model.addConstr(out_vars[neuron_idx] == (net_expr + b[neuron_idx]))
                    model.addConstr(out_vars[neuron_idx] >= lb[neuron_idx],name = "out_%d_LB"%neuron_idx)
                    model.addConstr(out_vars[neuron_idx] <= ub[neuron_idx],name = "out_%d_UB"%neuron_idx)
                    A_up = nn.layers[layer_idx]['Relu_sym'].upper[neuron_idx]
                    A_low = nn.layers[layer_idx]['Relu_sym'].lower[neuron_idx]
                    model.addConstr(LinExpr(A_up[:-1],state_vars)  + A_up[-1]  >= out_vars[neuron_idx],name = "out_%d_sym_UB"%neuron_idx)
                    model.addConstr(LinExpr(A_low[:-1],state_vars)  + A_low[-1]  <= out_vars[neuron_idx],name = "out_%d_sym_LB"%neuron_idx)

                
        # print('Number of fixed Relus:', len(self.fixed_relus))
    def __create_init_model(self):
        env = gp.Env(empty=True)
        env.setParam('OutputFlag', 0)
        env.setParam('DualReductions', 0)
        env.start()
        model = Model(env = env)
        model.params.OutputFlag = 0
        model.params.DualReductions = 0
        model.addVars(self.__input_dim,name = self.in_vars_names, lb  = -1*GRB.INFINITY)  
        model.addVars(self.__input_dim,name = self.relu_vars_names[:self.__input_dim], lb = -1*GRB.INFINITY)      
        model.addVars(self.__hidden_units,name = self.relu_vars_names[self.__input_dim:], lb = 0)
        model.addVars(self.__input_dim + self.__hidden_units,name = self.net_vars_names ,lb = -1* GRB.INFINITY)      
        model.addVars(self.__input_dim + self.__hidden_units,name = self.slack_vars_names,lb = 0)
        model.addVars(self.__output_dim,name = self.out_vars_names, lb = -1* GRB.INFINITY)
        model.update()

        return model

    def solve(self):
        
        #Create initial model
        model = self.__create_init_model()
        self.__prepare_problem(model,self.nn)
        # self.model.write('model.lp')
        self.convex_calls +=1
        if(self.INSTRUMENT):
            print('Instrumenting a new instance')
        model.optimize()
        if(model.Status == 3): #Infeasible
            # IIS_slack = []
            # try:
            #     model.computeIIS() 
            #     fname = 'result.ilp'
            #     model.write(fname)
            # except Exception as e:
            #     print(e)
            status = 'UNSAT'
            return None,status
        else:   
            status = 'UNKNOWN'
            SAT,infeasible_relus = self.check_SAT(model) 
            # for relu_idx, phase in infeasible_relus:
            #     if(relu_idx < 55):
            #         if(phase):
            #             model.addConstr(model.getVarByName('s[%d]'%relu_idx) == 0)
            #         else:
            #             model.addConstr(model.getVarByName('y[%d]'%relu_idx) == 0)
            # model.update()
            x= None
            u = None
            if(SAT):
                # print('Solution found')
                x = [model.getVarByName(var_name).X for var_name in self.in_vars_names]
                u = [model.getVarByName(var_name).X for var_name in self.out_vars_names]
                status = 'SolFound'  
            else:
                status = 'UNKNOWN'
                layers_masks = []
                for layer_idx,layer in enumerate(self.nn.layers):
                    if(layer_idx < 1):
                        continue
                    layers_masks += [-1*np.ones((layer['num_nodes'],1))]
                for l,n in self.nn.active_relus:
                    layers_masks[l-1][n] = 1
                for l,n in self.nn.inactive_relus:
                    layers_masks[l-1][n] = 0
                non_lin_relus = [self._2dabs[l][n] for l,n in self.nn.nonlin_relus]

                paths = [1]
                status = self.dfs(model, deepcopy(self.nn), infeasible_relus,[],layers_masks,undecided_relus=copy(sorted(non_lin_relus)),paths = paths)
                # print(self.layer_stats[0])
                # print(status)
                # print('Paths:',paths)

        
        return (x,u),status

    def fix_relu(self, model, nn, fixed_relus):
        input_vars = [model.getVarByName(var_name) for var_name in self.in_vars_names]
        for relu_idx, phase in fixed_relus[-1:]:
            layer_idx,neuron_idx = self.abs2d[relu_idx]
            A_up = nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            A_low = A_up
            slack_var = model.getVarByName(self.slack_vars_names[relu_idx])
            relu_var  = model.getVarByName(self.relu_vars_names[relu_idx])
            if(phase == 1):
                model.addConstr(slack_var == 0,name="%d_active"%relu_idx)
                model.addConstr(LinExpr(A_low[:-1],input_vars) + A_low[-1] == relu_var,name ="y%d_active_LB"%relu_idx)
                model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  >= 0,name ="y%d_active_LB"%relu_idx)
            else:
                model.addConstr(relu_var == 0,name="%d_inactive"%relu_idx)
                model.addConstr(LinExpr(A_up[:-1],input_vars)  + A_up[-1]  <= 0,name ="y%d_inactive_UB"%relu_idx)
        
        # self.add_objective([idx for idx,_ in fixed_relus])

    def update_in_interval(self, nn):
        H_rep = np.zeros((0,nn.image_size +1 ))
        for layer_idx, neuron_idx in nn.active_relus:
            eq = nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
            b,A = -eq[-1], eq[:-1]
            H_rep = np.vstack((H_rep,np.hstack((-b,A))))
        try:
            for layer_idx, neuron_idx in nn.inactive_relus:
                eq = nn.layers[layer_idx]['in_sym'].upper[neuron_idx]
                b,A = -eq[-1], eq[:-1]
                H_rep = np.concatenate((H_rep,np.hstack((b,-A)).reshape((1,6))),axis = 0)
                self.MAX_DEPTH = 2

            A = cdd.Matrix(H_rep)
            A.rep_type = 1
            p = cdd.Polyhedron(A)
        
            vertices = np.array(p.get_generators())[:,1:]
            hrect_min = np.min(vertices,axis = 0).reshape((-1,1))
            hrect_max = np.max(vertices,axis = 0).reshape((-1,1))
            new_bound = np.hstack((hrect_min,hrect_max))
            new_bound[:,1] = np.minimum(new_bound[:,1],self.orig_net.input_bound[:,1])
            new_bound[:,0] = np.maximum(new_bound[:,0],self.orig_net.input_bound[:,0])
        except Exception as e:
            new_bound = nn.input_bound       

        return new_bound

    def set_neuron_bounds(self,model,nn, layer_idx,neuron_idx,phase,layers_masks,bounds = None):
        if(phase == 0):
            layers_masks[layer_idx-1][neuron_idx] = 0
            # self.nn.update_bounds(layer_idx,neuron_idx,[np.array(0),np.array(0)],layers_masks)
        elif(phase == 1):
            layers_masks[layer_idx-1][neuron_idx] = 1
            # self.nn.update_bounds(layer_idx,neuron_idx,bounds,layers_masks)
            
        else:
            layers_masks[layer_idx-1][neuron_idx] = -1

        nn.recompute_bounds(layers_masks)
        # bounds = self.update_in_interval(nn)
        # nn.input_bound = bounds
        # nn.recompute_bounds(layers_masks)
        # nn.input_bound = copy(self.orig_net.input_bound)

        self.fix_after_propgt(model,nn)

    def getIIS(self,fname):
        IIS = []
        self.model.computeIIS()
        fname = 'result1.ilp'
        self.model.write(fname)
        with open(fname, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if('B:' in line):
                    IIS.append(int(line.strip().split('_')[0][1:]))
        return IIS  

        
    def fix_after_propgt(self,model,nn):
        
        fixed_relus  = [(self._2dabs[layer_idx][relu_idx],1) for layer_idx,relu_idx in nn.active_relus] 
        fixed_relus += [(self._2dabs[layer_idx][relu_idx],0) for layer_idx,relu_idx in nn.inactive_relus]

        for relu_idx,phase in fixed_relus:
            if(phase == 1 and model.getConstrByName("%d_active"%relu_idx) is None):
                model.addConstr(model.getVarByName(self.slack_vars_names[relu_idx]) == 0,name = "%d_active"%relu_idx)
            elif(phase == 0 and model.getConstrByName("%d_inactive"%relu_idx) is None):
                model.addConstr(model.getVarByName(self.relu_vars_names[relu_idx]) == 0, name = "%d_inactive"%relu_idx)
        in_vars  = [model.getVarByName(var_name) for var_name in self.in_vars_names]
        for l_idx, relu_idx in nn.nonlin_relus:
            abs_idx = self._2dabs[l_idx][relu_idx]
            relu_var = model.getVarByName(self.relu_vars_names[abs_idx])
            net_var  = model.getVarByName(self.net_vars_names[abs_idx])
            in_ub = nn.layers[l_idx]['in_ub'][relu_idx]
            in_lb = nn.layers[l_idx]['in_lb'][relu_idx]
            # L_ub = nn.layers[l_idx]['L_ub'][relu_idx]
            # L_lb = nn.layers[l_idx]['L_lb'][relu_idx]
            if(in_lb < 0 and in_ub > 0):
                if(model.getConstrByName("%d_relaxed"%abs_idx) is not None):
                    model.remove(model.getConstrByName("%d_relaxed"%abs_idx))
                    model.remove(model.getConstrByName("%d_sym_UB"%abs_idx))
                # if(model.getConstrByName("%d_relaxed_L"%abs_idx) is not None):
                #     model.remove(model.getConstrByName("%d_relaxed_L"%abs_idx))    

                A_up = nn.layers[l_idx]['Relu_sym'].upper[relu_idx]
                model.addConstr(LinExpr(A_up[:-1],in_vars)  + A_up[-1]  >= relu_var ,name= "%d_sym_UB"%abs_idx)

                factor = (in_ub/ (in_ub-in_lb))[0]
                model.addConstr(relu_var <= factor * (net_var- in_lb),name="%d_relaxed"%abs_idx)

                # factor = (L_ub/ (L_ub-L_lb) )
                # model.addConstr(relu_var <= factor * (net_var- L_lb),name="%d_relaxed_L"%abs_idx)

        for neuron_idx in range(self.__output_dim):
            out_var = model.getVarByName(self.out_vars_names[neuron_idx])
            lb = nn.layers[nn.num_layers-1]['in_lb'][neuron_idx]
            ub = nn.layers[nn.num_layers-1]['in_ub'][neuron_idx]
            # L_ub = nn.layers[nn.num_layers-1]['L_ub'][neuron_idx]
            # L_lb = nn.layers[nn.num_layers-1]['L_lb'][neuron_idx]
            model.remove(model.getConstrByName("out_%d_sym_UB"%neuron_idx))
            model.remove(model.getConstrByName("out_%d_sym_LB"%neuron_idx))
            model.remove(model.getConstrByName("out_%d_LB"%neuron_idx))
            model.remove(model.getConstrByName("out_%d_UB"%neuron_idx))
                    
            model.addConstr(out_var >= lb,name = "out_%d_LB"%neuron_idx)
            model.addConstr(out_var <= ub,name = "out_%d_UB"%neuron_idx)
            A_up = nn.layers[nn.num_layers-1]['Relu_sym'].upper[neuron_idx]
            A_low = nn.layers[nn.num_layers-1]['Relu_sym'].lower[neuron_idx]
            model.addConstr(LinExpr(A_up[:-1],in_vars)  + A_up[-1]  >= out_var, name = "out_%d_sym_UB"%neuron_idx)
            model.addConstr(LinExpr(A_low[:-1],in_vars)  + A_low[-1]  <= out_var,name = "out_%d_sym_LB"%neuron_idx)
            if(model.getConstrByName("out_%d_L_LB"%neuron_idx) is not None):
                    model.remove(model.getConstrByName("out_%d_L_LB"%neuron_idx))    
                    model.remove(model.getConstrByName("out_%d_L_UB"%neuron_idx))
            # model.addConstr(out_var >= L_lb,name = "out_%d_L__LB"%neuron_idx)
            # model.addConstr(out_var <= L_ub,name = "out_%d_L__UB"%neuron_idx)
                       
    def test_decision_validity(self,nn,fixed_relus):
        A = np.vstack((np.eye(nn.image_size),-np.eye(nn.image_size)))
        b = np.vstack((nn.input_bound[:,1].reshape((-1,1)),-nn.input_bound[:,0].reshape((-1,1))))
        for relu_idx,phase in fixed_relus:
            l_idx,n_idx = self.abs2d[relu_idx]
            # samples_idxs = np.where(self.phases[l_idx-1][samples_idxs,n_idx]  == phase)[0]
            eq = nn.layers[l_idx]['in_sym'].upper[n_idx]
            W = eq[:-1]
            c = -eq[-1]
            if(phase == 1):
                W = -W
                c = -c
            A = np.vstack((A,W))
            b = np.vstack((b,c))

        p = HPolytope(A,b.flatten())
        samples = p.generate_samples(walk_len=5, n_samples=5, seed=42)
        if(not np.all(np.isfinite(samples))):
            return False
        valid = (np.sum(p.A.dot(samples.T) - p.b.reshape((-1,1)) > 0) == 0)
        return valid
    def dfs(self, model, nn, infeasible_relus,fixed_relus,layers_masks, depth = 0,undecided_relus = [],paths = 0):

        s = time()        
        status = 'UNKNOWN'
        if(depth>self.MAX_DEPTH):
            return
        relu_idx,phase =  infeasible_relus[0]
        nonlin_relus = copy(undecided_relus)
        min_layer,_ = self.abs2d[nonlin_relus[0]]
        layer_idx,neuron_idx = self.abs2d[relu_idx]
        if(layer_idx > min_layer):
            relu_idx,phase = nonlin_relus[0],int(model.getVarByName('n[%d]'%nonlin_relus[0]).X >=0)
            layer_idx,neuron_idx = self.abs2d[relu_idx]

        nonlin_relus.remove(relu_idx)
        # print('DFS:',depth,"Setting neuron %d to %d"%(relu_idx,phase))
        layers_masks = deepcopy(layers_masks)
        network = deepcopy(nn)
        model.update()
        model1 = model.copy()
        # print('Prep Problem',time() - s)
        fixed_relus.append([relu_idx,phase])
        # SAT = self.compute_bounds_L(network,model1,fixed_relus)
        # if(SAT):
        #     print('solver found CE using samples')
        #     status = 'SolFound'
        #     return
        # if(network.layers[network.num_layers-1]['L_ub'] is None):
        #     return
        self.set_neuron_bounds(model1,network,layer_idx,neuron_idx,phase,layers_masks)

        self.fix_relu(model1, network, fixed_relus)
        # print('time of iteration',time() - s)
        if(self.INSTRUMENT):
            # print('Neurons fixed by solver:',self.nn.num_hidden_neurons - len(infeasible_relus),', Convex calls:',self.convex_calls.val)
            fixed = self.nn.num_hidden_neurons - len(infeasible_relus)
            print(len(infeasible_relus))
            ratio = fixed/ self.nn.num_hidden_neurons
            model_temp = model1.copy()
            model_temp.setObjective(1)
            model_temp.optimize()
            _,non_fixed = self.check_SAT(model_temp)
            print('ratio1:',ratio,'ratio2:',(self.nn.num_hidden_neurons-len(non_fixed))/self.nn.num_hidden_neurons)
        model1.optimize()
        if(model1.Status != 3): #Feasible solution
            self.layer_stats[layer_idx-1][0] +=  1
            SAT,infeasible_set = self.check_SAT(model1)
            valid = self.check_potential_CE(network, np.array([model1.getVarByName(var_name).X for var_name in self.in_vars_names]).reshape((-1,1)),self.target)
            if(SAT or valid):
                #print('Solution found')
                status = 'SolFound'  
            else:
                status = self.dfs(model1, network, infeasible_set,copy(fixed_relus),layers_masks,depth+1,nonlin_relus,paths)
        else:
            self.layer_stats[layer_idx-1][1] += 1
        if(status != 'SolFound'):
            paths[0] += 1 
            # if(self.model.Status == 3):
            #    IIS = self.getIIS('result1.ilp')
            #    if(len(IIS) and relu_idx != IIS[-1] and IIS[-1] in [n_idx for n_idx,_ in fixed_relus]):
            #        self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
            #        return status
            model.update()
            model1 = model.copy()
            network = deepcopy(nn)
            phase = 1 - phase
            # print('Backtrack, Setting neuron %d to %d'%(relu_idx,phase))
            fixed_relus[-1] = [relu_idx,phase]
            # SAT = self.compute_bounds_L(network,model1,fixed_relus)
            # if(SAT):
            #     print('solver found CE using samples')
            #     status = 'SolFound'
            #     return
            # if(network.layers[network.num_layers-1]['L_ub'] is None):
            #     return
            self.set_neuron_bounds(model1, network, layer_idx,neuron_idx,phase,layers_masks)
            #valid = self.test_decision_validity(network,fixed_relus)
            # self.__prepare_problem()

            self.fix_relu(model1,network,fixed_relus)
            # if(self.INSTRUMENT):
            #     print('Neurons fixed by solver:',self.nn.num_hidden_neurons - len(infeasible_relus),', Convex calls:',self.convex_calls.val)
            model1.optimize()
            if(model1.Status != 3): #Feasible solution
                self.layer_stats[layer_idx-1][0] += 1
                SAT,infeasible_set = self.check_SAT(model1)
                valid = self.check_potential_CE(network, np.array([model1.getVarByName(var_name).X for var_name in self.in_vars_names]).reshape((-1,1)),self.target)
                if(SAT or valid):
                    #print('Solution found')
                    status = 'SolFound'  
                else:
                    status = self.dfs(model1, network, infeasible_set,copy(fixed_relus),layers_masks,depth+1,nonlin_relus,paths)
            else:
                status = 'UNSAT'
                self.layer_stats[layer_idx-1][1] += 1

            #if(status != 'SolFound'):
            #    status = 'UNSAT'
        
            # self.set_neuron_bounds(layer_idx,neuron_idx,-1,layers_masks)
        
        return status
            



    def check_SAT(self,model):   
        y = np.array([model.getVarByName(var_name).X for var_name in self.relu_vars_names[self.__input_dim:]])
        net = np.array([model.getVarByName(var_name).X for var_name in self.net_vars_names[self.__input_dim:]])
        slacks = np.zeros_like(y)
        active_infeas = ((y-net) > eps) * (net > eps) #if y>net in net>0 domain
        inactive_infeas =  ((y > eps) * (net < eps))    #if y > 0 in net<0 domain
        active  = np.sort(np.where(active_infeas == True)[0])
        inactive = np.sort(np.where(inactive_infeas == True)[0])
        slacks[active] = y[active] - net[active]
        slacks[inactive] = y[inactive]
        layer_slacks = []
        for idx in active:
            abs_idx = self.__input_dim + idx
            layer,_ = self.abs2d[abs_idx]
            layer_slacks.append((layer,slacks[idx],abs_idx,1))
        for idx in inactive:
            abs_idx = self.__input_dim + idx
            layer,_ = self.abs2d[abs_idx]
            layer_slacks.append((layer,slacks[idx],abs_idx,0))

        def compare(l1,l2):
            if(l1[0] < l2[0]):
                return -1
            if(l2[0] < l1[0]):
                return 1
            if(l1[2] < l2[2]):
                return -1
            else:
                return 1
            #return choice([-1,1])
            
        layer_slacks = sorted(layer_slacks,key = cmp_to_key(compare))
        if(len(layer_slacks) != 0):
            infeas_relus = [(idx,phase) for _,_,idx,phase in layer_slacks]
            return False, infeas_relus
        return True, None

        offset = 0
        infeas_relus=[]
       
        active = list(np.where(active_infeas == True)[0] + self.__input_dim)
        inactive = list(np.where(inactive_infeas == True)[0] + self.__input_dim)            
        infeas_relus = [(n_idx,0) for n_idx in inactive]
        infeas_relus +=  [(n_idx,1) for n_idx in active]
        infeas_relus = sorted(infeas_relus)
        if(len(infeas_relus) != 0):
            infeas_relus = [(idx,phase) for idx,phase in infeas_relus]
            return False, infeas_relus
        return True, None

    def __prepare_problem(self,model, nn):
        #clear all constraints
        # self.model.remove(self.model.getConstrs())
        #Add external convex constraints
        for constraint in self.linear_constraints:
            vars = [model.getVarByName(var_name) for var_name in constraint['x']]
            model.addConstr(LinExpr(constraint['A'],vars), sense = constraint['sense'], rhs = constraint['rhs'])

        self.__add_NN_constraints(model, nn)
        self.add_objective(model, [])

    

    def add_objective(self,model, fixed_relus = None):
        slacks = [model.getVarByName(var_name) for var_name in self.slack_vars_names[self.__input_dim:]]
        init_weight = 1E-10
        weights = []
        for layer_idx,layer_size in enumerate(self.nn.layers_sizes[1:-1]):
            ub = np.maximum(0,self.nn.layers[layer_idx+1]['in_ub'])
            ub[ub > 0] = 1
            weights += list(init_weight * ub)
            #weights += [1] * layer_size
            init_weight *= 10000

        obj = LinExpr()
        if(fixed_relus):
            for idx in fixed_relus:
                weights[idx - self.__input_dim] = 0

        obj.addTerms(weights,slacks)
        model.setObjective(obj)
        model.update()


        
    
# layers_sizes = [2,3,1]
# image_size = layers_sizes[0]
# x = np.zeros((2,1))
# bounds = np.concatenate((x,x),axis = 1)
# nn = NeuralNetworkStruct(layers_sizes,input_bounds = bounds)
# solver = Solver(network = nn)
# A = np.eye(2)
# b = np.zeros(2)
# state_vars = [solver.state_vars[0],solver.state_vars[1]]
# solver.add_linear_constraints(A,state_vars,b,LpConstraintEQ)
# A = [[1, 0], [-1, 0], [0, 1], [0, -1]]
# b = [1,-0.1,1,-0.1]
# state_vars = [solver.state_vars[0],solver.state_vars[1]]
# solver.add_linear_constraints(A,state_vars,b)
# state_vars = [solver.out_vars[0]]
# A, b = [[-1]],[-0.1]
# solver.add_linear_constraints(A, state_vars, b)
# solver.solve()

# e = 0.1
# layers_sizes = [1,2,1]
# image_size = layers_sizes[0]
# bounds = np.zeros((1,2))
# bounds[:,1] = 1
# nn = NeuralNetworkStruct(layers_sizes,input_bounds = bounds)
# Weights= [np.concatenate((np.array([-1]),np.array([1])),axis = 0).reshape((2,1))]
# Weights.append(np.concatenate((np.array([[1],[1]])),axis = 0).reshape((1,2)))
# biases = [np.array([e,e-1]),np.zeros(2)]
# nn.set_weights(Weights,biases)
# solver = Solver(network = nn)
# state_vars = [solver.state_vars[0]]
# A, b = [[1],[-1]],[1,0]
# solver.add_linear_constraints(A, state_vars, b)
# state_vars = [solver.out_vars[0]]
# A, b = [[1],[-1]],[e,-e/2]
# solver.add_linear_constraints(A, state_vars, b)
# solver.solve()
