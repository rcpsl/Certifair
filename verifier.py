import sys,os
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
os.environ['OPENBLAS_NUM_THREADS']="1"

import torch
from solver import *
from time import time,sleep
from random import random, seed
import signal
import numpy as np
import glob
from NeuralNetwork import PytorchNN
from multiprocessing import Process, Value, Queue, JoinableQueue
import queue
import argparse
from os import path
import itertools
import tensorflow as tf
from intervals.symbolic_interval import SymbolicInterval
from property import AdultProperty, BoxProperty, CompasProperty, GermanProperty, LawProperty
from datasets.utils import Datasets
eps = 1E-10
SEED = 5
torch.manual_seed(SEED)
np.random.seed(SEED)
from tqdm import tqdm
class TimeOutException(Exception):
    def __init__(self, *args, **kwargs):
        pass
    def __str__(self):
        return 'TimeoutException'

def print_summary(network,prop, safety, time, extra = None):
    if(extra is not None):
        print(network[14:19],prop,safety,time,extra)
    else:
        print(network[14:19],prop,safety,time)

def alarm_handler(signum, frame):
    raise TimeOutException()

def check_property(network,x,target):
    res = network.evaluate(x)
    if(res[0] > 0 and res[1] < 0):
        # print('Potential CE succeeded:')
        # print('Individual 1:', decode_features(x[:30]) , res[0])
        # print('Individual 2:', decode_features(x[30:]) , res[1])
        return True
    # u = network.evaluate(x)
    # if(np.argmax(u) != target):
    #     # print("Potential CE succeeded")
    #     return True
    return False

def check_prop_samples(nn,samples,target):
    outs = nn.evaluate(samples.T).T
    outs = np.argmax(outs,axis = 1)
    return np.any(outs  != target)


def worker(name,job_queue : Queue, result_queue: JoinableQueue):

    while True:
        try:
            args = job_queue.get()
        except Exception as e:
            # print(e)
            print(f"Worker {name} exiting" )
            return
        else:
            try:
                network, input_bounds, check_property, property, TIMEOUT, task_id = args
                signal.signal(signal.SIGALRM, alarm_handler)
                signal.alarm(TIMEOUT)
                result,status = run_instance(network, input_bounds, check_property, property= property)
                result_queue.put(status)
            except TimeOutException as e:
                print(e)
                result_queue.put('Timeout')
            

    


def run_instance(network, input_bounds, check_property, property : BoxProperty, target = None):
    if network.layers[-1]['conc_lb'][1] > 0:
        return None, 'UNSAT'
    try:
        solver = Solver(network = network,property_check=check_property,target = target)
        # input_vars = [solver.state_vars[i] for i in range(len(solver.state_vars))]
        
        A = np.eye(network.image_size)
        lower_bound = input_bounds[:,0]
        upper_bound = input_bounds[:,1]
        solver.add_linear_constraints(A,solver.in_vars_names,lower_bound,GRB.GREATER_EQUAL)
        solver.add_linear_constraints(A,solver.in_vars_names,upper_bound,GRB.LESS_EQUAL)

        delta = property.delta
        n_features = len(input_bounds)//2
        #continious variables vary by delta
        cont_idices = property.numerical_features_idx
        for idx in cont_idices:
            A = np.zeros(len(input_bounds))
            A[idx] = 1
            A[n_features + idx] = -1
            
            solver.add_linear_constraints([A],solver.in_vars_names,[delta],GRB.LESS_EQUAL)
            solver.add_linear_constraints([A],solver.in_vars_names,[-delta],GRB.GREATER_EQUAL)
        #discrete values are equal except gender
        disc_indices = property.discrete_features_idx
        for idx in disc_indices:
            A = np.zeros(len(input_bounds))
            A[idx] = 1
            A[n_features + idx] = -1
            b = [0]
            solver.add_linear_constraints([A],solver.in_vars_names,b,GRB.EQUAL)

        output_vars = [solver.out_vars[i] for i in range(len(solver.out_vars))]
        A = np.zeros((network.output_size,network.output_size))
        A[0][0] = 1
        A[1][1] = -1
        b = [0.01, 0.01]
        solver.add_linear_constraints(A,solver.out_vars_names,b,GRB.GREATER_EQUAL)
        
        solver.preprocessing = False
        vars,status = solver.solve()
        # if(status == 'SolFound'):
        #     adv_found.value = 1
        return vars, status
        # print('Terminated')
    except Exception as e:
        raise e

#Features
#['age', 'education_num', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 
# 'workclass=Government', 'workclass=Other/Unknown', 'workclass=Private', 'workclass=Self-Employed', 
# 'education=Assoc', 'education=Bachelors', 'education=Doctorate', 'education=HS-grad', 'education=Masters', 'education=Prof-school', 'education=School', 'education=Some-college', 
# 'marital_status=Divorced', 'marital_status=Married', 'marital_status=Separated', 'marital_status=Single', 'marital_status=Widowed', 
# 'occupation=Blue-Collar', 'occupation=Other/Unknown', 'occupation=Professional', 'occupation=Sales', 'occupation=Service', 'occupation=White-Collar']
scale = np.array([9.0000e+01, 1.6000e+01, 1.0000e+00, 1.0000e+00, 9.9999e+04,
       4.3560e+03, 9.9000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00,
       1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
       1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
       1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
       1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00])

def decode_features(features):
    feature_dict = {}
    feature_dict['age'] = features[0].item() * scale[0]
    feature_dict['education_num'] = features[1].item() * scale[1]
    feature_dict['race'] = 'White' if features[2] == 1 else 'Not White'
    feature_dict['sex'] = 'Male' if features[3] == 1.0 else 'Female'
    feature_dict['capital_gain'] = features[4].item() * scale[4]
    feature_dict['capital_loss'] = features[5].item() * scale[5]
    feature_dict['hours_per_week'] = features[6].item() * scale[6]
    workclass_cat = ['Government', 'Other/Unknown', 'Private','Self-Employed']
    feature_dict['workclass'] =  workclass_cat[np.where(features[7:11] == 1)[0].item()]
    education_cat = ['Assoc','Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Prof-school', 'School', 'Some-college']
    feature_dict['education'] =  education_cat[np.where(features[11:19] == 1)[0].item()]
    marital_status_cat = ['Divorced', 'Married', 'Seperated', 'Single', 'Widowed']
    feature_dict['marital_status'] = marital_status_cat[np.where(features[19:24] == 1)[0].item()]
    occupation_cat = ['Blue-Collar', 'Other/Unkown', 'Professional', 'Sales', 'Service', 'White-Collar']
    feature_dict['occupation'] = occupation_cat[np.where(features[24:] == 1)[0].item()]
    return feature_dict

def uniform_partition(x, num_dim_partitions=2):
    
    n = x.shape[0]
    partitions = []
    for dim in range(n):
        points = np.linspace(x[dim,0], x[dim,1], num_dim_partitions +1)
        dim_partitions = []
        for i in range(len(points) - 1):
            dim_partitions.append([points[i],points[i+1]])
        partitions.append(dim_partitions)
    ret = itertools.product(*partitions)
    return ret

def generate_all_combs(cats):
   
    one_hot_vals = [list(range(v)) for v in cats]
    power_set = itertools.product(*one_hot_vals)
    # result = []
    # for item in power_set:
    #     comb = []
    #     for i,val in enumerate(item):
    #         if cats[i] == 2: #binary
    #             comb += [val]
    #         else:
    #             comb += list(one_hot(val,cats[i]))
    #     result.append(comb)
    return power_set
def getDiscreteEmbedding(comb, cat_featues_len):
    embedding = []
    for val, max_val in zip(comb,cat_featues_len):
        if(max_val == 2): 
            embedding += [val]
        else:
            embedding += tf.keras.utils.to_categorical(val,max_val).tolist()
    return np.array(embedding)
        
def main(args):
    #Parse args
    nnet = args.network
    # nn = KerasNN()
    nn = PytorchNN()
    nn.parse_network_aux(nnet)
    PropertyClass = None
    if args.dataset in Datasets.property_cls:
        PropertyClass = Datasets.property_cls[args.dataset]
    else:
        raise NotImplementedError("PropertyClass not implemented")
        
    property = getattr(PropertyClass, args.property)()
    # property = PropertyClass.property4()
    p_bounds = property.input_bounds
    delta = property.delta
    TIMEOUT = int(args.timeout)
    MAX_DEPTH = int(args.max_depth)
    #Init NN structure
    # cat_featues_len =[2,4, 8, 5, 6] 
    # cont_idx = np.array([0,1,4,5,6])
    # gender_idx = 3
    # n_features = 30
    cat_featues_len = PropertyClass.cat_features_len
    cont_idx = np.array(property.numerical_features_idx)
    disc_idx = np.array(property.discrete_features_idx)
    sens_idx = property.sensitive_attr_idx
    n_features = property.num_features
    n_cont = 7
    input_bounds = np.zeros((nn.image_size,2))
    input_bounds[cont_idx] = p_bounds[cont_idx]
    input_bounds[n_features + cont_idx] = p_bounds[cont_idx]
    input_bounds[sens_idx,:] = [1.0,1.0] #First input is majority group
    #Second input is always a minority attribute (ex. gender = female = 0)
    input_bounds[n_features+sens_idx] = [0.0,0.0]
    comb = [np.random.randint(low) for low in cat_featues_len]
    power_set = list(generate_all_combs(cat_featues_len))
    np.random.shuffle(power_set)
    fair = 0
    total = 0
    jobs_Q = Queue()
    result_Q = JoinableQueue()
    for idx,comb in enumerate(power_set):
        total += 1
        disc_val = getDiscreteEmbedding(comb, cat_featues_len).reshape((-1,1))

        input_bounds[disc_idx] = disc_val
        input_bounds[n_features + disc_idx,:] = disc_val
      
        bounds = np.copy(input_bounds)
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(TIMEOUT)
        try:
            start_time = time()
            # print('Norm:',delta)
            #Solve the problem for each other output
            # input_bounds = np.concatenate((lb,ub),axis = 1)
            # samples = sample_network(nn,input_bounds,15000)
            # SAT = check_prop_samples(nn,samples,target)
            # if(SAT):
            # #    adv +=1
            #    print_summary(nnet,img_name,'unsafe',time()-start_time)
            #    return
            network = deepcopy(nn)
            network.set_bounds(bounds)
            jobs_Q.put((network, bounds, check_property, property, TIMEOUT,idx))
            # result,status = run_instance(network, bounds, check_property, property= property)
            # if(status != 'SolFound'):
            #     # print(p_idx,idx, "FAIR partition")
            #     pass
            # else:
            #     # print(idx, "UNFAIR")
            #     # break
            #     pass
            #signal.alarm(0)

        except Exception as e:
            print(e)
        # except TimeOutException:
        #     # print_summary(nnet,"",'timeout',TIMEOUT)
        #     print(idx, 'Timeout')
        # if status != 'SolFound':
        #     # print(idx, 'FAIR')
        #     fair +=1
    
    n_cores = args.cores 
    processes = []
    num_tasks = len(power_set)
    for i in range(n_cores):
        p = Process(target = worker, args = (i, jobs_Q, result_Q))
        processes.append(p)
        processes[-1].start()
    # for p in processes:
    #     p.join()
    
    # prev_n_alive = -1
    # while(any(p.is_alive() for p in processes)):
    #     sleep(5)
    #     n_alive = np.sum([p.is_alive() for p in processes])
    #     if(n_alive != prev_n_alive):
    #         prev_n_alive = n_alive
    #         print(f"Progress:{100* result_Q.qsize() / num_tasks:.4f}%")
    #         # print(jobs_Q.qsize(), result_Q.qsize())
    
    results = []
    pbar = tqdm(total = num_tasks)
    old_progress =0
    while(len(results) != num_tasks):
        try:
            results.append(result_Q.get())
            pbar.update(len(results) - old_progress)
            old_progress = len(results)
        except Exception as e:
            sleep(5)
    pbar.close()
    for p in processes:
        p.terminate()
    for result in results:
        if result != 'SolFound':
            fair +=1
    how_fair = 100 * fair / len(results)
    print(f"Fair: {how_fair:.4f}")
    return how_fair

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PeregriNN model checker")
    parser.add_argument('--network',help="path to neural network nnet file")
    parser.add_argument('--dataset', type = str, default='adult')
    parser.add_argument('--property', type = str)
    parser.add_argument('--timeout',default=300,help="timeout value")
    parser.add_argument('--max_depth',default=300,help="Maximum exploration depth")
    parser.add_argument('--cores', default=8, type = int)
    # args = None
    args = parser.parse_args()
    main(args)
