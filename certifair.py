# %matplotlib inline
# Load all necessary packages
import sys
import os
os.environ['MKL_NUM_THREADS']="1"
os.environ['NUMEXPR_NUM_THREADS']="1"
os.environ['OMP_NUM_THREADS']="1"
os.environ['OPENBLAS_NUM_THREADS']="1"
import json
from datasets.utils import *

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
import numpy as np

SEED = 2312142
torch.manual_seed(SEED)
np.random.seed(SEED)

from intervals.symbolic_interval import SymbolicInterval
from intervals.interval_network import IntervalNetwork
from operators.flatten import Flatten
from operators.linear import Linear
from operators.activations import ReLU

from sklearn.preprocessing import  MaxAbsScaler
import matplotlib.pyplot as plt

from property import AdultProperty, BoxProperty,GermanProperty, LawProperty, CompasProperty
from models.mlp import MLP
import argparse
from types import SimpleNamespace
import verifier

def test_acc(net, loader, device = 'cpu'):
    correct = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
        
        return 100 * correct / len(loader.dataset)
def test(net, testloader, device = 'cpu'):
    correct = 0
    pos = 0
    cls_pos = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            pos += torch.sum(labels == 1)
            cls_pos += torch.sum(predicted == 1)
        
        acc = correct * 100 / len(testloader.dataset)
        print(f"Test accuracy: {acc:.4f}")
        print(f"Test set positivity rate: {pos * 100 / len(testloader.dataset):.4f}")
        print(f"Classifier positivity rate: {cls_pos * 100 / len(testloader.dataset):.4f}")
        
    return acc
def verify(net, testloader, property, device = 'cpu'):

    correct = 0
    total = 0
    verified = 0
    with torch.no_grad():
        unfair_err = 0
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # calculate outputs by running images through the network
            outputs = net(inputs)
            # the class with the highest energy is what we choose as prediction
            predicted = torch.round(torch.sigmoid(outputs))
            correct += (predicted == labels).sum().item()
            # close = property.is_inside(inputs,predicted)
            #Fairness error using sampling
            op_dict ={"Flatten":Flatten, "ReLU": ReLU, "Linear": Linear }
            inet = IntervalNetwork(net, op_dict).to(device)
            proj_inputs = property.project(inputs)
            proj_labels = torch.round(torch.sigmoid(net(proj_inputs)))
            input_bounds = property.adversarial_bounds(proj_inputs, proj_labels)
            in_bounds = input_bounds[(predicted == 1).squeeze() * inputs[:,property.sensitive_attr_idx] > 0 ]
            if(in_bounds.shape[0] ==0):
                continue
            total += in_bounds.shape[0]
            I = torch.zeros((inputs.shape[1], inputs.shape[1]+ 1), dtype = torch.float32)
            I = I.fill_diagonal_(1)
            I = I.unsqueeze(0).to(device)
            I = I.repeat(in_bounds.shape[0],1,1)
            input_sym = SymbolicInterval(in_bounds.unsqueeze(1).to(device),I,I, device = device)
            input_sym.concretize()
            sym_out = inet(input_sym)
            lb = sym_out.conc_lb
            verified += torch.sum(lb >0)
            n_samples = 1000
            r = in_bounds[...,1] - in_bounds[...,0]
            samples = r.unsqueeze(0) * torch.rand(n_samples, *in_bounds.shape[:-1],device =device) + in_bounds[...,0]
            male_samples_out = net(samples)
            unfair_cls = torch.any(male_samples_out < 0, dim = 0)
            err = torch.sum(unfair_cls)
            unfair_err += err
    acc = 100 * correct /  len(testloader.dataset)
    unfair_sampling = 100 * unfair_err / (total+ 1E-10)
    unfair_ver = 100 * (total-verified) / (total+1E-10)
    return (acc, unfair_sampling, unfair_ver,total)
    

def global_prop_loss(property : BoxProperty,inputs, interval_net):
    input_bounds = property.input_bounds.clone().unsqueeze(0)
    input_bounds = input_bounds.repeat(inputs.shape[0],1,1)
    disc_indices = property.discrete_features_idx
    input_bounds[:,disc_indices,0] = inputs[:,disc_indices]
    input_bounds[:,disc_indices,1] = inputs[:,disc_indices]
    # input_bounds[:,disc_indices,:] = torch.tensor([0.0,1.0], device =input_bounds.device)
    # input_bounds = input_bounds.repeat(1,2,1)
    input_bounds[:,property.sensitive_attr_idx, :] = torch.tensor([1.0,1.0]) # 1st input is male
    I = torch.zeros((input_bounds.shape[1], input_bounds.shape[1]+ 1), dtype = torch.float32)
    I = I.fill_diagonal_(1)
    I = I.unsqueeze(0).to(device)
    I = I.repeat(input_bounds.shape[0],1,1)
    input_bounds = input_bounds.unsqueeze(1)
    input_sym = SymbolicInterval(input_bounds.to(device),I,I, device = device)
    out_int = interval_net(input_sym)
    ub_1 = out_int.conc_ub
    lb_1 = out_int.conc_lb
    input_bounds_2 = input_bounds.clone()
    input_bounds_2[:,:,property.sensitive_attr_idx, :] = torch.tensor([0,0]) # 2nd input is female
    input_sym_2 = SymbolicInterval(input_bounds_2.to(device),I,I, device = device)
    out_int_2 = interval_net(input_sym_2)
    ub_2 = out_int_2.conc_ub
    lb_2 = out_int_2.conc_lb
    neg_majority = torch.where(ub_1 < 0)
    pos_minority = torch.where(lb_2 > 0)
    l = ub_1 - lb_2
    l[neg_majority] = 0
    l[pos_minority] = 0
    # return torch.mean(l)
    return torch.mean(torch.mean(torch.abs(ub_1 - lb_2)))

def train(net, trainloader, loss_criterion, optimizer, property, f_reg, epochs = 50, device ='cpu', fairness_loss = True, property_loss = True):

    op_dict ={"Flatten":Flatten, "ReLU": ReLU, "Linear": Linear }
    inet = IntervalNetwork(net, op_dict).to(device)     
    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        running_bound_loss = 0.0
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)


            # zero the parameter gradients
            optimizer.zero_grad()
            # print("Zeored the gradient")
            # forward + backward + optimize
            outputs = net(inputs)
            predicted = torch.round(torch.sigmoid(outputs))
            bound_loss = torch.tensor(0.0)
            #Get Symbolic bounds
            if(fairness_loss):
                if(property_loss):
                    bound_loss = global_prop_loss(property,inputs,inet)
                # property.product_min_similarity_box()
                else:
                    proj_inputs = property.project(inputs)
                    proj_labels = torch.round(torch.sigmoid(net(proj_inputs)))
                    input_bounds = property.adversarial_bounds(proj_inputs, proj_labels)
                    input_bounds = input_bounds.unsqueeze(1)
                    I = torch.zeros((inputs.shape[1], inputs.shape[1]+ 1), dtype = torch.float32)
                    I = I.fill_diagonal_(1)
                    I = I.unsqueeze(0).to(device)
                    I = I.repeat(inputs.shape[0],1,1)
                    input_sym = SymbolicInterval(input_bounds.to(device),I,I, device = device)
        
                    out_int = inet(input_sym)
                    ub = out_int.conc_ub
                    lb = out_int.conc_lb
                    #Bounds loss
                    # lb_proj = lb[close.type(torch.bool)]
                    lb_proj = lb[(proj_labels == 1).squeeze() * inputs[:,property.sensitive_attr_idx] > 0]
                    if(lb_proj.shape[0] > 0):
                        bound_loss = -1 * torch.mean(torch.log(torch.sigmoid(lb_proj))) #Bound loss if male and calssified 1
                        # bound_loss = -1 * torch.mean(lb_proj)
                    # bound_loss = -1 * torch.mean(lb[(labels == 1).squeeze() * inputs[:,3] == 1 * (lb < 0).squeeze()]) #Bound loss if male and calssified 1
            #Training loss
            tr_loss = loss_criterion(outputs,labels)
            loss = (1-f_reg) * tr_loss + f_reg * bound_loss
            # print("Calling backward, Zeored the gradient",loss)
            loss.backward()
            # for name, param in net.named_parameters():
            #     print(name, torch.isfinite(param.grad).all())
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            running_bound_loss += bound_loss.item()
        

        print(f"====Epoch {epoch}====")
        print("Epoch loss",running_loss)

    print('Finished Training')
    acc = test(net, trainloader, device)
    print(f"Training Accuracy: {acc:.4f}")
    # verify(net,trainloader,property, device = device)
    return acc

if __name__ == '__main__':
    # Get the dataset and split into train and test
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('dataset',help="[Adult, German, Crime, Compas]", default="adult", type=str.lower)
    parser.add_argument('property',help="Name of the property function")
    parser.add_argument('name',help="Name of the Experiment")
    parser.add_argument('--lr',help="Learning rate", type= float, default=0.001)
    parser.add_argument('--fr',help="Fairness regualrizer", type= float, default=0.01)
    parser.add_argument('--desc',help="Experiment description", default = "Description")
    parser.add_argument('--mode', help="[train, test]", default='train')
    parser.add_argument('--model_name', help="Model name to load during test mode only")
    parser.add_argument('--verify', help = "Runs the verifier on the trained network", default= True, action = argparse.BooleanOptionalAction)
    parser.add_argument('--bound_loss', help="Activate the fairness regularizer", default= True, action = argparse.BooleanOptionalAction)
    parser.add_argument('--batch_size', default = 256,type = int)
    parser.add_argument('--epochs', default = 50, type= int)
    parser.add_argument('--balance', help='Balanced training for unbalanced datasets', default= False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--property_loss', help = 'Use global fairness regularizer',default= False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--layers', help = "Hidden units per layer, comma seperated starting from the layer after the input", type = str, default = "20,20,1")
    parser.add_argument('--verification_cores', help = "Number of cpu cores for verification", type = int, default = 8)
    parser.add_argument('--device', help = "cuda:N, cpu", type = str, default = 'cuda')

    args = parser.parse_args()
    
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    EXPR_DIR = os.path.join(DIR_PATH,'experiments')
    if args.dataset in Datasets.loaders:
        loader = Datasets.loaders[args.dataset]
    else:
        raise NotImplementedError("Dataset not implemented")

    dataset_orig = loader()
    privileged_groups = [{'sex': 1}]
    unprivileged_groups = [{'sex': 0}]

    dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)

    #Normalize
    min_max_scaler = MaxAbsScaler()
    dataset_orig_train.features = min_max_scaler.fit_transform(dataset_orig_train.features)
    dataset_orig_test.features = min_max_scaler.transform(dataset_orig_test.features)

    layers_sizes = [dataset_orig_train.features.shape[1],20,20,1]
    if(args.layers != ""):
        layers_sizes = [int(l_size) for l_size  in args.layers.split(',')]
        layers_sizes = [dataset_orig_train.features.shape[1]] + layers_sizes
    debiased_model = MLP(layers_sizes= layers_sizes)

    batch_size = args.batch_size
    epochs = args.epochs
    train_data = TensorDataset(torch.tensor(dataset_orig_train.features, dtype = torch.float), torch.tensor(dataset_orig_train.labels, dtype = torch.float))
    test_data = TensorDataset(torch.tensor(dataset_orig_test.features, dtype = torch.float), torch.tensor(dataset_orig_test.labels, dtype = torch.float))

    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
                                            shuffle=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    device = torch.device(args.device)
    if(args.mode == "test"):
        debiased_model.load_state_dict(torch.load(os.path.join(EXPR_DIR,'models', args.model_name)))
        debiased_model.eval()
        debiased_model = debiased_model.to(device)
        test(debiased_model, testloader, device = device)
    else:      
        pos_weight = None 
        if(args.balance):
            neg_samples = torch.tensor(np.sum(dataset_orig_train.labels == 0, dtype = np.float32))
            pos_weight  = neg_samples / (len(dataset_orig_train.labels) - neg_samples)                         
        loss = torch.nn.BCEWithLogitsLoss(pos_weight= pos_weight)
        if args.dataset in Datasets.property_cls:
            prop_cls = Datasets.property_cls[args.dataset]
        else:
            raise NotImplementedError("PropertyClass not implemented")

        property = getattr(prop_cls, args.property)(device = device)
        # property = BoxProperty.property5(device = device)
        debiased_model  = debiased_model.to(device)
        optimizer = torch.optim.Adam(debiased_model.parameters(), lr=args.lr)
        #Hyperparameters
        f_reg = args.fr

        f =  open(os.path.join(EXPR_DIR,'logs', f"exp_{args.name}.log"), 'w')
        f.write(f"Experiment: {args.name}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Property: {args.property}\n")
        f.write("Description: " + args.desc +"\n")
        f.write(f"learning rate: {args.lr:.4f}\n")
        f.write(f"Fairness regularization:{f_reg:.4f}\n")
        f.write("------Property details------\n")
        f.write(f"Type: {property.type}\n")
        f.write(f"Input bounds\n {property.input_bounds}\n")
        f.write(f"delta\n {property.delta:.4f}\n")
        f.write(json.dumps(vars(args)) + '\n')
        f.flush()
        train_acc = train(debiased_model, trainloader, loss, optimizer, property, f_reg, device = device, epochs = epochs, fairness_loss= args.bound_loss, property_loss=args.property_loss)
        f.write(f'Accuracy of the network on the {len(trainloader.dataset)} training images: {train_acc:.4f} %\n')
        acc, unfair_sampling, unfair_ver, total = verify(debiased_model, testloader, property, device = device)
        print(f'Accuracy of the network on the {len(testloader.dataset)} test images: {acc:.4f} %')
        print(f'UnFairness (sampling) of the network on the {total} test images: {unfair_sampling:.4f} %')
        print(f'UnFairness (Lower bound) of the network on the {total} test images: {unfair_ver:.4f} %')
        f.write(f'Accuracy of the network on the {len(testloader.dataset)} test images: {acc:.4f} %\n')
        f.write(f'UnFairness (sampling) of the network on the {total} test images: {unfair_sampling:.4f} %\n')
        f.write(f'UnFairness (Lower bound) of the network on the {total} test images: {unfair_ver:.4f} %\n')
        f.flush()
        cpu_model = debiased_model.cpu()
        file_dir = os.path.join(EXPR_DIR,'models',f'model_ex_{args.name}.pth')
        torch.save(cpu_model.state_dict(), file_dir)
        if(args.verify == True):
            #Call the Fairness Verifier
            #Prep args
            v_args = SimpleNamespace()
            v_args.network = file_dir
            v_args.dataset = args.dataset
            v_args.property = args.property
            v_args.timeout = 300
            v_args.max_depth = 300
            v_args.cores= args.verification_cores
            print("====Calling Verifier====")
            how_fair = verifier.main(v_args)
            f.write(f'Verified Fairness: {how_fair:.4f} %\n')
        
        f.close()