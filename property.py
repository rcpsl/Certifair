import torch
from scipy.optimize import linprog as lp
class BoxProperty:
    def __init__(self, input_bounds, output_property, num_features_idx, sensitive_idx, delta = 0, type = "noise"):
        self.input_bounds = input_bounds
        self.output_property = output_property
        self.numerical_features_idx = num_features_idx
        self.sensitive_attr_idx = sensitive_idx
        self.discrete_features_idx = [i for i in range(input_bounds.shape[0]) if ((i not in self.numerical_features_idx) and (i != self.sensitive_attr_idx))]
        self.delta = torch.tensor(delta).to(self.input_bounds.device)
        self.type = type

    def is_inside(self, inputs, label):
        #Returns true if point intersects with property
        result = torch.zeros(inputs.shape[0],1)
        point = inputs
        inclusion = ((point >= self.input_bounds[:,0]) * (point <= self.input_bounds[:,1]))[:,self.numerical_features_idx]
        result[torch.all(inclusion, dim = -1)] = 1.0
        result[(label != self.output_property)] = 0.0
        return result.squeeze().type(torch.bool)

    def proximity(self, inputs, label):
        #Returns how CLOSE is the point from the INPUT box
        result = torch.zeros(inputs.shape[0],1)
        point = inputs
        inclusion = ((point >= self.input_bounds[:,0]) * (point <= self.input_bounds[:,1]))[:,self.numerical_features_idx]
        result[torch.all(inclusion, dim = -1)] = 1.0
        result[(label != self.output_property)] = 0.0
        return result.squeeze()

    def project(self,x):
        y = x.clone()
        p_bounds = self.input_bounds[self.numerical_features_idx]
        y[:,self.numerical_features_idx] = torch.maximum(torch.minimum(x[:,self.numerical_features_idx], p_bounds[:,1]), p_bounds[:,0])
        return y
        
    def adversarial_bounds(self, inputs, labels):
        adversarial_bounds = torch.zeros(inputs.shape[0], inputs.shape[1], 2).to(inputs.device)
        p_bounds = self.input_bounds
        adversarial_bounds[:,self.numerical_features_idx,0] = torch.maximum(p_bounds[self.numerical_features_idx,0], inputs[:,self.numerical_features_idx] - self.delta)
        adversarial_bounds[:,self.numerical_features_idx,1] = torch.minimum(p_bounds[self.numerical_features_idx,1], inputs[:,self.numerical_features_idx] + self.delta)
        
        # adversarial_bounds[:,self.discrete_features_idx] = torch.tensor([0.0,1.0]).to(adversarial_bounds.device)
        adversarial_bounds[:,self.discrete_features_idx] = inputs[:,self.discrete_features_idx].unsqueeze(-1)
        adversarial_bounds[:,self.sensitive_attr_idx,:] = torch.tensor([0,0]).to(adversarial_bounds.device) #female
        
        return adversarial_bounds
    
    def product_min_similarity_box(self):
        dims = len(self.numerical_features_idx)
        aug_dims = 2 *  dims
        min_box = torch.zeros(aug_dims,2)
        A_ub = torch.hstack((torch.eye(dims), -1*torch.eye(dims)))
        A_ub = torch.vstack((A_ub,-A_ub))
        b_ub = torch.ones(A_ub.shape[0],1) * self.delta.cpu()
        bounds = self.input_bounds[self.numerical_features_idx].cpu()
        bounds = torch.vstack((bounds,bounds))
        for dim in range(aug_dims):
            c = torch.zeros(aug_dims)
            c[dim] = 1.0
            res = lp(c, A_ub, b_ub, bounds = bounds)

            #Solve 2 linear programs to get a min and max of x_d
        return  
    @property
    def num_features(self):
        return self.input_bounds.shape[0]

class AdultProperty:
    num_features_idx = [0,1,4,5,6]
    cat_features_len = [2,4, 8, 5, 6] #allowed values for each cat variable
    sensitive_idx  = 3

    @classmethod
    def property1(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,0.5])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.1])
        input_bounds[5] = torch.tensor([0.0,0.1])
        input_bounds[6] = torch.tensor([0.45,0.6]) 
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.05)

    @classmethod
    def property2(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([ 0.1889,1.0])
        input_bounds[1] = torch.tensor([0.0625,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        input_bounds[5] = torch.tensor([0.0,1.0])
        input_bounds[6] = torch.tensor([ 0.0101,1.0]) 
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.02)
        
    @classmethod
    def property3(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.7,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.1])
        input_bounds[5] = torch.tensor([0.0,0.1])
        input_bounds[6] = torch.tensor([0.45,0.6])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.1)

    @classmethod
    def property4(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.18888889,1.0])
        input_bounds[1] = torch.tensor([0.0625,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        input_bounds[5] = torch.tensor([0.0,1.0])
        input_bounds[6] = torch.tensor([0.01010,1.0])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.03)

    @classmethod
    def property5(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.3])
        input_bounds[5] = torch.tensor([0.0,0.2])
        input_bounds[6] = torch.tensor([0.5,0.75])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.05)
    @classmethod
    def property6(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.18888889,1.0])
        input_bounds[1] = torch.tensor([0.0625,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        input_bounds[5] = torch.tensor([0.0,1.0])
        input_bounds[6] = torch.tensor([0.01010,1.0])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.00)

    @classmethod
    def property7(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.3])
        input_bounds[5] = torch.tensor([0.0,0.2])
        input_bounds[6] = torch.tensor([0.5,0.75])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.02)

    @classmethod
    def property8(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.3])
        input_bounds[5] = torch.tensor([0.0,0.2])
        input_bounds[6] = torch.tensor([0.5,0.75])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.03)

    @classmethod
    def property9(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.3])
        input_bounds[5] = torch.tensor([0.0,0.2])
        input_bounds[6] = torch.tensor([0.5,0.75])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.07)
    
    @classmethod
    def property10(cls, device = 'cpu'):
        #TODO: Handle discrete variables
        input_bounds = torch.zeros((30,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.4,1.0])
        input_bounds[1] = torch.tensor([0.5,0.7])
        input_bounds[4] = torch.tensor([0.0,0.3])
        input_bounds[5] = torch.tensor([0.0,0.2])
        input_bounds[6] = torch.tensor([0.5,0.75])
        # input_bounds[3] = torch.tensor([1.0, 1.0]) # Male
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.1)
class GermanProperty:
    num_features_idx = [0,1,2,3]
    cat_features_len = [3,3,3,3]
    sensitive_idx  = 4

    @classmethod
    def property1(cls, device = 'cpu'):

        input_bounds = torch.zeros((17,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.05)
    @classmethod
    def property2(cls, device = 'cpu'):

        input_bounds = torch.zeros((17,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.0)

class LawProperty:
    num_features_idx = [0,1]
    cat_features_len = [2,2,24,5]
    sensitive_idx  = 4 

    @classmethod
    def property1(cls, device = 'cpu'):
        input_bounds = torch.zeros((34,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.6,1.0])
        input_bounds[1] = torch.tensor([0.3,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.02)

    @classmethod
    def property2(cls, device = 'cpu'):
        input_bounds = torch.zeros((34,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.03)

    @classmethod
    def property3(cls, device = 'cpu'):
        input_bounds = torch.zeros((34,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.0)

class CompasProperty:
    num_features_idx = [1,3,4,5,6]
    cat_features_len = [2,2,3,3]
    # num_features_idx = [0,2]
    # cat_features_len = [3,3]
    sensitive_idx  = 2

    @classmethod
    def property1(cls, device = 'cpu'):
        # input_bounds = torch.zeros((9,2), dtype = torch.float32, device = device)
        input_bounds = torch.zeros((14,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.02)

    @classmethod
    def property2(cls, device = 'cpu'):
        # input_bounds = torch.zeros((9,2), dtype = torch.float32, device = device)
        input_bounds = torch.zeros((14,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.0)

class BankProperty:
    num_features_idx = [1,2,3,4,5]
    cat_features_len = [2,2,2,3,7,3]
    sensitive_idx  = 0

    @classmethod
    def property1(cls, device = 'cpu'):
        input_bounds = torch.zeros((22,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)
        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.3)

class Compas2DProperty:
    # num_features_idx = [1,3,4,5,6]
    # cat_features_len = [2,2,3,3]
    num_features_idx = [0,2]
    cat_features_len = [3,3]
    sensitive_idx  = 1

    @classmethod
    def property1(cls, device = 'cpu'):
        input_bounds = torch.zeros((9,2), dtype = torch.float32, device = device)
        # input_bounds = torch.zeros((14,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.02)

    @classmethod
    def property2(cls, device = 'cpu'):
        input_bounds = torch.zeros((9,2), dtype = torch.float32, device = device)
        # input_bounds = torch.zeros((14,2), dtype = torch.float32, device = device)
        input_bounds[0] = torch.tensor([0.0,1.0])
        input_bounds[1] = torch.tensor([0.0,1.0])
        input_bounds[2] = torch.tensor([0.0,1.0])
        input_bounds[3] = torch.tensor([0.0,1.0])
        input_bounds[4] = torch.tensor([0.0,1.0])
        output_property = torch.tensor([1], dtype =torch.int, device = device)

        return BoxProperty(input_bounds, output_property, cls.num_features_idx, cls.sensitive_idx, delta = 0.0)
