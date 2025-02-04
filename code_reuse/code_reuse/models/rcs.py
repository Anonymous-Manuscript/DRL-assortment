from models.__init__ import Model
import numpy as np
import gurobipy as gp
from gurobipy import GRB 

class RcsModel(Model):
    @classmethod
    def code(cls):
        return 'rcs'
    @classmethod
    def feature(cls):
        return ['alpha', 'preference_order']
    
    @classmethod
    def from_data(cls, data):
        return cls(data['products'], data['alpha'], data['preference_order'])
    
    @classmethod
    def simple_detetministic(cls, products):
        return cls(products, [1]+[0.5] * (len(products)-1), [i for i in range(len(products))])
    
    
    def __init__(self, products, alpha, preference_order):
        super(RcsModel, self).__init__(products)
        if alpha[0] != 1:
            raise Exception('alpha_0 should be 1')
        if len(alpha) != len(products)  or len(alpha) != len(preference_order):
            info = (len(alpha), len(products))
            raise Exception('Incorrect amount of alpha (%s) for amount of products (%s)' % info)
        self.alpha = alpha
        self.preference_order = preference_order
        self.products = products
        self.runtime = 0

    def probability_of(self, transaction):
        if transaction.product not in transaction.offered_products:
            return 0
        position = self.preference_order.index(transaction.product)
        temp = 1
        for i in self.preference_order[position+1:]:
            if i in transaction.offered_products:
                temp = temp * ( 1 - self.alpha[i])
        return self.alpha[transaction.product] * temp 
    def parameters_vector(self):
        return self.alpha
    
    def estimate_from_transaction(self, products, transaction):
        env = gp.Env(empty=True)
        env.setParam("OutputFlag",0)
        env.start()

        n = len(products)

        problem = gp.Model('RCS', env=env)

        alpha = problem.addVars(n,lb=0,ub=1,name = 'alpha')
        problem.addConstr((alpha[0] == 1))

        minus_alpha = problem.addVars(n,lb=0,ub=1)
        problem.addConstrs((1 - alpha[i] == minus_alpha[i] for i in range(n)))

        log_alpha = problem.addVars(n, lb = -GRB.INFINITY, ub = 0)
        log_minus_alpha = problem.addVars(n, lb = -GRB.INFINITY, ub = 0)

        for i in range(n):    
            problem.addGenConstrLog(alpha[i],log_alpha[i])
            problem.addGenConstrLog(minus_alpha[i],log_minus_alpha[i])

        x = problem.addVars(n, n, vtype = GRB.BINARY, name = 'x')
        problem.addConstrs((x[i,i] == 0 for i in range(n)))
        problem.addConstrs((x[0,i] == 0 for i in range(n)))
        for i in range(n):
            for j in range(n):
                if i != j:
                    problem.addConstr((x[i,j] + x[j,i] == 1))
                    for k in range(n):
                        if k != i and k!=j:
                            problem.addConstr((x[i,j] + x[j,k] + x[k,i] <= 2))

        obj1 = gp.quicksum(log_alpha[tra.product] for tra in transaction)
        obj2 = gp.quicksum((x[i,tra.product] * log_minus_alpha[i] for tra in transaction for i in tra.offered_products))

        problem.setObjective(obj1+obj2, GRB.MAXIMIZE)

        problem.optimize()

        parameters = np.zeros(n)
        x_array = np.zeros([n,n])
        for i in range(n):
            parameters[i] = alpha[i].X
            for j in range(n):
                x_array[i,j] = x[i,j].X

        self.products = products
        self.alpha = list(parameters)
        self.preference_order = list(list(np.argsort(np.sum(x_array,axis=1))))
        self.runtime = problem.Runtime


    def data(self):
        return {
            'alpha': self.alpha,
            'preference_order': self.preference_order
        }


