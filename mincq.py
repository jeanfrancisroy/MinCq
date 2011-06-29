#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Created on Jun 27, 2011

@author: Jean-Francis Roy
'''

import numpy as np
from qp import QP
from dataset import Dataset
import argparse

class MinCqQP:
    """ 
    This class prepares the MinCq QP to be solved by CVXOPT
    """
    
    def __init__(self, X, Y, margin, majorityvote, U=None):
        self.X = X
        self.Y = Y
        self.margin = margin
        self.psiMatrix = majorityvote.psiMatrix(self.X)
        
        if U is not None:
            self.U = U
            self.UpsiMatrix = majorityvote.psiMatrix(self.U)
        else:
            self.U = self.X
            self.UpsiMatrix = self.psiMatrix

        self.n = np.shape(self.psiMatrix)[0]
        
    def __call__(self):
        P = 2 * self.createMMatrix()
        q = self.createqvector()
        G = self.createGmatrix()
        h = self.createhvector()
        A = self.createAmatrix()
        b = self.createb()
        
        initvals = {}
                
        return (P,q,G,h,A,b,initvals)
        
    def createMMatrix(self):
        M = []
        
        for i in range(self.n):
            Mi = [0]*self.n
            for j in range(self.n):
                Mi[j] = np.mean(np.multiply(self.UpsiMatrix[i],self.UpsiMatrix[j]))
            M.append(Mi)
        
        self.M = np.matrix(M)
        
        return self.M

    def createqvector(self):            
        return np.array(-1.0 * np.mean(self.M, axis=1))
        
    def createAmatrix(self):
        # Fixed margin.
        A = np.mean(np.multiply(self.Y, self.psiMatrix), axis=1)
        return np.matrix(A)
    
    def createGmatrix(self):
        G = []
        
        # Force le poids des classificateurs à appartenir à l'intervalle [0,1/n].
        for i in range(self.n):
            g = [0]*self.n
            g[i] = 1.0
            G.append(g)

        for i in range(self.n):
            g = [0]*self.n
            g[i] = -1.0
            G.append(g)
 
        return np.matrix(G).T

    def createhvector(self):
        h = [0] * (2*self.n)
        
        # Force le poids des classificateurs à appartenir à l'intervalle [0,1/n].
        for i in range(self.n):
            h[i] = float(1)/self.n
        for i in range(self.n, 2*self.n):
            h[i] = 0

        return np.matrix(h).T

    def createb(self):
        # Fixed margin.
        mis = np.mean(np.multiply(self.Y, self.psiMatrix), axis=1)

        b = 0.5 * (self.margin + np.mean(mis))

        return np.matrix(b)
    
def sign(x):
    if x>0:
        return 1
    else:
        return -1
class MajorityVoteEstimator ():
    
    def __init__(self, majorityvote) :
        self.majorityvote = majorityvote

    def predict(self, x, id=None):
        return self.majorityvote.evaluate(x)
    
class Kernel:
    def __init__(self, kernelFunc, *args):
        self.kernelFunc = kernelFunc
        self.args = args
        
    def evaluate(self, x1, x2):
        return self.kernelFunc(x1, x2, *self.args)
    
def rbfKernel(a, b, gamma):
    """
    RBF kernel
    
    a, b -- Two vectors with same length
    gamma -- RBF kernel parameter
    """
    c = a-b
    return np.exp(-gamma * np.dot(c,c))
    
class MajorityVote:

    def __init__(self):
        self.voters = np.array([])
        self.Q = np.array([])

    def getNbVoters(self):
        return len(self.voters)
                        
    def addVoter(self,voter):
        self.voters = np.append(self.voters, voter)
        
    def classify(self, X):
        evaluations = np.transpose(map(lambda s: s.evaluate(X), self.voters))
        return map(sign, np.dot(evaluations, self.Q))
    
    def evaluate(self, X):
        return self.classify(X)
    
    def getMargin(self, (X,Y)):
        return np.multiply(Y, np.dot(self.Q.T, self.psiMatrix(X)))

    def psiMatrix(self, X):
        # Returns a matrix of voters evaluations on the examples. A voter is on
        # a line and an example is on a column.
        # shape : [n,m]
        return np.matrix(map(lambda s: s.evaluate(X), self.voters))

class DecisionStump:
    """Generic Attribute Threshold Classifier
    
    Override with specific loss to specialize
    
    nAttribute -- idx of attribute to check
    nThreshold -- threshold value
    nDirection -- {+1, -1} use to create inverse stump
    loss -- loss function to use, should accept 2 args
    
    classify -- distance from threshold, sign indicate positive or negative classification
    getRisk  -- evaluate classifier's risk on dataset using loss
    getLossOnExample -- evaluate classifier on example using loss
     
    """
    def __init__(self,nAttribute, nThreshold, nDirection):
        self.nAttribute = nAttribute
        self.nDirection = nDirection
        self.nThreshold = nThreshold
        
    def classify(self, X):
        # Lol.
        return map(lambda x:((x[self.nAttribute] > self.nThreshold)*2 - 1) * self.nDirection, X)
    
    def evaluate(self, X):
        return self.classify(X)


class DecisionStumps(MajorityVote):
    """Decision stumps majority vote
    
    createStumps -- create regular and inverse stumps for dataset using (stumps) class. 
    Default = BasicDesicionStump 

    """
    def __init__(self,X, stumps = DecisionStump):
        MajorityVote.__init__(self)
        self.X = X
        self.cStumps = stumps
    
    @classmethod
    def createStumps(cls, dataset, addInverseStumps = True, stumps = DecisionStump):
        stumps = cls(dataset, stumps)
      
        stumps.addRegularStumps()
        
        if (addInverseStumps):
            stumps.addRegularInverseStumps()
    
        return stumps    
 
    def addRegularStumps(self):
        if len(self.X) != 0:
            for i in range(len(self.X[0])):
                t = self.findExtremums(self.X,i)
                inter = (t[1]-t[0])/11.0
                # We don't add stumps if the attribute has only one possible value.
                if inter > 0:
                    for x in range(10):
                        self.addStump(i,t[0]+inter*(x+1),1)
    
    def addRegularInverseStumps(self):
        if len(self.X) != 0:
            for i in range(len(self.X[0])):
                t = self.findExtremums(self.X,i)
                inter = (t[1]-t[0])/11.0
                # We don't add stumps if the attribute has only one possible value.
                if inter > 0:
                    for x in range(10):
                        self.addStump(i,t[0]+inter*(x+1),-1)
                        
    def addStump(self,nAttribute, nThreshold, nDirection):
        self.voters = np.append(self.voters, self.cStumps(nAttribute,nThreshold, nDirection))
        
    def findExtremums(self,vExample,i):
        mini = np.Infinity
        maxi = -np.Infinity
        for t in vExample:
            if t[i] < mini:
                mini = t[i]
            if t[i] > maxi:
                maxi = t[i]
        return (mini,maxi)

class DummyVoter:
    def __init__(self, M):
        self.M = M
        
    def evaluate(self, X):
        return [self.M]*len(X)

class KernelVoter:
    def __init__(self, kernel, direction, xi):
        self.xi = xi
        self.kernel = kernel
        self.direction = direction

    def evaluate(self, X):
        return [self.direction * self.kernel.evaluate(self.xi, x) for x in X]
    
class KernelVote(MajorityVote):
    def __init__(self, X, kernel, cVoter = KernelVoter):
        MajorityVote.__init__(self)
        self.X = X
        self.cVoter = cVoter
        self.kernel = kernel
    
    @classmethod
    def createVoters(cls, X, useBias, autoComplemented, kernel):
        vote = cls(X, kernel, KernelVoter)
      
        vote.addRegularVoters()
        if (useBias):
            M = np.amax(map(lambda f: f.evaluate(X), vote.voters))
            vote.addRegularBiasVoter(M)
        
        if (autoComplemented):
            vote.addRegularInverseVoters()
            if (useBias):
                vote.addInverseBiasVoter(M)
    
        return vote    
 
    def addRegularVoters(self):
        if len(self.X) != 0:
            for i in range(len(self.X)):
                self.addVoter(self.X[i], 1)
    
    def addRegularInverseVoters(self):
        if len(self.X) != 0:
            for i in range(len(self.X)):
                self.addVoter(self.X[i], -1)
    
    def addRegularBiasVoter(self, M):
        self.voters = np.append(self.voters, DummyVoter(M))
    
    def addInverseBiasVoter(self, M):
        self.voters = np.append(self.voters, DummyVoter(-M))
                
    def addVoter(self, x, nDirection):
        self.voters = np.append(self.voters, self.cVoter(self.kernel, nDirection, x))

class MinCqLearner():
    """
    Given a set of unlabeled examples U, a set Xidx of indices representing whose
    example are labeled, and given a set Y of labels, MinCq's mu parameter, a voter type, and kernel 
    parameters kArgs, MinCqLearner prepares the MinCq QP and solves it using CVXOPT, then 
    returns a MajorityVoteTransductiveEstimator.
    
    To use MinCq on its supervised version, simply provide a set Xidx containing all indices from U.
    
    All sets should be Numpy arrays.
    """
    def __init__(self):
        self.debug = 0
        
    def learn(self, trainX, trainY, allX, mu, voters, *kArgs):
        
        # Preparing the majority vote
        if (voters == "rbf"):
            kernelFunc = rbfKernel
            majorityvote = KernelVote.createVoters(trainX, True, False, Kernel(kernelFunc, *kArgs))
        elif (voters == "decisionstumps"):
            majorityvote = DecisionStumps.createStumps(trainX, False)
        
        majorityvote.Q = np.array([1.0/len(majorityvote.voters)]*len(majorityvote.voters))
        
        # Training (creating the QP)
        (P,q,G,h,A,b,initvals) = MinCqQP(trainX, trainY, mu, majorityvote, allX)()
            
        # Solving and getting the resulting weights
        params = {"q":q, "G":G, "h":h, "A": A, "b":b, "initvals":initvals, "debug":self.debug}
        solver = QP(P, params)
        status = 'None'
        try:
            ret = solver.solve()
            status = ret['status']
            majorityvote.Q = np.array(ret['x'])
            majorityvote.Q = np.array(map(lambda(x):2*x[0] - 1.0/len(majorityvote.voters), majorityvote.Q))
            
            # Creating the estimator
            estimator = MajorityVoteEstimator(majorityvote)

        except ValueError:
            estimator = None
        
        return estimator


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='MinCq is a Machine Learning algorithm presented at International Conference on Machine Learning (ICML) 2011. Please see the extended version of the paper for more information. http://graal.ift.ulaval.ca/publications.php. Please note that the input parameters are very important : use the default ones at your own risk of bad results ;-).')
    
    parser.add_argument("--mu", dest="mu", type=float, default=1e-04, help="Indicates to which value the first moment of the margin of the Q-weighted distribution on the voters must be fixed. Default is 1e-04.")
    parser.add_argument("--voters", dest="voters", default="decisionstumps", help="Defines the nature of the voters to be considered. Either decisionstumps (for 10 Decision Stumps per attribute) or rbf (for RBF kernel functions centered on the training examples. Default : decisionstumps.")
    parser.add_argument("--gamma", dest="gamma", type=float, default=0.05, help="Defines the gamma parameter of the RBF kernel. Only used if --voters is set to rbf. Default: 0.05")
    parser.add_argument("--transductive", dest="transductive", action='store_const', const=True, default=False, help="Defines if the transductive framework (where the examples from the testing set will be used without their labels) is to be considered.")
    parser.add_argument("training_set", help="Defines the file containing the training set, where each line defines an example, the first column defines the label in {-1, 1}, and the next columns represent the features (real-valued).")
    parser.add_argument("testing_set", help="Defines the file containing the testing set, with the same file structure than the training set.")


    args = parser.parse_args()
    
    print("Training set: %s" %(args.training_set))
    print("Testing set: %s" %(args.testing_set))
    print("mu: %f" %(args.mu))
    print("voters: %s" %(args.voters))
    if(args.voters == "rbf"):
        print("gamma: %f" %(args.gamma))
    print("transductive: %s" %(args.transductive))
    
    print("")
    print("Loading files...")
    
    dTrain = Dataset()
    dTrain.loadFromFile(args.training_set)
    if (dTrain.X is None):
        raise Exception("Cannot load the training data")

    dTest = Dataset()
    dTest.loadFromFile(args.testing_set)
    if (dTest.X is None):
        raise Exception("Cannot load the testing data")
    
    allX = np.concatenate((dTrain.X, dTest.X), 0)

    print("Solving...")
    
    l = MinCqLearner()
    if (args.transductive):
        e = l.learn(dTrain.X, dTrain.Y, allX, args.mu, args.voters, args.gamma)
    else:
        e = l.learn(dTrain.X, dTrain.Y, dTrain.X, args.mu, args.voters, args.gamma)
        
    nbTrainErr = len(np.where(dTrain.Y != e.predict(dTrain.X))[0])
    nbTestErr = len(np.where(dTest.Y != e.predict(dTest.X))[0])
    
    print("Done!")
    print("")
    print("Training risk : %f" %(float(nbTrainErr) / len(dTrain.Y)))
    print("Testing risk : %f" %(float(nbTestErr) / len(dTest.Y)))