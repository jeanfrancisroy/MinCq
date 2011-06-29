#!/usr/bin/env python
#-*- coding:utf-8 -*-
'''
Created on Jun 27, 2011

@author: Jean-Francis Roy

This module encapsulates the QP solver provided by CVXOPT. 
See the manual of CVXOPT for more information about the 
input parameters of the QP solver.
'''

from cvxopt.solvers import qp, options
from cvxopt import matrix, sparse
import sys

class QP:

    defaultparams={"q" : None,
                   "G" : None,
                   "h" : None,
                   "A" : None,
                   "b" : None,
                   "initval" : None,
                   "debug" : True,
                   }

    def __init__(self, P, params={}, useSparseMatrix = False):
        QP.defaultparams.update(params)
        tmp = QP.defaultparams
        self.P = P
        self.q = tmp["q"]
        self.G = tmp["G"]
        self.h = tmp["h"]
        self.A = tmp["A"]
        self.b = tmp["b"]
        self.initvals = tmp["initvals"]
        self.debug = tmp["debug"]
        self.useSparseMatrix = useSparseMatrix
        
    def solve(self):
        if (self.debug):
            print "preparing QP problem"
        if (self.P != None):
            self.P = matrix(self.P)
            if self.useSparseMatrix:
                self.P = sparse(self.P)
            if (self.debug):
                print "P", self.P.size
        if (self.q != None):
            self.q = matrix(self.q)
            if self.useSparseMatrix:
                self.q = sparse(self.q)
            if (self.debug):
                print "q", self.q.size
        if (self.G != None):
            self.G = matrix(self.G).T
            if self.useSparseMatrix:
                self.G = sparse(self.G)
            if (self.debug):
                print "G", self.G.size
        if (self.h != None):
            self.h = matrix(self.h)
            if self.useSparseMatrix:
                self.h = sparse(self.h)
            if (self.debug):
                print "h", self.h.size
        if (self.A != None):
            self.A = matrix(self.A).T
            if self.useSparseMatrix:
                self.A = sparse(self.A)
            if (self.debug):
                print "A", self.A.size
        if (self.b != None):
            self.b = matrix(self.b)
            if self.useSparseMatrix:
                self.b = sparse(self.b)
            if (self.debug):
                print "b", self.b.size
        
        if (self.debug):
            print "solving..."
        
        if (not self.debug):
            sys.stdout = open("/dev/null", "w")
        
        ret = qp(self.P, self.q, self.G, self.h, self.A, self.b, initvals=self.initvals)
        
        if (not self.debug):
            sys.stdout = sys.__stdout__
        
        return ret
