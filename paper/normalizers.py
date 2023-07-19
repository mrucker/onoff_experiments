from math import log
import numpy as np
import scipy.optimize as so

class BettingNormCS:
    def __init__(self, *, g, tau, gamma, alpha, lb):

        #g     -- a function which accepts a beta and an action
        #tau   -- the same tau from our paper
        #gamma -- the same gamma from our paper
        #alpha -- the level of significance we desire

        #note: This is based on https://arxiv.org/abs/2010.09686
        #   In the original paper the goal is to estimate the mean from random variables
        #   in our work we want to design a random variables that has a desired mean. That
        #   means we fix our mean and then update our random variable until the confidence
        #   interval for our random variable equalling our desired mean is small

        assert tau > 1, tau
        assert gamma > 0, gamma
        assert 0 < alpha < 1, alpha

        self.g = g
        self.tau = tau
        self.betamin = gamma*(1 - tau) / gamma
        self.betamax = gamma*1
        self.t = 0
        self.alist = np.array([])
        self.lamlist = np.array([0])
        self.nulist = np.array([0])
        self.lamgrad = 0
        self.nugrad = 0
        self.betaminus = self.betamin
        self.betaplus = self.betamax
        self.alpha = alpha
        self.thres = -log(alpha/2)
        self.gamma = gamma

        self.lb = lb

    def betlowercs(self):

        g = self.g(self.alist[-1], self.betaminus)
        nabla = (1 - g) / (1 + self.nulist[-1] * (1 - g))
        self.nugrad += nabla**2
        G = self.tau - 1
        D = 1/(2 * (self.tau - 1))
        gamma = 1 / (4 * G * D)
        epsilon = 1 / (gamma**2 * D**2)

        ytp1 = self.nulist[-1] + nabla / (gamma * (epsilon + self.nugrad))
        xtp1 = max(0, min(1/(2 * (self.tau - 1)), ytp1))
        self.nulist = np.append(self.nulist,xtp1)

    def betuppercs(self):
        g = self.g(self.alist[-1], self.betaplus)
        nabla = (g - self.lb) / (1 + self.lamlist[-1] * (g - self.lb))
        self.lamgrad += nabla**2

        G = self.tau - self.lb
        D = 1
        gamma = 1 / (4 * G * D)
        epsilon = 1 / (gamma**2 * D**2)

        ytp1 = self.lamlist[-1] + nabla / (gamma * (epsilon + self.lamgrad))
        xtp1 = max(0, min(1, ytp1))
        self.lamlist = np.append(self.lamlist,xtp1)

    def addobs(self, a):
        self.alist = np.append(self.alist,a)
        self.t += 1

        self.betlowercs()
        self.betuppercs()

    def uppercswealth(self, beta):
        return np.log(1+(self.g(self.alist,beta)-self.lb)*self.lamlist[:-1]).sum()

    def updateuppercs(self):
        if self.betaplus <= self.betamin:
            return

        maxbeta = self.betaplus
        maxbetawealth = self.uppercswealth(maxbeta)
        if maxbetawealth < self.thres:
            return

        minbeta = self.betamin
        minbetawealth = self.uppercswealth(minbeta)
        if minbetawealth > self.thres:
            self.betaplus = self.betamin
            return

        res = so.root_scalar(f = lambda beta: self.uppercswealth(beta) - self.thres, method = 'brentq', bracket = [minbeta, maxbeta])
        assert res.converged, res
        self.betaplus = res.root

    def lowercswealth(self, beta):
        return np.log(1+(1-self.g(self.alist,beta))*self.nulist[:-1]).sum()

    def updatelowercs(self):
        if self.betaminus >= self.betamax:
            return

        minbeta = self.betaminus
        minbetawealth = self.lowercswealth(minbeta)
        if minbetawealth < self.thres:
            return

        maxbeta = self.betamax
        maxbetawealth = self.lowercswealth(maxbeta)
        if maxbetawealth > self.thres:
            self.betaminus = self.betamax
            return

        res = so.root_scalar(f = lambda beta: self.lowercswealth(beta)-self.thres, method = 'brentq', bracket = [ minbeta, maxbeta ])
        assert res.converged, res
        self.betaminus = res.root

    def getci(self):
        return self.betaminus/self.gamma, self.betaplus/self.gamma
