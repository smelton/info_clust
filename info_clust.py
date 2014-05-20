import numpy as np
from numpy.random import multivariate_normal as mn
import matplotlib.pyplot as plt

## Helper Functions

def normalize(P):
	return (P.T/P.sum(1).T).T

def get_SCi(Cluster,i,P,S):
	## sum over all elements j of P(j in Cluster)*S(i,j)
	N,N_C = P.shape
	return sum([P[j,Cluster]*S[i,j] for j in xrange(N)])/sum([P[j,Cluster] for j in xrange(N)])

def get_SC(Cluster,P,S):
	##sum over all elements j,i P(i in cluster)*P(j in cluster)*Sim(i,j)
	N,N_C = P.shape
	S_clust = np.array([P[i,Cluster]*P[j,Cluster]*S[i,j] for i in xrange(N) for j in xrange(N) if i<j])
	P_clust = np.array([P[i,Cluster]*P[j,Cluster] for i in xrange(N) for j in xrange(N) if i<j])
	return S_clust.sum()/P_clust.sum()

def update(P,T,S):
	N,N_C = P.shape
	P_new = P.copy()
	PC = 1.0*P.sum(0)/N
	SCis = np.array([[get_SCi(clust,i,P,S) for clust in xrange(N_C)] for i in xrange(N)])
	SCs = np.array([get_SC(clust,P,S) for clust in xrange(N_C)])
	for point in xrange(N):
		P_new[point] = PC*np.exp((1.0/T)*(2*SCis[point,:] - SCs))
	P_new = normalize(P_new)
	return P_new

def similarity_score(P,S):
	__, N_C = P.shape
	SCs = np.array([get_SC(clust,P,S) for clust in xrange(N_C)])
	return SCs.sum()

def discretize_P(P):
	NP = P.copy()
	for i,k in enumerate(P):
		NP[i,:] = np.round(k)
	return NP

## main function

def info_cluster(S, N_C, T = 0.1, epsilon = 10.0**(-30.0)):
	N = S.shape[0]
	Dist = normalize(np.random.random([N,N_C]))
	conv = False
	for step in xrange(500):
		oldDist = Dist.copy()
		Dist = update(Dist,T,S)
		if abs((Dist-oldDist).sum())<= epsilon:
			print 'Convergence after ' + str(step) + ' iterations'
			conv = True
			break
	if not conv:
		print 'Did not converge'
	return Dist
