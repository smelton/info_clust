import numpy as np
from numpy.random import multivariate_normal as mn
import matplotlib.pyplot as plt

from info_clust import *

def produce_sample_data(N1,N2,N3):
	Means = np.array([[-10,-3],[0,0],[10,10]])
	Vars = np.array([[[1,0.5],[0.5,0.7]],[[2,-2.4],[-2.4,1]]])
	SAMPLES = np.vstack([mn(Means[0],Vars[0],N1),mn(Means[1],Vars[1],N2),mn(Means[2],Vars[1],N3)])
	return SAMPLES

def plot_samples_2d(samples):
	import matplotlib.pyplot as plt
	plt.scatter(samples.T[0],samples.T[1])
	plt.show()
	return True

def e_d(a,b):
	#insert data metic
	if len(a) != len(b):
		return False
	else:
		return np.sqrt(sum([(a[i]-b[i])**2 for i in xrange(len(a))]))

def get_sim(samples):
	N = len(samples)
	S = np.zeros([N,N])
	S = np.array([[np.exp(-e_d(a,b)/5.0) for a in samples] for b in samples])
	return S

def plot_clusters(data,P):
	DP = discretize_P(P)
	pot_cols = np.array([rainbow(P.shape[1])]*len(data))
	colors = pot_cols[DP==1.]
	plt.scatter(data.T[0],data.T[1],c = colors)
	plt.show()

def rainbow(N):
	from pylab import get_cmap
	cmap = get_cmap('Spectral')
	return [cmap(1.*k/N) for k in xrange(N)]

def main():
	data = produce_sample_data(50,40,50)
	S = get_sim(data)
	N_C = 3 # number of clusters

	Dist = info_cluster(S,N_C)

	plot_clusters(data,Dist)


if __name__ == "__main__":
	main()