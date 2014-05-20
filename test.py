import numpy as np
from numpy.random import multivariate_normal as mn
import matplotlib.pyplot as plt
from pylab import get_cmap
import info_clust as ic

def produce_sample_data(N1,N2,N3):
	Means = np.array([[-10,-3],[0,0],[10,10]])
	Vars = np.array([[[1,0.5],[0.5,0.7]],[[2,-2.4],[-2.4,1]]])
	return np.vstack([mn(Means[0],Vars[0],N1),mn(Means[1],Vars[1],N2),mn(Means[2],Vars[1],N3)])

def d(a,b):
	#insert metic
	dim = len(a)
	return np.exp(-np.sqrt(sum([(a[i]-b[i])**2 for i in xrange(dim)]))/5.0)

def get_sim(samples):
	return np.array([[d(a,b) for a in samples] for b in samples])

def plot_clusters(data,P):
	DP = ic.discretize_P(P)
	pot_cols = np.array([rainbow(P.shape[1])]*len(data))
	colors = pot_cols[DP==1.]
	plt.scatter(data.T[0],data.T[1], c = colors)
	plt.show()

def rainbow(N):
	cmap = get_cmap('Spectral')
	return [cmap(1.*k/N) for k in xrange(N)]

def main():
	data = produce_sample_data(50,40,50)
	S = get_sim(data)
	N_C = 3 # number of clusters
	Dist = ic.info_cluster(S,N_C)
	plot_clusters(data,Dist)

if __name__ == "__main__":
	main()