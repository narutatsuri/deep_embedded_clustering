from util.functions import *
from util import *
import numpy as np
from tqdm import tqdm


def q_dist(z, 
           mu, 
           alpha=alpha):
    """
    z_i: Vector representation for z_i (representation of x_i in embedded space)
    mu: Vector representation of cluster centroids
    alpha: Degrees of freedom for Student's t-distribution. Default value is 1.
    """
    q = np.empty((z.shape[0], mu.shape[0]), dtype=np.float64)
    z = np.asarray(z); mu = np.asarray(mu)
    for i, z_i in enumerate(z):
        denominator = sum([(1 + (np.linalg.norm(z_i - mu_j_prime)**2)/alpha)**(-(alpha+1)/2) for mu_j_prime in mu])
        for j, mu_j in enumerate(mu):
            q[i][j] = (1 + (np.linalg.norm(z_i - mu_j)**2)/alpha)**(-(alpha+1)/2)/denominator
            
    return q

def p_dist(q):
    """
    """
    q = np.asarray(q)
    p = np.empty(q.shape, dtype=np.float64)
    for index_i in (range(q.shape[0])):
        for index_j in (range(q.shape[1])):
            f_j = sum(column(q, index_j))
            q_ij = q[index_i][index_j]
                        
            p[index_i][index_j] = (q_ij**2/f_j)/sum([q[index_i][j]**2/sum(column(q, j)) for j in range(q.shape[1])])
            
    return p


def kl_div(p,
           q):
    """
    Calculate KL Divergence between P and Q, where P, Q are R^i*j matrices.
    """
    p = np.asarray(p); q = np.asarray(q)
    return sum([sum([p[i][j] * np.log(p[i][j]/q[i][j]) for j in range(0, p.shape[1])]) for i in range(0, p.shape[0])])
    