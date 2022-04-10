from util import *
import numpy as np
from tqdm import tqdm
import random


def q_dist(z, 
           mu, 
           alpha=alpha):
    """
    INPUTS:     
        z:      Collection of representations for z (representation of x in embedded space).
        mu:     Vector representation of cluster centroids.
        alpha:  Degrees of freedom for Student's t-distribution. Default value is 1.
    RETURNS:    
        q:      Matrix with computed q distributions.
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
    INPUTS:     
        q:      Matrix with computed q distributions from q_dist().
    RETURNS:    
        p:      Matrix with computed p distributions.
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
    Calculate KL Divergence between P and Q, where P, Q are i x j matrices.
    INPUTS: 
        p:      Distribution
        q:      Distribution
    RETURNS:    
        KL Divergence between p and q
    """
    p = np.asarray(p); q = np.asarray(q)
    return sum([sum([p[i][j] * np.log(p[i][j]/q[i][j]) for j in range(0, p.shape[1])]) for i in range(0, p.shape[0])])
    

def column(matrix, i):
    """
    Gets column of matrix. 
    INPUTS:     
        Matrix, Int of column to look at
    RETURNS:    
        Array of the column
    """
    return [row[i] for row in matrix]

def sample_dataset(x,
                   y,
                   no_samples):
    """
    """
    dataset = []
    dataset_labels = []
    if y != []:
        unique_y = set(y)
        samples_per_y = int(no_samples/len(unique_y))
                
        for label in unique_y:
            extracted_indices = random.sample([i for i, y_label in enumerate(y) if y_label == label], 
                                              samples_per_y)
            dataset.append([x[index] for index in extracted_indices])
            dataset_labels += [label] * samples_per_y
            
        return np.asarray(dataset), dataset_labels
    else:
        return np.asarray(random.sample(x, samples_per_y)), y