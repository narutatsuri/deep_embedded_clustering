import plotly.express as px
from util import *
from util.functions import *
import pandas as pd

    
def plot_embeddings(plot_result, 
                    embeddings,
                    cluster_centroids,
                    labels,
                    name=None):
    """
    
    INPUTS:     

    RETURNS:    
        None
    """
    if plot_result:
        if representation_dim == 2:
            plot_embeddings_2d(embeddings=embeddings,
                               cluster_centroids=cluster_centroids,
                               labels=labels)
        elif representation_dim == 3:
            plot_embeddings_3d(embeddings=embeddings,
                               cluster_centroids=cluster_centroids,
                               labels=labels)
    else:
        if representation_dim == 2:
            save_embeddings_2d(embeddings=embeddings,
                               cluster_centroids=cluster_centroids,
                               labels=labels,
                               name=name)
        # elif representation_dim == 3:
        #     save_embeddings_3d(embeddings=embeddings,
        #                        cluster_centroids=cluster_centroids,
        #                        labels=labels)

def plot_embeddings_2d(embeddings,
                       cluster_centroids,
                       labels):
    """
    
    INPUTS:

    RETURNS:
        None
    """
    df = pd.DataFrame(columns=["x", "y", "label"])

    for index, point in enumerate(embeddings):
        x = np.array(point[0])
        y = np.array(point[1])
        df = df.append(pd.DataFrame([[x, y, labels[index]]], 
                                    columns=["x", "y", "label"]))
    
    # for index, point in enumerate(cluster_centroids):
    #     x = np.array(point[0])
    #     y = np.array(point[1])
    #     df = df.append(pd.DataFrame([[x, y, cluster_num+1]], 
    #                                 columns=["x", "y", "label"]))

    fig = px.scatter(df, x="x", y="y", color="label")

    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.show()
    
def save_embeddings_2d(embeddings,
                       cluster_centroids,
                       labels,
                       name):
    """
    
    INPUTS:

    RETURNS:
        None
    """
    df = pd.DataFrame(columns=["x", "y", "label"])

    for index, point in enumerate(embeddings):
        x = np.array(point[0])
        y = np.array(point[1])
        df = df.append(pd.DataFrame([[x, y, labels[index]]], 
                                    columns=["x", "y", "label"]))
    
    # for index, point in enumerate(cluster_centroids):
    #     x = np.array(point[0])
    #     y = np.array(point[1])
    #     df = df.append(pd.DataFrame([[x, y, cluster_num+1]], 
    #                                 columns=["x", "y", "label"]))

    fig = px.scatter(df, x="x", y="y", color="label")

    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.write_image(save_fig_dir + name + ".png")
    
    
def plot_embeddings_3d(embeddings,
                       cluster_centroids,
                       labels):
    """
    
    INPUTS:     
    RETURNS:    None
    """
    df = pd.DataFrame(columns=["x", "y", "z", "label"])

    for index, point in enumerate(embeddings):
        x = np.array(point[0])
        y = np.array(point[1])
        z = np.array(point[2])
        df = df.append(pd.DataFrame([[x, y, z, str(index+1)]], 
                                    columns=["x", "y", "z", "label"]))
        
    # for index, point in enumerate(cluster_centroids):
    #     x = np.array(point[0])
    #     y = np.array(point[1])
    #     z = np.array(point[2])        
    #     df = df.append(pd.DataFrame([[x, y, z, cluster_num+1]], 
    #                                 columns=["x", "y", "z", "label"]))

    fig = px.scatter_3d(df, x="x", y="y", z="z", color="label")
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.show()