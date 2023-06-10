import numpy as np
import matplotlib.pyplot as plt
# from tslearn.clustering import KShape, TimeSeriesKMeans
# from tslearn.preprocessing import TimeSeriesScalerMeanVariance


def make_test_data(motif_length = 100):
    n_dims = 3
    motif1 = np.sin(np.linspace(0, np.pi, motif_length))
    motif2 = np.sin(np.linspace(0, 2*np.pi, motif_length))
    data = np.random.uniform(size = (motif_length, n_dims))
    for dim in range(n_dims):
        if np.random.random() <= 0.5:
            data[:,dim] += motif1
        else:
            data[:,dim] += motif2
    return data.T    


def plot_output(model, name):
    #create a figure
    fig, ax = plt.subplots(3,8, sharex = True, sharey = True, figsize = (10,10))
    fig.subplots_adjust(hspace = 0, wspace = 0)
    fig.suptitle(name)
    
    #set ylabels
    for i in range(3):
        ax[i,0].set_yticks([])
        ax[i,0].set_ylabel(f"dim {i}")
    
    #plot each motif on the axes associated with the cluster label.
    for motif, l in zip(motifs, model.labels_):
        for i, trace in enumerate(motif.T):
            ax[i,l].plot(trace, alpha = 0.05, lw = 0.2, c = "gray")
    
    #plot the cluster centers on top in red
    for i, center in enumerate(model.cluster_centers_):
        ax[0, i].set_title(f"cluster {i}")
        for j in range(np.min(center.shape)):
            ax[j,i].plot(center[:,j], c = "r", alpha = 0.8, lw = 0.5)

#generate test data
motifs = np.dstack([make_test_data() for _ in range(1000)]).T
print(motifs.shape)
#fit TimeSeriesKMeans model for comparison
# kmeans = TimeSeriesKMeans(n_clusters = 8, n_init = 5, max_iter = 100)
# kmeans.fit(motifs)

# #fit KShape model
# kshapes = KShape(n_clusters = 8, n_init = 5, max_iter = 100)
# kshapes.fit(motifs)

# models = {"kmeans": kmeans,
#           "kshapes": kshapes}
# #Visualize the models
# for model in models.keys():
#     plot_output(models[model], model)