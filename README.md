# Deep Embedded Clustering
Implementation of DEC (https://arxiv.org/abs/1511.06335) in Keras, work in progress.
## Debug branch
This branch is for experimentation.

### Experiments
The following experiments test the robustness of DEC.
Failure of these tests signify the need for hyperparameter adjustments.
 - Try constant $\mu$ (i.e. don't update cluster centroids)
 - Try $\mu_1 = [1, 0, 0, ..., 0], \mu_2 = [0, 1, 0, ..., 0]$, ...