import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(888)

# Setting initial means
means = 5

# Sample random X coordinates
sample_1 = np.abs(np.random.randn(1000)*10)
sample = pd.DataFrame(sample_1.T)

# Sample random Y coordinates
sample_2 = np.abs(np.random.randn(1000)*7+50)
sample[1] = sample_2

# Set initial random centroids
centroids_1 = np.random.choice(sample[0], means)
centroids = pd.DataFrame(centroids_1)
centroids_2 = np.random.choice(sample[1], means)
centroids[1] = centroids_2

# Create an empty array for groupings
groupings = sample[0]*0

# Create an empty array for centroids
centroids_dup = centroids[0]*0


# Defining the distance function
def euclidian_distance(a, b):
    """Calculate euclidian distance between observation a and b dimensions[observations x variables]"""
    return np.sqrt(np.dot((a - b)**2, pd.DataFrame([1, 1])))


# Define first iteration grouping function
def get_groups(sample, centroid):
    """Assign mean groups based on [n x 2] input vector and [n x 2] centroid vector. Returns groupings separately."""
    for i in range(0, len(sample_1)):
        tempvari = np.array(sample[i: i+1])
        for j in range(0, means):
            tempvarj = np.array(centroid[j: j+1])
            centroids_dup[j] = euclidian_distance(tempvari, tempvarj)
        min_index = np.where(centroids_dup == min(centroids_dup))
        groupings[i] = min_index[0]
    return groupings


# Define grouping function
def get_groups2(sample, centroid):
    """Assign mean groups based on [n x 2] input vector and [n x 2] centroid vector. Returns groupings separately."""
    for i in range(0, len(sample_1)):
        tempvari = np.array(sample[i: i+1])
        for j in range(0, means):
            tempvarj = np.array(centroid[j: j])
            centroids_dup[j] = euclidian_distance(tempvari, tempvarj)
        min_index = np.where(centroids_dup == min(centroids_dup))
        groupings[i] = min_index[0]
    return groupings


# Define convergence function
def convergence_test(previous_iteration, new_iteration):
    """Returns a boolean whether the two values have converged"""
    return (previous_iteration - new_iteration).all() == 0


# Define main clustering function
def clustering(samplee, centroid):
    """"Put in the sample and the latest centroid array"""
    centroids = centroid.copy()
    get_groups2(samplee, centroid)
    sample_side = sample.copy()
    sample_side[2] = groupings
    centroids_update = sample_side.groupby(2).mean()
    return centroids_update


# Initial group assignment, initial centroid iteration
get_groups(sample, centroids)
sample_side = sample.copy()
sample_side[2] = groupings
centroids_update = sample_side.groupby(2).mean()

# Consecutive iterations
iterations = 10  # Select the number of iterations
count = 0
while count <= iterations:
    centroids = centroids_update
    centroids_update = clustering(sample, centroids_update)
    print("iteration number:", count, "\n", centroids_update)
    count += 1
    print("convergence:", convergence_test(centroids, centroids_update)[0] == True, "\n")
    if convergence_test(centroids, centroids_update)[0] == True:
        break

# Plotting the result
get_groups2(sample, centroids_update)
sample_side = sample.copy()
sample_side[2] = groupings
plt.scatter(sample_side[0], sample_side[1], c=sample_side[2])
plt.show()