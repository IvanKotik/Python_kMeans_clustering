import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(888)


# Sample random X coordinates
sample_1 = np.abs(np.random.randn(100) * 10)
sample = pd.DataFrame(sample_1.T)

# Sample random Y coordinates
sample_2 = np.abs(np.random.randn(100) * 7 + 50)
sample[1] = sample_2


# Defining the distance function
def euclidian_distance(a, b):
    """Calculate euclidian distance between observation a and b dimensions[observations x variables]"""
    return np.sqrt(np.dot((a - b)**2, pd.DataFrame([1, 1])))


# Consecutive iterations
def clustering(sample, means, iterations):
    """Put in number of iterations, give the up-to-date centroid list and sample"""

    # Set initial random centroids
    centroids_1 = np.random.choice(sample[0], means)
    centroids = pd.DataFrame(centroids_1)
    centroids_2 = np.random.choice(sample[1], means)
    centroids[1] = centroids_2

    # Create an empty array for groupings
    groupings = sample[0] * 0

    # Create an empty array for centroids
    centroids_dup = centroids[0] * 0

    # First grouping
    for i in range(0, len(sample_1)):
        tempvari = np.array(sample[i: i + 1])
        for j in range(0, means):
            tempvarj = np.array(centroids[j: j + 1])
            centroids_dup[j] = euclidian_distance(tempvari, tempvarj)
        min_index = np.where(centroids_dup == min(centroids_dup))
        groupings[i] = min_index[0]

    global sample_side
    sample_side = sample.copy()
    sample_side[2] = groupings

    global centroids_update
    centroids_update = sample_side.groupby(2).mean()

    # Continuous iterations
    count = 0
    while count <= iterations:
        centroids = centroids_update

        centroids = centroids.copy()
        for i in range(0, len(sample_1)):
            tempvari = np.array(sample[i: i + 1])
            for j in range(0, means):
                tempvarj = np.array(centroids[j: j])
                centroids_dup[j] = euclidian_distance(tempvari, tempvarj)
            min_index = np.where(centroids_dup == min(centroids_dup))
            groupings[i] = min_index[0]
        sample_side = sample.copy()
        sample_side[2] = groupings
        centroids_update = sample_side.groupby(2).mean()
        print("iteration number:", count, "\n", centroids_update)
        count += 1
        print("convergence:", ((centroids - centroids_update).all() == 0)[0] == True, "\n")
        if ((centroids - centroids_update).all() == 0)[0] == True:
            break


clustering(sample, 3, 15)

#def sum_calculations(sample, num_means):

# Prepare the centroids to be merged with sample
centroids_update[2] = centroids_update.index
centroids_update.columns = ["x-axis", "y-axis", "groups"]
sample_side.columns = ["x-axis", "y-axis", "groups"]
sample_check = sample_side.merge(centroids_update, on="groups")

# Creation of the sum column for further usage
sample_check["ersum"] = np.sqrt((sample_check["x-axis_x"]-sample_check["x-axis_y"])**2 + (sample_check["y-axis_x"]-sample_check["y-axis_y"])**2)
total_ersum = np.sum(sample_check["ersum"])


means = 10
mean_relationship = [np.nan]*means

for i in range(1, means):
    clustering(sample, i, 15)
    # Prepare the centroids to be merged with sample
    centroids_update[2] = centroids_update.index
    centroids_update.columns = ["x-axis", "y-axis", "groups"]
    sample_side.columns = ["x-axis", "y-axis", "groups"]
    sample_check = sample_side.merge(centroids_update, on="groups")

    # Creation of the sum column for further usage
    sample_check["ersum"] = np.sqrt((sample_check["x-axis_x"] - sample_check["x-axis_y"]) ** 2 + (
    sample_check["y-axis_x"] - sample_check["y-axis_y"]) ** 2)
    total_ersum = np.sum(sample_check["ersum"])
    mean_relationship[i] = total_ersum


plt.plot(mean_relationship)
plt.title("Elbow method of determining mean-choice")
plt.ylabel("Total euclidian distance between observations and their respected group means")
plt.xlabel("Number of means")
plt.show()
