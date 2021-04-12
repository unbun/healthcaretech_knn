# Unnas Hussain
# 04/12/201
# modified from https://machinelearningmastery.com/tutorial-to-implement-k-nearest-neighbors-in-python-from-scratch/

from collections import Counter
import math
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import csv
known_angle_column = 1;
flexion_power_column = 3;
extension_power_column = 4;

EXTENSION_SETTINGS = {
    'plt_color' : 'blue',
    'plt_title' : 'Extension KNN',
    'plt_type' : '^',
    'min' : 0,
    'max' : 80
}

FLEXION_SETTINGS = {
    'plt_color' : 'red',
    'plt_title' : 'Flextion KNN',
    'plt_type' : 'P',
    'min' : -80,
    'max' : 0
}

def knn(data, query, k, distance_fn, choice_fn, settings=FLEXION_SETTINGS):
    neighbor_distances_and_indices = []

    xs = []
    ys = []
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        if settings['min'] <= example[1] <= settings['max']:
            distance = distance_fn(example[:-1], query)

            # 3.2 Add the distance and the index of the example to an ordered collection
            neighbor_distances_and_indices.append((distance, index))
            xs.append(example[0])
            ys.append(example[1])

    plt.plot(xs, ys, settings['plt_type'], color=settings['plt_color'])
    plt.title(settings['plt_title'])
    plt.show()

    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)

    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]

    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][-1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices, choice_fn(k_nearest_labels)


def mean(labels):
    return sum(labels) / len(labels)


def mode(labels):
    return Counter(labels).most_common(1)[0][0]


def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(float(point1[i]) - float(point2[i]), 2)
    return math.sqrt(sum_squared_distance)


def main():
    '''
    # Regression Data
    #
    # Column 0: power (input)
    # Column 1: angle (predicition)  + --> ext, - --> flex
    '''

    # trial, angle, calculated angle, flexion power, extension power

    file = open("training_data1.csv", encoding='utf-8-sig')
    wrist_data_n = np.loadtxt(file, delimiter=',')


    flex_slice = wrist_data_n[1:, [flexion_power_column, known_angle_column]]
    flex_reg_data = list(flex_slice);

    givenPower = input("Input total power value: ") # float(.01102960);
    reg_query = [givenPower]
    K = 4

    reg_k_nearest_neighbors, flex_reg_prediction = knn(
        flex_reg_data, reg_query, k=K, distance_fn=euclidean_distance, choice_fn=mean, settings=FLEXION_SETTINGS
    )

    ext_slice = list(wrist_data_n[1:, [extension_power_column, known_angle_column]])
    ext_reg_data = list(ext_slice);

    reg_k_nearest_neighbors, ext_reg_prediction = knn(
        ext_reg_data, reg_query, k=K, distance_fn=euclidean_distance, choice_fn=mean, settings=EXTENSION_SETTINGS
    )

    print("If flexion: ", flex_reg_prediction, "\nIf extension: ", ext_reg_prediction);


if __name__ == '__main__':
    main()