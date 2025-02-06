import os
import numpy as np
import csv
from general_utility import *
import pickle
from sklearn.mixture import GaussianMixture
import random
from scipy.fft import *

def read_map(map_path):
    with open(map_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter='\t')
        map = [row for row in reader]
    return map

def get_attribute(id, attr, map):
    for row in map:
        if id in row['id_scan']:
            return row[attr]

def filter(points, attr, val):

    has = []
    hasnt = []
    for point in points:

        if val.lower() in getattr(point, attr).lower(): has.append(point)
        else: hasnt.append(point)

    return has, hasnt

def load(filepath):

    with open(filepath, "rb") as file:
        return pickle.load(file)

def save(instance, filepath):

    with open(filepath, "wb") as file:
        pickle.dump(instance, file)

def find_same_attribute(points):

    common_attrs = vars(points[0]).keys()

    for attr in common_attrs:
        if attr == 'scan_type': continue
        first_val = getattr(points[0], attr)
        
        try:
            if all(getattr(point, attr) == first_val for point in points):
                return attr, first_val
        except ValueError as e:
            continue

    return None, None

def sample(points, f=0.05):
    
    sample_size = int(len(points) * f)
    sampled_points = random.sample(points, sample_size)
    
    return sampled_points

def remove_empty(points, func=np.mean, remove_below_threshold=True):

    processed_ys = np.array([func(point.y) for point in points]).reshape(-1, 1)
    
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(processed_ys)
    
    labels = gmm.predict(processed_ys)
    
    cluster_values = gmm.means_.flatten()
    small_label = np.argmin(cluster_values)
    large_label = np.argmax(cluster_values)
    
    remove_label = small_label
    if remove_below_threshold == False:
        remove_label = large_label

    filtered_points = [point for point, label in zip(points, labels) if label != remove_label]
    
    original_count = len(points)
    removed_count = original_count - len(filtered_points)
    removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0
    print(f"Removed {removed_percentage:.2f}% of the data.")
    
    return filtered_points

def fourier(points):

    for point in points:
        N = len(point.y)

        frequencies = rfftfreq(N, d=(point.x[1] - point.x[0]))
        # amplitudes = rfft(invert(point.y)) * 2.0 / N 
        amplitudes = rfft(point.y) * 2.0 / N 

        point.x = frequencies
        point.y = amplitudes

    return points

def update_points(points, point_class, params = None):

    new_points = []

    for point in points:

        new_params = point.__dict__
        if params is not None: new_params.update(params)

        new_points.append(point_class(new_params))
        
    return new_points
