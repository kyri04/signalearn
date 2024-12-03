import os
import numpy as np
import csv
from general_utility import *
import pickle
from scipy.signal import argrelextrema
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

def remove_empty(points):

    y = [point.y for point in points]

    mean_ys = []
    for i in range(len(y)):
        mean_ys.append(np.mean(y[i]))

    counts, bin_edges = np.histogram(mean_ys, bins=3000)
    smoothed_counts = np.convolve(counts, np.ones(10)/10, mode='same')

    minima_indices = argrelextrema(smoothed_counts, np.less)[0]

    if len(minima_indices) > 0:
        min_index = minima_indices[0]
        threshold = bin_edges[min_index]
    else:
        threshold = np.min(mean_ys)

    filtered_points = [point for point, mean_y in zip(points, mean_ys) if mean_y >= threshold]

    original_count = len(points)
    filtered_count = len(filtered_points)
    removed_count = original_count - filtered_count
    removed_percentage = (removed_count / original_count) * 100 if original_count > 0 else 0

    print(f"Removed {removed_percentage:.2f}% of points.")

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