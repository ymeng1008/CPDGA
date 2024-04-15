import pandas as pd

"""
def count_overlap_points(file1_path, file2_path):
    # Read the CSV files into DataFrames
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # Create sets of points using the 'Dimension 1' and 'Dimension 2' columns
    points_set1 = set(zip(df1['Dimension 1'], df1['Dimension 2']))
    points_set2 = set(zip(df2['Dimension 1'], df2['Dimension 2']))

    # Calculate the intersection of the two sets
    overlap_points = points_set1.intersection(points_set2)

    # Print the count of overlapping points
    print(f"Number of overlapping points: {len(overlap_points)}")

# Example usage:
file1_path = 'consistent_points_1.csv'
file2_path = 'consistent_points_2.csv'
count_overlap_points(file1_path, file2_path)
"""
import csv
import numpy as np
from sklearn.metrics import pairwise_distances

# 读取CSV文件并获取点的列表
def read_csv(file_path):
    points = []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过标题行
        for row in reader:
            x, y = map(float, row)
            points.append([x, y])
    return np.array(points)

# 定义网格单元的距离阈值
grid_cell_distance_threshold = 0.5037436712355841  # 你需要设置合适的值

# 读取两个CSV文件中的点
points1 = read_csv('consistent_points_1.csv')
points2 = read_csv('consistent_points_2.csv')

# 计算点之间的欧几里德距离
distances_to_cluster = pairwise_distances(points2, points1, metric='euclidean')

# 找到每个点与consistentpoints1中的某个点的最小距离
min_distances = np.min(distances_to_cluster, axis=1)

# 统计满足条件的点对数量
result_count = np.sum(min_distances <= grid_cell_distance_threshold)

# 输出结果
print(f"与consistentpoints1中的某个点的距离小于或等于 {grid_cell_distance_threshold} 的consistentpoints2中的点的数量为: {result_count}")
