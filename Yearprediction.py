"""
2023.12.11
尝试不同年份的恶意域名Conformal Prediction之后是否有范围的重合
例如:
1.2019年恶意url+2020年恶意url各20000个一起降维（T-SNE/UMAP）
  可视化一下看大致的范围，用不同颜色，那这里的标签应该选用年份
2.Conformal Prediction ，两个年份的数据用不同颜色
"""
import re
import pickle
import math
import wordfreq
import operator
import string
import tld
import csv
import numpy as np
import pandas as pd
from configparser import ConfigParser
from collections import Counter, defaultdict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import LabelEncoder
# import umap
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import norm
from matplotlib.colors import LinearSegmentedColormap, Normalize

data_tsne_global = None
data_tsne = None
data_tsne_global2 = None
data_tsne_Unknown = None

# T-SNE降维
def traindataset_tSNE(csv_file_path):
    # 加载CSV文件
    # 读取训练集数据
    # global data_tsne  
    df = pd.read_csv(csv_file_path)
    
    # /home/dym/恶意域名检测/代码/web_dga/data/features/raw_test_features.csv
    # 提取特征列
    data = df.iloc[:, 2:]  # 假设前两列是标签和域名，所以从第3列开始提取特征
    # 假设标签存储在 'label' 列中
    labels = df["label"]

    # 创建一个t-SNE模型
    # n_components：指定要降维到的目标维度数。
    # random_state：这个参数用于控制随机性。
    # perplexity：高维空间中每个对象的有效邻居数量
    # distance metric：距离度量，'euclidean' 欧氏距离
    tsne = TSNE(n_components=2, perplexity=100, metric='euclidean',random_state=42,learning_rate=1500)
    # 对数据进行降维
    data_tsne = tsne.fit_transform(data)

    # 创建一个新的DataFrame包含降维后的数据
    tsne_df = pd.DataFrame(data_tsne, columns=["Dimension 1", "Dimension 2"])
    tsne_df["Label"] = labels
    # 不需要保留原始域名列，因为 t-SNE 主要关注数据的分布和相似性，而不是域名本身

    # 可视化
    plt.figure(figsize=(8, 6))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    # 根据标签着色数据点
    for i, label in enumerate(df['label'].unique()):
        plt.scatter(tsne_df.loc[tsne_df['Label'] == label, 'Dimension 1'],
                tsne_df.loc[tsne_df['Label'] == label, 'Dimension 2'],
                c=colors[i],
                label=label,
                s=5)

    plt.title('t-SNE Visualization')
    # 二维投影
    # 生成的二维散点图反映了原始高维数据点之间的相似性和距离关系
    # 保留数据点之间的相似性关系
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('TSNE_2019and2020.png')
    plt.show()

    """
    降维后使用DBSCAN进行聚类
    
    """
    # 初始化DBSCAN模型
    dbscan = DBSCAN(eps=5, min_samples=10)
    # 对降维后的数据进行DBSCAN聚类
    labels_p = dbscan.fit_predict(data_tsne)
    # 将聚类结果添加到原始数据中

    data["Cluster"] = labels_p
    # 绘制散点图，每个聚类用不同颜色表示
    plt.scatter(data_tsne[:, 0], data_tsne[:, 1], c=labels_p)
    plt.title("t-SNE with DBSCAN Clustering")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig('TSNE_DBSCAN_2019and2020.png')
    plt.show()
    # 返回降维后的结果
    return data_tsne


# 非一致性度量的两个函数
# 基于k-最近邻（k-NN）的非一致性度量
def k_nn_nonconformity_measure(D, zi, k):
    """
    训练对象集合D和新对象z
    k（邻居数量）
    计算非一致性度量Ai
    """
    # 计算对象zi与其他对象的距离
    dist = np.linalg.norm(D - zi, axis=1)
    # 对距离进行排序，找到第k个最近邻的距离
    k_nearest_distances = np.partition(dist, k)[:k]
    # 计算非一致性度量Ai
    Ai = np.sum(k_nearest_distances)
    return Ai

# 基于核密度估计（KDE）的非一致性度量函数
def kde_nonconformity_measure(D, zi, bandwidth):
    """
    训练对象集合D
    新对象zi，
    以及核带宽（bandwidth），不知道取多少合适
    计算非一致性度量Ai
    """
    # 初始化核密度估计器
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    # 拟合核密度估计器到训练对象集合D
    kde.fit(D)
    # 计算新对象zi的核密度分数
    log_density = kde.score_samples(zi.reshape(1, -1))
    # 计算非一致性度量Ai
    Ai = -log_density
    return Ai





def calculate_p_value(D, A, epsilon, z):
    """
    输入：
    一组对象的集合D，z1到zn-1
    非一致度量函数A(KNN和KDE)
    两种方法后续再补充进来
    显著水平
    新对象z(也就是每个格点)
    输出：
    p_value
    布尔值表示是否与训练对象一致
    """
    # 初始化P-Value pn
    pn = 0.0

    # 步骤1：设置zn为新对象z，并扩展D
    D = np.vstack([D, z])
    # 步骤2：计算每个对象zi与其他对象的不一致性度量αi
    # alphas 是一个空列表，用于存储每个对象zi与其他对象的不一致性度量αi
    alphas = []
    print(len(D)) # 增加了一个点应该是316个点了

    for i in range(len(D)):
        # 移除对象zi并计算αi
        D_minus_i = np.delete(D, i, axis=0)
        # 衡量不一致性
        # 这里有两个方法，直接用A的值来判断
        if(A==0):
            alpha_i = k_nn_nonconformity_measure(D_minus_i, D[i],1)
        else:
            alpha_i = kde_nonconformity_measure(D_minus_i, D[i],0.1)
            
        # 记录了训练对象中每个对象与其他对象的不一致性度量
        # print(f"alpha_i = {alpha_i}")
        alphas.append(alpha_i)
    
    # 步骤3：生成随机数τ，介于0和1之间的随机值
    tau = np.random.uniform(0, 1)

    # 步骤4：计算P-Value pn
    alpha_n = alphas[-1]  # 新对象z的不一致性度量αn
    # 满足 αi > αn 的i的数量
    num_alpha_gt_alpha_n = np.sum(np.array(alphas) > alpha_n)
    # 满足 αi = αn 的i的数量
    num_alpha_eq_alpha_n_and_i_lt_n = np.sum(np.array(alphas) == alpha_n)
    # 根据公式计算p_value
    pn = (num_alpha_gt_alpha_n + num_alpha_eq_alpha_n_and_i_lt_n * tau) / len(D)
    
    # 步骤5：判断是否与训练对象一致
    is_consistent = pn > epsilon
    
    return pn, is_consistent












def CP_cluster():
    # 步骤1：创建d维格点网格
    # 因为已经降维过了，所以此时维度=2
    # features是不是应该指向降维后的特征点，(n_samples, n_features),(对象数，2（已经降维过的）)
    # 应该就是降维后的data_tsne
    features= data_tsne
    print(features.shape)
    # 获取数据的形状
    num_objects, num_features = features.shape
    # 每个特征的值范围内，有多少个格点要生成，需要后续再调整
    num_points_per_dimension=1000
    d = 2  # 特征的数量
    print(d)
    # 存储每个特征的格点
    
    all_grid_points = []
    # 特征应该只有两个
    for feature_index in range(num_features):
        feature = features[:, feature_index]  # 获取一个特征的数据
        # 对于每个特征，分别计算该特征的最小值和最大值。确定特征值的范围。
        
        # 计算该特征的最小值和最大值
        min_val, max_val = np.min(feature), np.max(feature)
        
        print(min_val)
        print(max_val)
        
        # 将特征值范围分成若干等间距的格点
        num_points_per_dimension = int((max_val - min_val) / 0.5)  # 10是一个可以调整的参数，表示每个特征上的格点数量
        print(num_points_per_dimension)
        # np.linspace() 函数，创建一个在最小值和最大值之间等间距分布的一维数组
        grid = np.linspace(min_val, max_val, num=num_points_per_dimension)
        all_grid_points.append(grid)
    # 使用np.meshgrid函数将 grid_points 列表中的所有特征的格点组合成一个d维格点网格
    
    # 生成格点网络
    grid_points = np.array(np.meshgrid(*all_grid_points)).T.reshape(-1, num_features)
    grid_cell_distance_threshold =  (max_val - min_val) / num_points_per_dimension
    # 可视化一下
    print("当前网格单元距离阈值为:", grid_cell_distance_threshold)
    # 可视化结果
    plt.figure(figsize=(8, 6))

    # 绘制格子线
    for i in range(num_features):
        for point in all_grid_points[i]:
            plt.axvline(point, color='gray', linestyle='--', linewidth=0.5)

    # 绘制格点
    plt.plot(grid_points[:, 0], grid_points[:, 1], 'rx', markersize=8)  # 使用"x"形状，设置标记大小

    # 设置坐标轴标签
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    # 设置图标题
    plt.title('2D Grid Points')

    # 显示格子图
    plt.grid(True)
    plt.show()


    # 步骤2：计算P-Values
    # 存储每个格点的p_value
    p_values = []
    # 遍历了经过整形的 grid_coordinates 
    # grid_coordinates 本来是一个 d 维网格，但在这里通过 reshape(-1, d) 被重新整形成一个一维数组，以便逐个访问每个格点。
    
    for grid_point in grid_points.reshape(-1, d):
        # 使用CP算法计算P-Value，将grid_point视为新对象z
        # 一组对象的集合D可以直接用data_tsne表示吗
        # calculate_p_value()有四个参数：data_tsne、A(准备直接用整数来指示两个不同函数)、epsilon（暂时取0.05？）、grid_point
        print(f"grid_point = {grid_point}")
        p_value = calculate_p_value(data_tsne_global,0,0.05,grid_point)
        # p_value = calculate_p_value(data_umap_global,0,0.05,grid_point)
        print(p_value) 
        p_values.append(p_value)
    
    # 为 data_tsne_global2 进行一致性预测
    p_values2 = []
    for grid_point in grid_points.reshape(-1, d):
        p_value = calculate_p_value(data_tsne_global2, 0, 0.05, grid_point)
        p_values2.append(p_value)



    # 步骤3：一致性预测
    # 需要可视化一致性预测结果

    epsilon = 0.05  # 显著性值ε,出错率，可以修改进行对比
    # consistent_points,存储一致性预测为一致的格点
    # grid_coordinates.reshape(-1, d)将之前创建的一维格点坐标重新整形为一个 d 维数组，然后使用 zip 函数将格点坐标和对应的 P-Values 进行配对
    # zip(grid_coordinates.reshape(-1, d), p_values)是将格点坐标和对应的P-Value（概率值）一一对应起来，这样每个格点都有一个对应的P-Value。

    # 一致性的格点，小于设定的最低阈值就不显示颜色了
    # 一致性的格点，小于设定的最低阈值就不显示颜色了
    consistent_points = [
        point for point, (p_value, is_consistent) in zip(grid_points.reshape(-1, d), p_values) 
        if is_consistent]
    x_coords, y_coords = zip(*consistent_points)

    df = pd.DataFrame(consistent_points, columns=['Dimension 1', 'Dimension 2'])

    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('consistent_points_1.csv', index=False)

    print(f"Consistent points saved to consistent_points_1.csv")
    consistent_points2 = [
        point for point, (p_value, is_consistent) in zip(grid_points.reshape(-1, d), p_values2) 
        if is_consistent]
    x_coords2, y_coords2 = zip(*consistent_points2)

    df2 = pd.DataFrame(consistent_points2, columns=['Dimension 1', 'Dimension 2'])

    # 将 DataFrame 存储为 CSV 文件
    df2.to_csv('consistent_points_2.csv', index=False)

    print(f"Consistent points saved to consistent_points_2.csv")
    # 提取 p_value 值，较大的 p_value 对应较深的颜色
    p_values_for_color = [p_value for p_value, is_consistent in p_values if is_consistent]
    p_values_for_color2 = [p_value for p_value, is_consistent in p_values2 if is_consistent]

    # 创建一个散点图来可视化一致性预测结果，根据 p_value 的大小设置颜色深浅
    cmap = plt.cm.get_cmap('Reds', len(p_values_for_color))
    cmap2 = plt.cm.get_cmap('Blues', len(p_values_for_color2))

    plt.figure(figsize=(20, 16))

    # 绘制 data_tsne_global 的点
    scatter1=plt.scatter(x_coords, y_coords, s=30, c=p_values_for_color, cmap=cmap, edgecolors='k', marker='x', linewidth=0.5, label='2019')

    # 绘制 data_tsne_global2 的点
    scatter2=plt.scatter(x_coords2, y_coords2, s=30, c=p_values_for_color2, cmap=cmap2, edgecolors='k', marker='+', linewidth=0.5, label='2020')

    # 设置图表标题和坐标轴标签
    plt.title('Consistency Prediction Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加颜色条
    cbar1 = plt.colorbar(mappable=scatter1, ticks=np.linspace(0.2, 0.8, 10), label='Color Intensity (p_value) - 2019')
    cbar2 = plt.colorbar(mappable=scatter2, ticks=np.linspace(0.2, 0.8, 10), label='Color Intensity (p_value) - 2020')
    # 显示图例
    plt.legend()

    # 保存图像
    plt.savefig('consistency_prediction_result_2019and2020.png')

    # 显示整体图
    plt.show()
         
    """#步骤4：使用邻近规则进行聚类，这里先暂时选用DBSCAN
    X = np.array(consistent_points)
    distance_matrix = pairwise_distances(X, metric='euclidean')
    db = DBSCAN(eps=10, min_samples=5).fit(distance_matrix)
    
    # 绘制聚类结果
    labels = db.fit_predict(distance_matrix)
    unique_labels = set(labels)

    # 创建一个颜色映射，为每个簇分配不同的颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 8))

    for label, color in zip(unique_labels, colors):
        cluster_mask = labels == label  # 根据聚类标签创建一个布尔掩码
        xy = X[cluster_mask]  # 选择属于当前聚类的点
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=color, label=f'Cluster {label}', edgecolors='k')

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()

    plt.show()
    """


def CP_cluster_2():
    # 步骤1：创建d维格点网格
    # 因为已经降维过了，所以此时维度=2
    # features是不是应该指向降维后的特征点，(n_samples, n_features),(对象数，2（已经降维过的）)
    # 应该就是降维后的data_tsne
    features= data_tsne_global
    # features= data_umap_global
    print(features.shape)
    # 获取数据的形状
    num_objects, num_features = features.shape
    # 每个特征的值范围内，有多少个格点要生成，需要后续再调整
    num_points_per_dimension=1000
    d = 2  # 特征的数量
    print(d)
    # 存储每个特征的格点
    
    all_grid_points = []
    # 特征应该只有两个
    # 特征应该只有两个
    for feature_index in range(num_features):
        feature = features[:, feature_index]  # 获取一个特征的数据
        # 对于每个特征，分别计算该特征的最小值和最大值。确定特征值的范围。
        
        # 计算该特征的最小值和最大值
        min_val, max_val = np.min(feature), np.max(feature)
        
        print(min_val)
        print(max_val)
        
        # 将特征值范围分成若干等间距的格点
        num_points_per_dimension = int((max_val - min_val) / 0.5)  # 10是一个可以调整的参数，表示每个特征上的格点数量
        print(num_points_per_dimension)
        # np.linspace() 函数，创建一个在最小值和最大值之间等间距分布的一维数组
        grid = np.linspace(min_val, max_val, num=num_points_per_dimension)
        all_grid_points.append(grid)
    # 使用np.meshgrid函数将 grid_points 列表中的所有特征的格点组合成一个d维格点网格
    """grid_coordinates = np.meshgrid(*grid_points)
    grid_coordinates = np.stack(grid_coordinates, axis=-1)
    print(grid_coordinates.shape)"""
    # 生成格点网络
    grid_points = np.array(np.meshgrid(*all_grid_points)).T.reshape(-1, num_features)
    grid_cell_distance_threshold =  (max_val - min_val) / num_points_per_dimension
    # 可视化一下
    # 可视化结果
    plt.figure(figsize=(8, 8))

    # 绘制格子线
    for i in range(num_features):
        for point in all_grid_points[i]:
            plt.axvline(point, color='gray', linestyle='--', linewidth=0.5)

    # 绘制格点
    plt.plot(grid_points[:, 0], grid_points[:, 1], 'rx', markersize=8)  # 使用"x"形状，设置标记大小

    # 设置坐标轴标签
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    """plt.scatter(grid_coordinates[:, 0], grid_coordinates[:, 1], marker='o', s=10)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.title('2D Grid Points')
    plt.grid(True)
    plt.show()  """
    # 设置图标题
    plt.title('2D Grid Points')

    # 显示格子图
    plt.grid(True)
    plt.show()


    # 步骤2：计算P-Values
    # 存储每个格点的p_value
    p_values = []
    # 遍历了经过整形的 grid_coordinates 
    # grid_coordinates 本来是一个 d 维网格，但在这里通过 reshape(-1, d) 被重新整形成一个一维数组，以便逐个访问每个格点。
    
    for grid_point in grid_points.reshape(-1, d):
        # 使用CP算法计算P-Value，将grid_point视为新对象z
        # 一组对象的集合D可以直接用data_tsne表示吗
        # calculate_p_value()有四个参数：data_tsne、A(准备直接用整数来指示两个不同函数)、epsilon（暂时取0.05？）、grid_point
        print(f"grid_point = {grid_point}")
        p_value = calculate_p_value(data_tsne_global,0,0.05,grid_point)
        # p_value = calculate_p_value(data_umap_global,0,0.05,grid_point)
        print(p_value) # p_value也算出来了，但是好像死循环了
        p_values.append(p_value)
    
    # 步骤3：一致性预测
    # 需要可视化一致性预测结果

    epsilon = 0.05  # 显著性值ε,出错率，可以修改进行对比
    # consistent_points,存储一致性预测为一致的格点
    # grid_coordinates.reshape(-1, d)将之前创建的一维格点坐标重新整形为一个 d 维数组，然后使用 zip 函数将格点坐标和对应的 P-Values 进行配对
    # zip(grid_coordinates.reshape(-1, d), p_values)是将格点坐标和对应的P-Value（概率值）一一对应起来，这样每个格点都有一个对应的P-Value。

    # 一致性的格点，小于设定的最低阈值就不显示颜色了
    consistent_points = [
    point for point, (p_value, is_consistent) in zip(grid_points.reshape(-1, d), p_values) 
    if is_consistent]
    # 解压缩 x 和 y 坐标
    x_coords, y_coords = zip(*consistent_points)

    df = pd.DataFrame(consistent_points, columns=['Dimension 1', 'Dimension 2'])

    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('consistent_points_1.csv', index=False)

    print(f"Consistent points saved to consistent_points_1.csv")
    # 提取 p_value 值，较大的 p_value 对应较深的颜色
   
    p_values_for_color = [p_value for p_value, is_consistent in p_values if is_consistent]
    
    # 创建一个散点图来可视化一致性预测结果，根据 p_value 的大小设置颜色深浅
    
    # 调整颜色映射，使用更细致的红色渐变
    cmap = plt.cm.get_cmap('Reds', len(p_values_for_color))

    # 创建一个散点图来可视化一致性预测结果
    plt.figure(figsize=(10, 8))
    # 添加颜色条，用于表示颜色和 p 值之间的对应关系
    # cbar = plt.colorbar(ticks=np.linspace(0, 1, len(p_values_for_color)), label='Color Intensity (p_value)')
    # cbar.set_ticklabels([f'{p:.2f}' for p in p_values_for_color])
    

    

    # 添加颜色条
    
    scatter1=plt.scatter(x_coords, y_coords, s=30, c=p_values_for_color, cmap=cmap, edgecolors='k', marker='x', linewidth=0.5, label='2019')
    cbar1 = plt.colorbar(mappable=scatter1,ticks=np.linspace(0.2, 0.8, 10), label='Color Intensity (p_value)')
    scatter2=plt.scatter(data_tsne_global2[:, 0], data_tsne_global2[:, 1],s=15,c='blue', marker='o', label='2022')
    plt.title('Consistency Prediction Results (After Adding New Samples)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()  # 添加图例，显示新的点的标签
    plt.savefig('consistency_prediction_result_2019and2020.png')
    plt.show()


         
    
if __name__ == "__main__":
    csv_file_path = 'data/2019data_5000.csv'
    # 2019年和2020年数据一起降维
    data_tsne = traindataset_tSNE(csv_file_path)
    # 降维完就把两年的数据划分开
    training_size = 1000
    data_tsne_global = data_tsne[:training_size]
    data_tsne_global2 = data_tsne[training_size:]
    df = pd.DataFrame(data_tsne_global2, columns=['Dimension 1', 'Dimension 2'])

    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('consistent_points_2.csv', index=False)

    print(f"Consistent points saved to consistent_points_2.csv")
    
    CP_cluster_2()