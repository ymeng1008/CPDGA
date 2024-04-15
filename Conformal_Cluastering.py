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
plt.switch_backend('agg')

# 全局变量
data_tsne_global = None
data_tsne = None
data_tsne_global2 = None
data_tsne_Unknown = None



"""
t-SNE进行特征降维
"""
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
                label=label)

    plt.title('t-SNE Visualization')
    # 二维投影
    # 生成的二维散点图反映了原始高维数据点之间的相似性和距离关系
    # 保留数据点之间的相似性关系
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('TSNE_2family.png')
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
    plt.savefig('TSNE_DBSCAN_2family.png')
    plt.show()
    # 返回降维后的结果
    return data_tsne

# 非一致性度量的两个函数
# 基于k-最近邻（k-NN）的非一致性度量
# 这里k取得是1，应该多调几个参数试试
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
"""
p_value的计算
"""
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
    print(len(D)) 

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

def calculate_only_p_value(D, A, z):
    """
    输入：
    一组对象的集合D，z1到zn-1
    非一致度量函数A(KNN和KDE)
    两种方法后续再补充进来
    新对象z
    输出：
    p_value
    
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
    
    
    
    return pn

"""
测试对象分类并计算confidence和credibility

"""
def CP_cluster_Unknown():
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
    grid_cell_distance_threshold =  (max_val - min_val) / num_points_per_dimension
    # 生成格点网络
    grid_points = np.array(np.meshgrid(*all_grid_points)).T.reshape(-1, num_features)
        
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

    # 提取 p_value 值，较大的 p_value 对应较深的颜色
   
    p_values_for_color = [p_value for p_value, is_consistent in p_values if is_consistent]
    
    # 创建一个散点图来可视化一致性预测结果，根据 p_value 的大小设置颜色深浅

    # 调整颜色映射，使用更细致的红色渐变
    cmap = plt.cm.get_cmap('Reds', len(p_values_for_color))

    # 创建一个散点图来可视化一致性预测结果
    plt.figure(figsize=(10, 8))

    # 绘制第一张图（未添加新样本点）
    
    plt.scatter(x_coords, y_coords, s=50, c=p_values_for_color, cmap=cmap, edgecolors='k', marker='x', linewidth=0.5)

    # 设置图表标题和坐标轴标签
    # 设置指定的刻度和范围
    x_ticks = [-24,-20, -15,-10, -5,0, 5,10, 15,19]
    y_ticks = [-42,-40,-30,-20,-10, 0,10,20,26]
    # 要隐藏的刻度值
    hidden_x_ticks = [-24,19 ]
    hidden_y_ticks = [-42,26 ]
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.ylim(y_ticks[0], y_ticks[-1])
    # 隐藏特定刻度值
    plt.xticks([tick for tick in x_ticks if tick not in hidden_x_ticks])
    plt.yticks([tick for tick in y_ticks if tick not in hidden_y_ticks])
    
    plt.title('Consistency Prediction Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')

    # 显示网格线
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加颜色条
    cbar = plt.colorbar(ticks=np.linspace(0.2, 0.8, 10), label='Color Intensity (p_value)')


    # 显示整体图
    plt.savefig('Conformal_Prediction.png')
    plt.show()


         
    #步骤4：聚类，这里先暂时选用DBSCAN
    # consistent_points 是训练集预测出来的点
    # 将 consistent_points 转换为 DataFrame
    df = pd.DataFrame(consistent_points, columns=['Dimension 1', 'Dimension 2'])

    # 将 DataFrame 存储为 CSV 文件
    df.to_csv('consistent_points.csv', index=False)

    print(f"Consistent points saved to consistent_points.csv")
    
    X = np.array(consistent_points)
    distance_matrix = pairwise_distances(X, metric='euclidean')



    """
    网上学的一个确定参数的方法
    """
    # 选择 k 的值
    k = 2 * X.shape[1] - 1

    # 计算每个点的第 k 近邻距离
    k_distances = np.sort(distance_matrix, axis=1)[:, k]

    # 绘制 K-distance 图
    plt.plot(np.arange(k_distances.shape[0]), k_distances[::-1])
    plt.title('K-distance Graph')
    plt.xlabel('Data Point Index')
    plt.ylabel(f'{k}-distance')
    plt.savefig('K_distance_Graph.png')
    plt.show()


    eps = 5  # You can adjust this value based on the K-distance analysis
    min_samples = 10
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = db.fit_predict(distance_matrix)
    # 所有不同聚类标签的集合
    unique_labels = set(labels)

    # 创建一个颜色映射，为每个簇分配不同的颜色
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    plt.figure(figsize=(10, 8))

    for label, color in zip(unique_labels, colors):
        # labels存储了每个数据点所属的聚类标签
        # 指示哪些点属于当前聚类
        cluster_mask = labels == label  # 根据聚类标签创建一个布尔掩码
        xy = X[cluster_mask]  # 选择属于当前聚类的点
        plt.scatter(xy[:, 0], xy[:, 1], s=50, c=color, label=f'Cluster {label}', edgecolors='k')

    plt.title('DBSCAN Clustering Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('CP_cluster.png')
    plt.show()

    # 步骤五，测试点根据邻近规则聚类
    # 测试点的p值计算按理来说与预测方式（比如KNN...)得出的概率分数有关，但是这里我们采取的邻近规则聚类方法没有这个概率分数，所以还是用了之前的p值计算方法
    # 论文中：如果一个测试对象与某个聚类中的某一个点的距离小于网格单元的举例，那么它就被认为属于该聚类
    # 如果一个点与多个聚类相关联，那么这些聚类将被合并
    # 不知道还用不用np.array
    # 初始化一个字典来追踪每个测试对象所属的所有聚类
    
    # 因为通过confidence和credibility可以识别这些与多个聚类相关联的特殊点
    test_objects = np.array(data_tsne_Unknown) 
    # 跟踪每个测试对象所属的聚类
    test_objects_clusters = {i: [] for i in range(len(test_objects))}
    

    # 按照论文里说的这种聚类方法，会有点并不属于任何一个聚类，那这种情况下它的预测标签是什么呢，是一个新的吗
    # 遍历每个聚类
    for label, color in zip(unique_labels, colors):
        # labels存储了每个数据点所属的聚类标签
        # 指示哪些点属于当前聚类
        cluster_mask = labels == label
        # 获取当前聚类中的点存到cluster_points中
        cluster_points = X[cluster_mask]

        # 计算每个测试对象与当前聚类中任一点的距离
        distances_to_cluster = pairwise_distances(test_objects, cluster_points, metric='euclidean')
        min_distances = np.min(distances_to_cluster, axis=1)

        # 判断是否属于当前聚类
        belong_to_cluster = min_distances <= grid_cell_distance_threshold
        belong_to_cluster_indices = np.where(belong_to_cluster)[0]

        
        # 更新测试对象的聚类信息
        # 将当前聚类标签添加到属于该聚类的测试对象的聚类信息中
        for idx in belong_to_cluster_indices:
            test_objects_clusters[idx].append(label)

    

    # 合并有重叠测试对象的聚类
    for test_object, cluster_labels in test_objects_clusters.items():
        if len(cluster_labels) > 1:
            # 找到多个聚类
            combined_label = min(cluster_labels)  # 使用最小的聚类标签作为合并后的标签
            for label in cluster_labels:
                labels[labels == label] = combined_label  # 更新所有这些聚类的标签
                # 修改此处，更新test_objects_clusters[idx]中的标签值
            test_objects_clusters[test_object].insert(0, combined_label)
            # 更新所有聚类中的测试对象的聚类信息
            # 这部分是我新加的，不知道对不对
            for idx in belong_to_cluster_indices:
                if any(l in cluster_labels for l in test_objects_clusters[idx]):
                    test_objects_clusters[idx].insert(0, combined_label)
    # 更新unique_labels以反映合并后的聚类
    # labels 是存储了每个数据点所属聚类的数组。
    # np.unique(labels) 返回数组中的唯一值，并按升序排列。
    # 结果存储在 unique_labels 变量中，该变量是一个包含数据集中所有聚类标签的唯一值的数组。
    unique_labels = np.unique(labels)
    
    # 补充：计算每个测试对象的信息
    
    # 应该是需要改一下test_objects_clusters[idx]，这个后续再说
    test_objects_info = []
    # 遍历测试对象
    for idx, test_object in enumerate(test_objects):
        # 获取当前测试对象的聚类标签
        cluster_labels = test_objects_clusters[idx]
        if len(cluster_labels) == 0:
            # 如果不属于任何类，取所有聚类中最大p值为credibility
            # 这句代码我不太懂，可能有问题
            # unique_labels 好像是后面的
            # X中就是预测点按标签的集合
            p_values_for_clusters = [calculate_only_p_value(X[labels == i], 0, test_object) for i in unique_labels]
            max_p_value = max(p_values_for_clusters)
            credibility = max_p_value
            # confidence = 1 - 与所有聚类中第二大的p值
            sorted_p_values = sorted(p_values_for_clusters, reverse=True)
            confidence = 1 - sorted_p_values[1]
            # 聚类标签为 "nolabel"
            cluster_label = "nolabel"
            # 将 "P_Values" 设置为 {f"Cluster{i}": p_values_for_clusters[i]}
            p_values_for_clusters_dict = {f"Cluster{label}": p_values_for_clusters[i] for i, label in enumerate(unique_labels)}
        else:
            # 如果属于某个聚类，取与所属类别的p值为credibility
            cluster_label = cluster_labels[0]
            print(f"cluster_labels = {cluster_labels}")
            p_value_with_cluster = calculate_only_p_value(X[labels == cluster_label], 0, test_object)
            credibility = p_value_with_cluster
            # confidence = 1 - 与所有聚类中第二大的p值
            other_cluster_labels = set(unique_labels) - set(cluster_labels)
            other_p_values = [calculate_only_p_value(X[labels == label], 0, test_object) for label in other_cluster_labels]
            confidence = 1 - max(other_p_values)
            # 将 "P_Values" 设置为 {f"Cluster{cluster_label}": p_value_with_cluster}
            p_values_for_clusters_dict = {f"Cluster{cluster_label}": p_value_with_cluster}
            # 将 "P_Values_For_Clusters" 设置为 {f"Cluster{i}": p_value for i, p_value in zip(other_cluster_labels, other_p_values)}
            p_values_for_clusters_dict.update({f"Cluster{i}": p_value for i, p_value in zip(other_cluster_labels, other_p_values)})

        # 记录测试对象的信息
        test_object_info = {
            "Point": tuple(test_object),
            "Cluster": cluster_label,
            "P_Values": p_values_for_clusters_dict,
            "Credibility": credibility,
            "Confidence": confidence,
            
        }
        test_objects_info.append(test_object_info)

    # 把测试对象相关信息存到.csv文件中    
    csv_file_path = 'test_objects_info.csv'

    # 对应列名
    field_names = ['Point', 'Cluster', 'P_Values', 'Credibility', 'Confidence', 'P_Values_For_Clusters']

    # Write the test_objects_info list to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(test_objects_info)

    print(f"Test object information successfully written to {csv_file_path}")

    # 绘制聚类结果
    plt.figure(figsize=(10, 8))
    for label, color in zip(unique_labels, colors):
        cluster_mask = labels == label
        xy = X[cluster_mask]
        plt.scatter(xy[:, 0], xy[:, 1], s=50, label=f'Cluster {label}',  color=color,edgecolors='k')

        # 绘制分配到该聚类的测试对象
        test_cluster_mask = np.array([label in clusters for clusters in test_objects_clusters.values()])
        
        # 选择属于当前聚类的测试对象
        test_cluster_points = test_objects[test_cluster_mask]
        
        # 先绘制彩色圆点
        plt.scatter(test_cluster_points[:, 0], test_cluster_points[:, 1], s=50, color=color, marker='o')
        # 再绘制黑色的“x”标记
        plt.scatter(test_cluster_points[:, 0], test_cluster_points[:, 1], s=50, c='k', marker='x', edgecolors='k')

        # 在这里计算有聚类的点的confidence和credibility
        # 输出格式：(X,Y)----confidence----credibility----label
        
    # 绘制未分配到任何聚类的测试对象，这些测试对象怎么办呢
    not_assigned_mask = np.array([len(clusters) == 0 for clusters in test_objects_clusters.values()])
    not_assigned_points = test_objects[not_assigned_mask]
    plt.scatter(not_assigned_points[:, 0], not_assigned_points[:, 1], s=50, c='gray', marker='x', label='Unassigned Test Object')
    
    plt.title('Clustering Results with Merged Clusters')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('CP_testpoint_cluster.png')
    plt.show()

    # 在这里计算有聚类的点的confidence和credibility
    # 输出格式：(X,Y)----confidence----credibility----nolabel

    # 需要计算每个测试点的credibility和confidence
    # 对于没被分配到任何聚类的测试对象，取对某个聚类最大p值为credibility





if __name__ == "__main__":
    csv_file_path = './Family_test/CKM.csv'
    data_tsne = traindataset_tSNE(csv_file_path)
    #csv_file_path2 = './CC/BCZ.csv'
    #data_tsne_Unknown = traindataset_tSNE(csv_file_path2)
    # 切分数据集
    training_size = 700
    data_tsne_global = data_tsne[:training_size]
    data_tsne_Unknown = data_tsne[training_size:]

    # 确认切分是否正确
    print(len(data_tsne_global))  # 应该输出 700
    print(len(data_tsne_Unknown))   # 应该输出原始数据长度减去训练集长度

    """
    这里对测试集和训练集降维结果聚类看一下
    """
    plt.scatter(data_tsne_global[:, 0], data_tsne_global[:, 1], label='Training Set', marker='o')
    plt.scatter(data_tsne_Unknown[:, 0], data_tsne_Unknown[:, 1], label='Test Set', marker='x')

    plt.title('Training and Test Data Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.savefig('training_test_visualization.png')
    plt.show()

    # DBSCAN降维

    # 初始化DBSCAN模型
    plt.figure()  # 创建新图形
    dbscan = DBSCAN(eps=5, min_samples=10)
    # 对降维后的数据进行DBSCAN聚类
    labels_p = dbscan.fit_predict(data_tsne_global)
    # 将聚类结果添加到原始数据中

    
    # 绘制散点图，每个聚类用不同颜色表示
    plt.scatter(data_tsne_global[:, 0], data_tsne_global[:, 1], c=labels_p)
    # 设置X轴和Y轴的范围
    # 设置指定的刻度和范围
    
    x_ticks = [-24,-20, -15,-10, -5,0, 5,10, 15,19]
    y_ticks = [-42,-40,-30,-20,-10, 0,10,20,26]
    # 要隐藏的刻度值
    hidden_x_ticks = [-24,19 ]
    hidden_y_ticks = [-42,26 ]
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.ylim(y_ticks[0], y_ticks[-1])
    # 隐藏特定刻度值
    plt.xticks([tick for tick in x_ticks if tick not in hidden_x_ticks])
    plt.yticks([tick for tick in y_ticks if tick not in hidden_y_ticks])
    
    plt.title("t-SNE with DBSCAN Clustering")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig('TSNE_DBSCAN_TRAIN.png')
    plt.show()
    



    # 初始化DBSCAN模型
    plt.figure()  # 创建新图形
    dbscan = DBSCAN(eps=5 , min_samples=5)
    # 对降维后的数据进行DBSCAN聚类
    labels_p = dbscan.fit_predict(data_tsne_Unknown)
    # 将聚类结果添加到原始数据中

    
    # 绘制散点图，每个聚类用不同颜色表示
    plt.scatter(data_tsne_Unknown[:, 0], data_tsne_Unknown[:, 1], c=labels_p)
    # 设置X轴和Y轴的范围
    # 设置指定的刻度和范围
    
    x_ticks = [-24,-20, -15,-10, -5,0, 5,10, 15,19]
    y_ticks = [-42,-40,-30,-20,-10, 0,10,20,26]
    # 要隐藏的刻度值
    hidden_x_ticks = [-24,19 ]
    hidden_y_ticks = [-42,26 ]
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    plt.xlim(x_ticks[0], x_ticks[-1])
    plt.ylim(y_ticks[0], y_ticks[-1])
    # 隐藏特定刻度值
    plt.xticks([tick for tick in x_ticks if tick not in hidden_x_ticks])
    plt.yticks([tick for tick in y_ticks if tick not in hidden_y_ticks])
    
    plt.title("t-SNE with DBSCAN Clustering")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig('TSNE_DBSCAN_TEST.png')
    plt.show()



    CP_cluster_Unknown()
