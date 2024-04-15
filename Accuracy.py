import pandas as pd

# 读取CSV文件
file_path = 'test_objects_info.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path, delimiter=',')  # 假设CSV文件使用制表符作为分隔符
print(df.columns)
# 计算在Cluster列中值为'nolabel'的比例
nolabel_percentage = (df['Cluster'] == 'nolabel').mean() * 100

print(f"The percentage of 'nolabel' in the Cluster column is: {nolabel_percentage:.2f}%")

"""
确定阈值的划分
"""

file_path = 'test_objects_info.csv'  # 替换为你的文件路径
df = pd.read_csv(file_path, delimiter=',')  # 假设CSV文件使用制表符作为分隔符
# 根据条件筛选数据
filtered_data = df[df['Cluster'] == 'nolabel']

# 计算Credibility和Confidence的平均值、最大值和最小值
credibility_mean = filtered_data['Credibility'].mean()
credibility_max = filtered_data['Credibility'].max()
credibility_min = filtered_data['Credibility'].min()

confidence_mean = filtered_data['Confidence'].mean()
confidence_max = filtered_data['Confidence'].max()
confidence_min = filtered_data['Confidence'].min()

# 打印结果
print("Credibility Mean:", credibility_mean)
print("Credibility Max:", credibility_max)
print("Credibility Min:", credibility_min)

print("Confidence Mean:", confidence_mean)
print("Confidence Max:", confidence_max)
print("Confidence Min:", confidence_min)