"""
域名特征重要性分析


"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy

import os


"""
XGBoost分析
"""
"""# 读取数据集
data = pd.read_csv("./data/features/non1.csv")


# 分割数据集为特征和标签
X = data.drop(['domain_name', 'label'], axis=1)
y = data['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建XGBoost分类器
model = xgb.XGBClassifier(objective="binary:logistic", random_state=42)

# 训练模型
model.fit(X_train, y_train)

# 预测测试集
y_pred = model.predict(X_test)

# 计算准确性
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 获取特征重要性
feature_importance = model.feature_importances_

# 将特征名称与重要性值一一对应
feature_names = X.columns
feature_importance_dict = dict(zip(feature_names, feature_importance))

# 对特征重要性进行排序
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

# 打印特征重要性
print("Feature Importance:")
for feature, importance in sorted_feature_importance:
    print(f"{feature}: {importance}")

# 可视化特征重要性
plt.figure(figsize=(12, 6))
plt.bar(range(len(feature_importance)), feature_importance)
plt.xticks(range(len(feature_importance)), feature_names, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance")
plt.title("XGBoost Feature Importance")
plt.show()
"""

"""
随机森林
"""
""""
# 读取你的数据集
df = pd.read_csv("./data/features/non1.csv")

# 假设你的标签是 'label' 列
labels = df['label']

# 删除标签列以及其他不需要的列
features = df.drop(['label', 'domain_name'], axis=1)

# 初始化随机森林分类器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf_classifier.fit(features, labels)

# 获取特征重要性
feature_importances = rf_classifier.feature_importances_

# 将特征重要性与特征名字对应起来
feature_importance_df = pd.DataFrame({'Feature': features.columns, 'Importance': feature_importances})

# 排序特征重要性
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 打印排好序的特征重要性
print(feature_importance_df)

# 绘制特征重要性条形图
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.title('Random Forest Feature Importance')
plt.show()

"""
"""
遍历文件夹 year_Data 下的每个年份的 CSV 文件。
对每个年份的数据集应用随机森林算法，计算特征重要性。
将每个年份的特征重要性可视化到一张大图上，每个小图代表一个年份。
将所有年份的34个特征的平均重要性可视化到第二张图上。
"""

def visualize_feature_importance(year_data_folder):
    # 读取所有年份的数据集
    all_years_data = {}
    for year_file in os.listdir(year_data_folder):
        year, _ = os.path.splitext(year_file)
        data_path = os.path.join(year_data_folder, year_file)
        all_years_data[year] = pd.read_csv(data_path)

    # 创建大图
    fig, axes = plt.subplots(len(all_years_data), 1, figsize=(10, 5 * len(all_years_data)))

    # 创建空字典存储所有年份的特征重要性
    all_feature_importances = {}

    # 循环处理每个年份的数据
    for i, (year, data) in enumerate(all_years_data.items()):
        # 假设你的标签是 'label' 列
        labels = data['label']

        # 删除标签列以及其他不需要的列
        features = data.drop(['label', 'domain_name'], axis=1)

        # 初始化随机森林分类器
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # 训练模型
        clf.fit(features, labels)

        # 提取特征重要性
        feature_importances = clf.feature_importances_

        # 存储特征重要性到字典
        all_feature_importances[year] = feature_importances

        # 绘制小图
        axes[i].barh(features.columns, feature_importances)
        axes[i].set_title(f'Feature Importance - {year}')
        axes[i].set_xlabel('Importance')
        axes[i].set_ylabel('Feature')

    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig('feature_importance_everyyear.png')
    # plt.show()
    # 创建大图
    # fig, axes = plt.subplots(nrows=len(features.columns), ncols=1, figsize=(15, 3 * len(features.columns)))

    # 转换特征重要性为 DataFrame
    feature_importances_df = pd.DataFrame(all_feature_importances, index=features.columns)
    # 对DataFrame按照年份进行排序
    feature_importances_df = feature_importances_df.sort_index(axis=1)
    # 循环遍历每个特征
    for i, feature in enumerate(features.columns):
        # 获取特征在不同年份上的重要性
        feature_importance_values = feature_importances_df.loc[feature]
        # 关闭之前的图形
        plt.close()
        # 绘制小图
        
        plt.figure(figsize=(8, 5))  # 设置小图的大小
        plt.plot(feature_importance_values, marker='o')
        plt.title(f'Feature Importance Over Years - {feature}')
        plt.xlabel('Year')
        plt.ylabel('Importance')
        # 保存每张小图为不同的文件，文件名包含特征的名称
        plt.savefig(f'feature_importance_{feature}.png')
    
        # 清除当前图，以便下一次循环时创建新的图
        plt.clf()
        """# 绘制小图
        feature_importance_values.plot(kind='line', ax=axes[i])
        axes[i].set_title(f'Feature Importance Over Years - {feature}')
        axes[i].set_xlabel('Year')
        axes[i].set_ylabel('Importance')

        # 调整布局
        plt.tight_layout()
        # 保存每张小图为不同的文件，文件名包含特征的名称
        plt.savefig(f'feature_importance_{feature}.png')
    
        # 清除当前图，以便下一次循环时创建新的图
        plt.clf()"""
    # 显示图
    # plt.show()


def visualize_feature_importance_boruta(year_data_folder):
    # 读取所有年份的数据集
    all_years_data = {}
    for year_file in os.listdir(year_data_folder):
        year, _ = os.path.splitext(year_file)
        data_path = os.path.join(year_data_folder, year_file)
        all_years_data[year] = pd.read_csv(data_path)

    # 创建大图
    fig, axes = plt.subplots(len(all_years_data), 1, figsize=(10, 5 * len(all_years_data)))

    # 创建空字典存储所有年份的特征重要性
    all_feature_importances = {}

    # 循环处理每个年份的数据
    for i, (year, data) in enumerate(all_years_data.items()):
        # 标签是 'label' 列
        # 提取特征和标签
        X = data.drop(['domain_name', 'label'], axis=1)
        y = data['label']
        # 构建随机森林模型
        rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', max_depth=5, random_state=42)

        # 初始化Boruta
        boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)

        # 执行特征选择
        boruta_selector.fit(X.values, y.values)

        # 获取选中的特征
        selected_features = X.columns[boruta_selector.support_].to_list()

        # 保存选中的特征
        print(f"Selected features for {year}: {selected_features}")
        print(dir(boruta_selector))
        # 获取特征重要性
        # python 版本的boruta不会计算特征重要性。。。
        importances = boruta_selector.ranking_
        print(importances)
        
        feature_importances = pd.Series(importances, index=X.columns)
        
        # 存储特征重要性到字典
        all_feature_importances[year] = feature_importances

        # 绘制小图
        axes[i].barh(X.columns, feature_importances)
        axes[i].set_title(f'Feature Importance - {year}')
        axes[i].set_xlabel('Importance')
        axes[i].set_ylabel('Feature')
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig('feature_importance_everyyear.png')
    plt.show()

    # 创建大图
    # fig, axes = plt.subplots(nrows=len(X.columns), ncols=1, figsize=(15, 3 * len(X.columns)))

    # 转换特征重要性为 DataFrame
    feature_importances_df = pd.DataFrame(all_feature_importances, index=X.columns)
    # 循环遍历每个特征
    for i, feature in enumerate(X.columns):
        # 获取特征在不同年份上的重要性
        feature_importance_values = feature_importances_df.loc[feature]

        # 绘制小图
        
        plt.figure(figsize=(8, 5))  # 设置小图的大小
        plt.plot(feature_importance_values, marker='o')
        plt.title(f'Feature Importance Over Years - {feature}')
        plt.xlabel('Year')
        plt.ylabel('Importance')
        # 保存每张小图为不同的文件，文件名包含特征的名称
        plt.savefig(f'feature_importance_{feature}.png')
    
        # 清除当前图，以便下一次循环时创建新的图
        plt.clf()

    

if __name__ == "__main__":

    
    visualize_feature_importance('Year_Data')
    # visualize_feature_importance_boruta('Year_Data')
    # 怎么融合各个年份的重要性值以及变化趋势来为特征分配一个权重

