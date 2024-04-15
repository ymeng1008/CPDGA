# -*- coding: utf-8 -*-
"""
Created on 2022/1/3 15:45

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
import pandas as pd
import numpy as np
import math
import string
import collections
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import model_from_json
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM


class LSTMClassifier:
    def __init__(self):
        self.model = None
        self.valid_chars = {'q': 17, '0': 27, 'x': 24, 'd': 4, 'l': 12, 'm': 13, 'v': 22, 'n': 14, 'c': 3, 'g': 7, '7': 34, 'u': 21, '5': 32, 'p': 16, 'h': 8, 'b': 2, '6': 33, '-': 38, 'z': 26, '3': 30, 'f': 6, 't': 20, 'j': 10, '1': 28, '4': 31, 's': 19, 'o': 15, 'w': 23, '9': 36, 'r': 18, 'i': 9, 'e': 5, 'y': 25, 'a': 1, '.': 37, '2': 29, '_': 39, '8': 35, 'k': 11}
        self.maxlen = 178
        self.max_features = 40
        self.max_epoch = 20  # 20
        self.batch_size = 128
        self.tld_list = []
        self.isload_ = False
        with open(r'./data/tld.txt', 'r', encoding='utf8') as f:
            for i in f.readlines():
                self.tld_list.append(i.strip()[1:])

        score_df = pd.read_csv(r"./data/lstm_score_rank.csv", names=['score'])
        self.score_l = score_df['score'].tolist()

    def build_binary_model(self):
        """Build LSTM model for two-class classification"""
        self.model = Sequential()
        self.model.add(Embedding(self.max_features, 128, input_length=self.maxlen))
        self.model.add(LSTM(128))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))
        self.model.add(Activation('sigmoid'))#添加一个Sigmoid激活函数层，用于将模型的输出映射到0到1之间，表示二分类问题中的类别概率
        self.model.compile(loss='binary_crossentropy',optimizer='rmsprop')


    # 创建一个基于类别权重的字典，这些权重将用于训练机器学习模型中，特别是用于分类问题。
    def create_class_weight(self, labels_dict, mu):
        """Create weight based on the number of sld name in the dataset"""
        # labels_dict 包含了每个类别的样本数量
        labels_dict = dict(labels_dict)
        # 提取 labels_dict 中的所有键（类别标签），以便后续遍历。
        keys = labels_dict.keys()
        # 计算数据集中所有样本的总数
        # labels_dict[1] 表示类别标签为1的样本数量
        # labels_dict[0] 表示类别标签为0的样本数量
        total = labels_dict[1] + labels_dict[0]
        class_weight = dict()
        for key in keys:
            #目的是赋予样本数量较少的类别更大的权重
            score = math.pow(total/float(labels_dict[key]), mu)
            class_weight[key] = score
        return class_weight

    def train(self, model_folder, train_feature_add):
        """
        训练模型
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 批量测试文件路径
        :return:
        """
        # 模型存储的文件夹路径，这里包括模型文件和权重文件的保存路径
        model_add = "{}/LSTM_model.json".format(model_folder)  # 模型文件
        model_weight = "{}/LSTM_model.h5".format(model_folder)  # 权重文件
        # 获取训练和测试数据， domain，label
        # 从 train_feature_add 加载训练特征数据，只取前两列，也就是domain_name和label
        train_df = pd.read_csv(train_feature_add, header=[0])
        train_df = train_df.iloc[:, 0:2]
        #self.data_pro 函数用于对域名（URL）进行预处理，以将原始数据转换为模型可以接受的输入
        train_df["domain_name"] = train_df["domain_name"].apply(self.data_pro)
        # 将域名和标签转换为列表
        sld = train_df["domain_name"].to_list()
        label = train_df["label"].to_list()
        #这行代码使用列表推导式将每个域名字符串 x 中的字符转换为整数。具体地，它将每个字符 y 映射到 self.valid_chars 字典中的整数值，
        # 并将结果存储在列表 X 中。这一步是为了将字符级别的域名数据转换为模型可以处理的数值特征。
        X = [[self.valid_chars[y] for y in x] for x in sld]
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        #  将列表 label 转换为NumPy数组 y，以便于后续的模型训练。这是域名的标签或类别信息。
        y = np.array(label)
        # 用于分割数据集成训练集和测试集。
        # 每次迭代都重新划分
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
        #  使用StratifiedShuffleSplit对象的split方法分割数据集。
        for train, test in sss.split(X, y):
            # 分成训练集 (X_train, y_train) 和测试集 (X_test, y_test)
            X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
            print("---train:{}---test:{}----y_train:{}----y_test:{}".format(len(X_train), len(X_test), len(y_train),
                                                                            len(y_test)))
            # shuffle
            np.random.seed(4)  # 1024
            # 包含了训练集样本的索引
            index = np.arange(len(X_train))
            #使用随机种子设置的随机性，打乱训练集样本的顺序，以增加模型的训练多样性。
            np.random.shuffle(index)
            # 根据打乱后的索引数组重新排列训练集的特征和标签，以确保它们的顺序与索引数组一致。
            X_train = np.array(X_train)[index]
            y_train = np.array(y_train)[index]
            # build model
            self.build_binary_model()
            # train
            # 再次对训练集进行分层随机划分，将数据划分为训练集和 holdout 验证集。这里的目的是在每个 epoch 结束后评估模型的性能。
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
            for train, test in sss1.split(X_train, y_train):
                X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]  # holdout验证集
            #使用 collections.Counter 统计训练集中每个类别的样本数量
            labels_dict = collections.Counter(y_train)
            # 计算每个类别的样本权重。
            class_weight = self.create_class_weight(labels_dict, 0.3)
            print('----class weight:{}'.format(class_weight))
            # 20
            best_acc = 0.0 # 跟踪最佳验证集准确度
            best_model = None # 保存最佳模型
            for ep in range(self.max_epoch):
                self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=1, class_weight=class_weight)
                t_probs = self.model.predict_proba(X_holdout)
                t_result = [0 if x <= 0.5 else 1 for x in t_probs]
                t_acc = accuracy_score(y_holdout, t_result)
                print("epoch:{}--------val acc:{}---------best_acc:{}".format(ep, t_acc, best_acc))
                if t_acc > best_acc:
                    best_model = self.model
                    best_acc = t_acc
            model_json = best_model.to_json()
            # 模型的权重保存在HDF5中
            # 模型的结构保存在JSON文件或者YAML文件中
            with open(model_add, "w") as json_file:
                json_file.write(model_json)
                self.model.save_weights(model_weight)
            print("Saved two-class model to disk")

        # 计算训练集分数
        # self.load(model_folder)
        y_pred = self.model.predict_proba(X, batch_size=self.batch_size, verbose=1)
        y_pred = y_pred.flatten()
        df = pd.DataFrame(y_pred)
        df = df.sort_values(by=0, ascending=False)
        df.to_csv("./data/model/LSTM_train_scores.csv", index=False, header=None)

    def load(self, model_folder):
        """
        将模型文件和权重值读取
        :param model_folder: 模型存储文件夹
        :return:
        """
        # 构建了模型结构（.json 文件）和模型权重（.h5 文件）的文件路径。
        model_add = "{}/LSTM_model.json".format(model_folder)  # 模型文件
        model_weight_add = "{}/LSTM_model.h5".format(model_folder)  # 权重文件
        with open(model_add, 'r') as json_file:
            # 读取打开的模型文件的内容，将其存储在变量model中
            model = json_file.read()
        #将从文件中读取的模型结构字符串解析为Keras模型对象
        self.model = model_from_json(model)
        # 权重参数
        self.model.load_weights(model_weight_add)
        # 读取之前训练过程中保存的训练分数
        score_df = pd.read_csv("{}/LSTM_train_scores.csv".format(model_folder), names=['score'])
        self.score_l = score_df['score'].tolist()
        self.isload_ = True
        # 存储分数不知道干什么用？
        # 添加分数计算模块，如果路径存在跳过，如果路径不存在，重新计算训练数据分数


# 将原始的URL字符串进行清洗和转换，以使其适用于机器学习模型的输入。
    def data_pro(self, url):
        """
        预处理字符串
        :param url:
        :return:
        """
        # 去除URL字符串两端的空格、点号和斜杠。
        url = url.strip().strip('.').strip('/')
        # 如果URL以 "http://" 开头，将其删除，只保留域名部分。这是为了去除常见的URL前缀。
        url = url.replace("http://", '')
        #  使用斜杠分割URL，只保留第一部分，即域名部分。
        url = url.split('/')[0]
        #  使用问号分割URL，只保留第一部分，即查询参数之前的部分。这是为了去除URL中的查询参数。
        url = url.split('?')[0]
        # 使用等号分割URL，只保留第一部分，即等号之前的部分。这是为了去除URL中的等号及其后面的内容。
        url = url.split('=')[0]
        # 使用点号分割URL，将其拆分为一个列表，其中每个元素代表URL中的一个部分。
        dn_list = url.split('.')
        # 对拆分后的列表进行逆序遍历
        #  如果是顶级域名或者"www"，则从列表中删除该元素。
        #  将处理后的域名部分重新连接成一个字符串，形成"short_url"。
        # 删除字符串中的方括号。
        # 将字符串转换为小写字母，以保持一致性。
        for i in reversed(dn_list):
            if i in self.tld_list:
                dn_list.remove(i)
            elif i == 'www':
                dn_list.remove(i)
            else:
                continue
        short_url = ''.join(dn_list)
        short_url = short_url.replace('[', '').replace(']', '')
        short_url = short_url.lower()
        return short_url


# 还不太懂为什么是这个计算方式
    def cal_p(self, s):
        """
        计算p_value, 二分查找
        :param s: float
        :return:
        """
        # p-value 是统计学中用于衡量观测数据与某个模型或假设之间的一致性的指标
        flag = 0  # score偏da的对应的
        # score_l 训练集分数
        for i in range(len(self.score_l)):
            # 找到第一个小于等于 0.5 的概率值后，flag 的值被更新为该索引减去 1
            # 确定模型输出的概率值是倾向于 0 还是 1
            if self.score_l[i] <= 0.5000000000000000:
                flag = i - 1
                break
        # print("flag:{}".format(flag))
        if s >= self.score_l[0]:# 最小的概率值
            return 1.0
        if s <= self.score_l[-1]:# 最大的概率值
            return 1.0
        if s == self.score_l[flag]: # 0.5 的概率值
            # return 1 / ((flag + 1) * 1.0)
            return 0.0

        high_index = len(self.score_l)
        low_index = 0
        while low_index < high_index:
            mid = int((low_index + high_index) / 2)
            if s > self.score_l[mid]:
                high_index = mid - 1
            elif s == self.score_l[mid]:
                if s > 0.5:
                    return (flag - mid + 1) / ((flag + 1) * 1.0)
                else:
                    return (mid - flag) / ((len(self.score_l) - flag - 1) * 1.0)
            else:
                low_index = mid + 1
        if s > 0.5:

            return round((flag - low_index) / ((flag + 1) * 1.0), 4)
        else:
            return round((low_index - flag) / ((len(self.score_l) - flag - 1) * 1.0), 4)

    def predict(self, model_folder, test_feature_add):
        """
        批量检测
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 批量测试文件路径
        :return:
        """
        # 先加载模型
        if not self.isload_:
            self.load(model_folder)
            self.isload_ = True
        # 获取测试数据， domain，label
        # 和训练时的流程一样
        df = pd.read_csv(test_feature_add, header=[0])
        df = df.iloc[:, 0:2]
        df["domain_name"] = df["domain_name"].apply(self.data_pro)
        sld = df["domain_name"].to_list()
        label = df["label"].to_list()
        X = [[self.valid_chars[y] for y in x] for x in sld]
        X = sequence.pad_sequences(X, maxlen=self.maxlen)
        y = np.array(label)
        # 预测
        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        # 转化为标签
        y_result = [0 if x <= 0.5 else 1 for x in y_pred]
        # 计算模型准确率召回率
        score = f1_score(y, y_result)
        precision = precision_score(y, y_result)
        recall = recall_score(y, y_result)
        acc = accuracy_score(y, y_result)
        print('LSTM accuracy:', acc)
        print('LSTM precision:', precision)
        print('LSTM recall:', recall)
        print('LSTM F1:', score)

    def predict_single_dname(self, model_folder, dname):
        """
        对单个域名进行检测，输出检测结果及恶意概率
        :param model_folder: 模型存储文件夹
        :param dname: 域名
        :return:
        """
        # 加载模型
        if not self.isload_:
            self.load(model_folder)
            self.isload_ = True
        # 对输入的域名进行预处理
        dname = dname.strip(string.punctuation)
        short_url = self.data_pro(dname)
        #训练时相同的方式进行字符级别的编码，并进行长度填充，确保输入与模型的期望长度一致
        sld_int = [[self.valid_chars[y] for y in x] for x in [short_url]]
        sld_int = sequence.pad_sequences(sld_int, maxlen=self.maxlen)
        sld_np = np.array(sld_int)
        # 编译模型
        #对域名数据进行预测
        self.model.compile(loss='binary_crossentropy', optimizer='rmsprop')
        # 空字符串的域名，模型预测为非恶意，概率为100%
        if short_url == '':
            score = 0.0
            p_value = 1.0
            label = 0
            print("\nLSTM dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, score, p_value))
            return label, score, p_value
        else:
            #对域名数据进行预测
            # 使用模型预测得到scores
            scores = self.model.predict(sld_np)
            score = scores[0][0] # 恶意概率的值，在0和1之间
            p_value = self.cal_p(score)

            if score > 0.5:
                label = 1
            else:
                label = 0
            print("\nLSTM dname:", dname)
            print('label:{}, pro:{}, p_value:{}'.format(label, score, p_value))
            return label, score, p_value
if __name__ == "__main__":
    # muldec = MultiModelDetection()

    # from feeds.danalysis import LDAClassifier
    clf = LSTMClassifier()
    clf.train(r"./data/model", r"./Year_Test/2019_2020_train.csv")
    clf.load(r"./data/model")
    clf.predict(r"./data/model", r"./Year_Test/2019_2020_test.csv")
    