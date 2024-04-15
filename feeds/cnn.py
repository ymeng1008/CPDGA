import pandas as pd
import numpy as np
import math
import string
import collections
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import model_from_json,Model
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D,Concatenate,Lambda,Flatten,Input
from keras import backend as K



class CNNClassifier:
    def __init__(self):
        self.model = None # 用于存储CNN模型
        self.valid_chars = {'q': 17, '0': 27, 'x': 24, 'd': 4, 'l': 12, 'm': 13, 'v': 22, 'n': 14, 'c': 3, 'g': 7, '7': 34, 'u': 21, '5': 32, 'p': 16, 'h': 8, 'b': 2, '6': 33, '-': 38, 'z': 26, '3': 30, 'f': 6, 't': 20, 'j': 10, '1': 28, '4': 31, 's': 19, 'o': 15, 'w': 23, '9': 36, 'r': 18, 'i': 9, 'e': 5, 'y': 25, 'a': 1, '.': 37, '2': 29, '_': 39, '8': 35, 'k': 11} # 不知道干什么用的，先保留
        # 按照毕业论文里提供的参数
        self.maxlen = 60   # 最大长度
        self.max_features = 40  #最大特征数
        self.max_epoch = 10  # 20 迭代次数
        self.batch_size = 128 #批次大小
        self.num_classes = 2  # 分类类别数量 二分类
        # 从一个文件 tld.txt 中读取数据并将其存储在 self.tld_list 中
        self.tld_list = [] 
        self.isload_ = False
        with open(r'./data/tld.txt', 'r', encoding='utf8') as f:
            for i in f.readlines():
                self.tld_list.append(i.strip()[1:])

    """def build_cnn_model(self):
      
        # 创建Sequential模型
        self.model = Sequential()
        # 添加嵌入层（Embedding Layer）
        self.model.add(Embedding(input_dim=self.max_features, output_dim=128, input_length=self.maxlen))
        # 定义一系列卷积核大小和滤波器数量
        kernel_sizes = [2, 3, 4, 5]
        num_filters = 256
        # 循环添加卷积层(一共有四个卷积层)
        for kernel_size in kernel_sizes:
            conv_layer = self.get_convolutional_layer(kernel_size, num_filters)
            self.model.add(conv_layer)
        # 合并不同卷积核的结果
        merged = Concatenate()(self.model.layers[-len(kernel_sizes):].output)
        merged = Flatten()(merged)  # 将三维张量展平为二维张量
        # 三个全连接层
        # 添加中间全连接层1
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))

        # 添加中间全连接层2
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dropout(0.5))

        # 添加输出层，输出类别概率
        self.model.add(Dense(1, activation='sigmoid'))
        # 编译模型
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
        self.model.summary()
    """
    def build_cnn_model(self):
    
        # 创建主输入层
        main_input = Input(shape=(self.maxlen,), dtype='int32', name='main_input')
        # 添加嵌入层（Embedding Layer）
        embedding = Embedding(input_dim=self.max_features, output_dim=128, input_length=self.maxlen)(main_input)
    
        # 定义一系列卷积核大小和滤波器数量
        kernel_sizes = [2, 3, 4, 5]
        num_filters = 256
    
        conv_layers = []
    
        # 循环添加卷积层(一共有四个卷积层)
        for kernel_size in kernel_sizes:
            conv_layer = self.get_convolutional_layer(kernel_size, num_filters)(embedding)
            conv_layers.append(conv_layer)
    
        # 合并不同卷积核的结果
        merged = Concatenate()(conv_layers)
    
        # 三个全连接层
        # 添加中间全连接层1
        middle = Dense(1024, activation='relu')(merged)
        middle = Dropout(0.5)(middle)
    
        # 添加中间全连接层2
        middle = Dense(512, activation='relu')(middle)
        middle = Dropout(0.5)(middle)
    
        # 添加输出层，输出类别概率
        output = Dense(1, activation='sigmoid')(middle)
    
        # 创建模型
        self.model = Model(inputs=main_input, outputs=output)
    
        # 编译模型
        self.model.compile(loss='binary_crossentropy', optimizer='adam')
    
        # 打印模型摘要
        self.model.summary()

    def get_convolutional_layer(self, kernel_size, filters):
        model = Sequential()
        model.add(Conv1D(filters=filters, input_shape=(self.maxlen, 128), kernel_size=kernel_size, padding='same', activation='relu', strides=1))
        model.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(filters, )))
        model.add(Dropout(0.5))
        return model

    

    # 不知道需不需要添加类别权重函数

    # 清洗和简化后的域名字符串，转换为可以处理的特征形式
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
    
    def train(self, model_folder,train_feature_add):
        """
        训练模型
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 批量测试文件路径
        :return:
        """
         # 模型存储的文件夹路径，这里包括模型文件和权重文件的保存路径
        model_add = "{}/CNN_model.json".format(model_folder)  # 模型文件
        model_weight = "{}/CNN_model.h5".format(model_folder)  # 权重文件
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
        # 数据划分,分层随机划分数据集的方法，它确保了训练集和测试集中各个类别的样本比例与原始数据集中的比例相同。
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=4)
        for train, test in sss.split(X, y):
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
            # 模型构建
            self.build_cnn_model()
            # 训练模型
            # 再次对训练集进行分层随机划分，将数据划分为训练集和 holdout 验证集
            sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.05, random_state=0)
            for train, test in sss1.split(X_train, y_train):
                X_train, X_holdout, y_train, y_holdout = X_train[train], X_train[test], y_train[train], y_train[test]
            
            # 使用 collections.Counter 统计训练集中每个类别的样本数量
            labels_dict = collections.Counter(y_train)
            # 计算每个类别的样本权重
            class_weight = self.create_class_weight(labels_dict, 0.3)
            print('----class weight:{}'.format(class_weight))
            
            # 20
            best_acc = 0.0  # 跟踪最佳验证集准确度
            best_model = None  # 保存最佳模型
            for ep in range(self.max_epoch):
                # class_weight 被传递给了fit函数以平衡样本数量的不均衡
                self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=1, class_weight=class_weight)
                t_probs = self.model.predict(X_holdout)
                t_result = [0 if x <= 0.5 else 1 for x in t_probs]
                t_acc = accuracy_score(y_holdout, t_result)
                print("epoch:{}--------val acc:{}---------best_acc:{}".format(ep, t_acc, best_acc))
                if t_acc > best_acc:
                    best_model = self.model
                    best_acc = t_acc
            # 保存模型 
            model_json = best_model.to_json()   
            # 模型的权重保存在HDF5中
            # 模型的结构保存在JSON文件或者YAML文件中
            with open(model_add, "w") as json_file:
                json_file.write(model_json)
                self.model.save_weights(model_weight)
            print("Saved two-class model to disk")
        # 计算训练集分数
        # self.load(model_folder)
        # 计算每个输入样本属于类别1的预测概率
        y_pred = self.model.predict(X, batch_size=self.batch_size, verbose=1)
        # 进行预测之后，将y_pred数组扁平化处理
        y_pred = y_pred.flatten()
        # 该DataFrame的每一行对应一个样本的预测
        df = pd.DataFrame(y_pred)
        # 排序 具有更高预测属于类别1的概率的样本将出现在DataFrame的顶部
        df = df.sort_values(by=0, ascending=False)
        df.to_csv("./data/model/CNN_train_scores.csv", index=False, header=None)
    
    def load(self, model_folder):
        """
        将模型文件和权重值读取
        :param model_folder: 模型存储文件夹
        :return:
        """
        # 构建了模型结构（.json 文件）和模型权重（.h5 文件）的文件路径。
        model_add = "{}/CNN_model.json".format(model_folder)  # 模型文件
        model_weight_add = "{}/CNN_model.h5".format(model_folder)  # 权重文件
        with open(model_add, 'r') as json_file:
            # 读取打开的模型文件的内容，将其存储在变量model中
            model = json_file.read()
        #将从文件中读取的模型结构字符串解析为Keras模型对象
        self.model = model_from_json(model)
        # 权重参数
        self.model.load_weights(model_weight_add)
        # 读取之前训练过程中保存的训练分数
        # score_df = pd.read_csv("{}/LSTM_train_scores.csv".format(model_folder), names=['score'])
        # self.score_l = score_df['score'].tolist()
        # self.isload_ = True

    # 预测部分
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
        print('CNN accuracy:', acc)
        print('CNN precision:', precision)
        print('CNN recall:', recall)
        print('CNN F1:', score)
    
    def cal_p(self, s):
        """
        计算p_value, 二分查找
        :param s: float，是根据模型预测得出的分数，恶意概率的值，在0和1之间
        >0.5 的就是DGA的样本得分，<0.5的就是正常样本的分数
        :return:
        """
        flag = 0  # score偏da的对应的
        for i in range(len(self.score_l)):
            # 找到分数表中0.5 的分界点的索引
            if self.score_l[i] <= 0.5000000000000000:
                flag = i - 1
                break
        # print("flag:{}".format(flag))
        # 这部分是符合论文中的计算方式的，p^0=p^1=n/n=1
        if s >= self.score_l[0]:
            return 1.0
        if s <= self.score_l[-1]:
            return 1.0
        if s == self.score_l[flag]:
            # return 1 / ((flag + 1) * 1.0)
            return 0.0

        high_index = len(self.score_l)
        low_index = 0
        while low_index < high_index:
            # 使用二分查找的方式来确定 s 在 self.score_l 中的位置
            mid = int((low_index + high_index) / 2)
            if s > self.score_l[mid]:
                high_index = mid - 1
            # 找到了s所在的位置
            elif s == self.score_l[mid]:
                # 也是符合论文中的计算方式的
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

if __name__ == "__main__":
    # muldec = MultiModelDetection()

    # from feeds.danalysis import LDAClassifier
    clf = CNNClassifier()
    #clf.train(r"./data/model", r"./Year_Test/2019_2020_train.csv")
    clf.load(r"./data/model")
    clf.predict(r"./data/model", r"./Year_Test/2019_2020_test.csv")
    
