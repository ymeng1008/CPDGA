# -*- coding: utf-8 -*-
"""
Created on 2020/8/16 12:38

__author__ = "Congyi Deng"
__copyright__ = "Copyright (c) 2021 NKAMG"
__license__ = "GPL"
__contact__ = "dengcongyi0701@163.com"

Description:

"""
from importlib import import_module
import warnings
warnings.filterwarnings('ignore')
import os
import sys
from configparser import ConfigParser
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 多模型检测
class MultiModelDetection:

    def __init__(self):
        # 存储配置信息。
        self._cfg = dict()
        cp = ConfigParser()
        cp.read('config.ini')
        self._cfg["model_path"] = cp.get('files', 'model_path')
        self._cfg["train_add"] = cp.get('files', 'train_add')
        self._cfg["test_add"] = cp.get('files', 'test_add')
        self._cfg["algorithm_lst"] = cp.get('feeds', 'algorithm_list').split(',')
        self._cfg["classifier_lst"] = cp.get('feeds', 'classifier_list').split(',')
        self._load_models()

    def _load_models(self):
        """
        将训练好的多个模型全部预加载到内存中
        :return:
        """
        self._clf_list = list() # _clf_list 用于存储加载的多个模型。
        for i in range(len(self._cfg["algorithm_lst"])):
            aMod = import_module('feeds.'+self._cfg["algorithm_lst"][i])
            aClass = getattr(aMod, self._cfg["classifier_lst"][i])
            clf = aClass()
            clf.load(self._cfg["model_path"])
            self._clf_list.append(clf)

    def multi_predict_single_dname(self, dname):
        """
        对单个域名进行多模型协同检测
        :param dname: 域名
        :return: (基础检测结果——字典类型,多模型检测结果——0安全1危险2可疑）
        """
        # 用于存储基础检测结果
        base_result = dict()
        base_result_t = dict()
        # 依次调用每个模型的 predict_single_dname 方法对域名进行检测
        for i in range(len(self._clf_list)):
            clf_pre_rs = self._clf_list[i].predict_single_dname(self._cfg["model_path"], dname)
            # label, score(模型返回的概率值), p_value
            base_result[self._cfg["classifier_lst"][i][:-10]] = [clf_pre_rs[0], format(clf_pre_rs[1], '.4f'),
                                                                 clf_pre_rs[2]]
            # self._cfg["classifier_lst"][i][:-10] 每个模型类名的前缀用作字典的键
            # p与0.01 比较， 确定标签是否分配为可疑（2）好像和论文中的判断方法不太一样？
            base_result_t[self._cfg["classifier_lst"][i][:-10]] = clf_pre_rs if clf_pre_rs[2] > 0.01 \
                else (2, clf_pre_rs[1], clf_pre_rs[2])
        rs_list = list()
        # 将每个模型的检测结果中的第一个元素（安全、危险、可疑）添加到 rs_list 中
        for j in base_result_t:
            rs_list.append(base_result_t[j][0])
        #  如果 rs_list 中的元素都相同，说明所有模型的检测结果一致
        if len(set(rs_list)) == 1:
            if list(base_result_t.values())[0][0] != 2:
                # 返回所有模型的检测结果,取第一个元素的第一个值（检测结果）作为最终结果
                result = list(base_result_t.values())[0][0]
                return base_result, result
            # 如果所有模型都返回了可疑结果，
            # 这里和论文写的一样吗，目前认为是一样的，推导了一遍，（cre*con）在这个代码中的意思其实就是P,P=max(p^0,p^1),
            # 所以其实排序找最大的P是没问题的
            # 根据p值计算每一个模型对于该样本的Confidence和credibility的值
            # 利用乘积（cre*con）选出质量最高的模型
            elif list(base_result_t.values())[0][0] == 2:  # 所有模型都表现很差
                # 排序，p值最高的模型排在最前面
                sort_result = sorted(base_result_t.items(), key=lambda base_result_t: base_result_t[1][2], reverse=True)
                # 直接用该分数值与0.5 进行比较
                # 为什么要和0.5进行比较
                # 是模型的质量实在太低了？？
                # 论文里没有这个部分还这样写吗
                if sort_result[0][1][2] <= 0.5:
                    # 如果最优模型质量太低的话，就忽略这个步骤
                    result = 2
                else:
                    result = sort_result[0][1][0]
                return base_result, result
        # 模型中的检测结果不一致
        # 这里和论文的方式不太一样，这里没有考虑投票的方式，
        # 而是直接采用了选取最大的（cre*con）
        # 论文中：先采用投票的方式决定Label("可疑结果不纳入其中")
        # 如果出现了票数相等的情况，考虑（cre*con）选最优
        new_result = dict()
        for k in base_result_t:
            # 将不是可疑结果的模型检测结果存储在 new_result 字典中
            if base_result_t[k][0] != 2:
                # 可疑结果不考虑其中
                new_result[k] = base_result_t[k]

        # 去除可疑结果后投票
        # 投票部分
        # 这部分只是简单的选取了票数作为选择标准
        # 如果需要加入权重再修改
        # 获取所有模型的标签结果
        model_labels = [new_result[k][0] for k in new_result]
        # 计算标签结果的投票数
        label_counts = Counter(model_labels)
        # 找到获得最高票数的标签
        max_vote_label = label_counts.most_common(1)[0][0] 
        # 如果只有一个标签获得最高票数，直接返回该标签
        if label_counts[max_vote_label] == 1:
            result = max_vote_label
        else:
            # 如果有多个标签获得最高票数，则综合考虑Confidence和Credibility（P）来选取最终的Label
            # 去除可疑标签后，只剩下了两个标签，两个标签票数一致，那么就只需计算每个标签的P值
            # 直接选取最大的P值所在的标签作为标签。
            sort_result = sorted(new_result.items(), key=lambda new_result: new_result[1][2], reverse=True)
            # 如果 p 值低于等于0.5，则将结果设为可疑（2）
            # 论文中好像不计入可疑结果
            # 这里先注释掉？
            #if sort_result[0][1][2] <= 0.5:
                # result = 2
            #else:
            # 否则，将结果设为排序后第一个模型的检测结果
            result = sort_result[0][1][0]
        # 返回基础检测模型的字典和多模型检测的字典
        return base_result, result

    # 多模型协同训练不用训练，直接预测就行了
    
    
    def multi_predict(self, model_folder, test_feature_add):
        """
        批量检测
        :param model_folder: 模型存储文件夹
        :param test_feature_add: 批量测试文件路径
        :return:
        """
        # 先加载模型
        if not self.isload_:
            self._load_models()
            self.isload_ = True
        # 获取测试数据， domain，label
        # 和训练时的流程一样
        

    


if __name__ == "__main__":
    # muldec = MultiModelDetection()

    from feeds.danalysis import LDAClassifier
    clf = LDAClassifier()
    # clf.train(r"./data/model", r"./data/features/train_features.csv")
    clf.load(r"./data/model")
    clf.predict(r"./data/model", r"./data/features/test_features.csv")


