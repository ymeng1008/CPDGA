# CPDGA
CPDGA，基于情报繁殖的DGA域名主动检测算法，可以主动预测未来恶意域名的分布范围。与传统方法不同，该方法不是用二进制标签（0或1）预测域名，而是通过计算每个域名在每个类别中的置信度和可信度来确定其标签。
1. 基于2019年至2023年连续五年的恶意域进行了概念漂移现象的分析和防御。该模型使用2019年的数据进行训练，然后利用生成的数据对未来数据进行预测。通过这种方式，CPDGA能够有效减轻检测模型中的概念漂移现象。实验结果表明，CPDGA在解决概念漂移问题方面取得了显著成效，改进率高达20.4％。
<img width="378" alt="1ece7ecb906e8c321cf58f151f22d81" src="https://github.com/ymeng1008/CPDGA/assets/167075268/28dafdb1-e67c-43d7-9a95-ce8534807e12">
<img width="595" alt="3952fef5a90110fafeb04961889dee0" src="https://github.com/ymeng1008/CPDGA/assets/167075268/0e4f0806-58aa-4cf8-8b2f-513e40144699">

  
2. 评估了CPDGA对对抗性恶意域名的检测能力。CPDGA对当前最先进的敌对DGAs（包括DeepDGA、CharBot、Khaos和ReplaceDGA）具有100％的准确性。


DGA_Malware_Domains_process.py：DGA域名预处理，特征提取归一化

Conformal_Clustering.py: CPDGA算法整体流程

YearPrediction.py: 2019年至2023年连续五年的恶意域相关概念漂移现象的分析和防御


