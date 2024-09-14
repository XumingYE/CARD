**目录结构**

1. 代码文件包含许多版本，如（xx_v1.py, xx_v2.py）一切都以最终版为准，如（xx_v2.py）

2. 日志代码为自己编写， log_v*.py 文件
3. delta 采用的是xdelta3 
4. fasttext_chunk 提取特征值时采用的是minhash
5. rabin fingerprint 采用的是pyrabin0.6

- fasttext_chunk

  使用神经网络模型来生成数据块的特征向量

  - 训练过程具体查看train.py, 里面有较详细的注释 
  - evaluationWithCache 使用模型进行预测

- finesse_chunk

  对应19'th FAST文章 

  - chunking_v3.py  主函数，具体参数参考文件里面的注释
  - subChunk.py 得到数据块的SFs

- N_SF chunk

  和finesse_chunk结构类似，不再赘述

