**目录结构**

1. 代码文件包含许多版本，如（xx_v1.py, xx_v2.py）一切都以最终版为准，如（xx_v2.py）

2. 日志代码为自己编写， log_v*.py 文件
3. delta 采用的是xdelta3 
4. fasttext_chunk 提取特征值时采用的是minhash
5. rabin fingerprint 采用的是pyrabin0.6
6. **代码包含虚拟环境**

- fasttext_chunk

  使用神经网络模型来生成数据块的特征向量

  - train_vsim 包含了训练过程，即具体查看train.py, 里面有较详细的注释 
  - usemodel_predect_vsim 使用模型进行预测，具体查看predect.py(之前拼写错了，应该是predict,  后面没改了，有强迫症的同学可以帮忙改一下，并删掉旧版本的 py 文件)。

- finesse_chunk

  对应19'th FAST文章 

  - chunking_v3.py  主函数，具体参数参考文件里面的注释
  - subChunk.py 得到数据块的SFs

- N_SF chunk

  和finesse_chunk结构类似，不再赘述

