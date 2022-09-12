from fastcdc import fastcdc
from hashlib import md5
from datetime import datetime

import os
import json
from finesse_chunk import subChunk, delta


def cdc_chunking(dir_name, source_file, avg_size=4096, fat=True, hf=md5, subChunk_count = 12, per_package_count=4, maxsize=None, similarity=0.3, to_dir_name="E:/finesse"):

    min_size = avg_size // 2
    if maxsize is None:
        max_size = avg_size * 2
    else:
        max_size = maxsize
    vir_count = 0
    folder = os.path.exists(to_dir_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(to_dir_name)
    group_count = subChunk_count//per_package_count


    # features = []
    # for line in open("E:/finesse", "r"):
    #     features.append(line.split())

    for home, dirs, files in os.walk(dir_name):
        for filename in files:
            file = os.path.join(home, filename)
    # 分离文件名 包含后缀
            lindex = file.rfind("/") if file.rfind("/") != -1 else file.rfind("\\")
            rindex = file.rfind(".")
            filename = file[lindex + 1:rindex]
            cdc = list(fastcdc(file, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            with open(file=file, mode="r", encoding='gb18030', errors='ignore') as r:
                for i in cdc:
                    local_similarity = 0
                    r.seek(i.offset)
                    content = r.read(i.length)
                    # 得到每个子块的super-features
                    sub_chunks = subChunk.get_fixed_chunks_by_bytes(str(content), chunk_count=subChunk_count)
                    sf = subChunk.SubChunk.get_features(sub_chunks, chunk_count=subChunk_count, group_count=per_package_count)
                    flag = False
                    similarity_chunk_index = -1
                    count = 0
                    for line in open(source_file):
                        for k in range(group_count):
                            if sf[k] in line.split():
                                local_similarity += 1
                            if local_similarity/group_count >= similarity:
                                flag = True
                                similarity_chunk_index = count
                                break
                        if flag:
                            break
                        count += 1
                    if flag:
                        # TODO 使用delta encoding 保存文件
                        a_file = open("E:/finesse_corpus/corpus"+str(count), "r")
                        a_data = a_file.read()
                        delta.get_diff_file(a_data, content, fp="E:/finesse_result"+"/"+"diff"+str(vir_count))
                        print("SUCCESS")
                        vir_count += 1
                    else:
                        with open("E:/finesse_result/origin"+str(vir_count), "w+") as origin:
                            origin.write(content)
                        vir_count += 1
                        print("sf:   "+str(sf))


class Chunk:
    def __init__(self, name, offset, data, length, features=None):
        self.name = name
        self.features = features
        self.offset = offset
        self.data = data
        self.length = length
        filename = "result_file/" + self.features
        with open(filename, "wb+") as f:
            f.write(self.data)


if __name__ == '__main__':
    # for i in range(2, 12):
    #     filename = "../source_file/thesis" + str(i) + ".txt"
    #     cdc_chunking(filename)
    cdc_chunking("kernel_3", "E:/finesse", to_dir_name="E:/finesse_result", avg_size=512, maxsize=1024)