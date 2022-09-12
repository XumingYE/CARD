import os
from hashlib import md5
from fastcdc import fastcdc
from gensim.models import FastText
import shutil


def pre_dir_operation(dir_name_ori, model, avg_size=4096, fat=True, hf=md5, maxsize=None):
    # 加载模型
    model = FastText.load(model)
    # 定义存放的文件位置
    dir_name = dir_name_ori + "_fasttext"
    folder = os.path.exists(dir_name)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(dir_name)
    else:
        shutil.rmtree(dir_name)
        os.makedirs(dir_name)
    # 定义Diff,Origin文件的计数器
    diff_count = 0
    origin_count = 0

    # 定义分块大小
    min_size = avg_size // 2
    if maxsize is None:
        max_size = avg_size * 2
    else:
        max_size = maxsize
    before_size = 0
    for home, dirs, files in os.walk(dir_name_ori):
        for filename in files:
            fp = os.path.join(home, filename)
            before_size += os.path.getsize(fp)
            cdc = list(fastcdc(fp, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            with open(file=fp, mode="rb") as r:
                for chunk in cdc:
                    # 利用offset和length信息读取到相应块的内容 offset 相对于文件的首部位置来计算
                    r.seek(chunk.offset)
                    data = r.read(chunk.length)

                    fixed_size = chunk.length // feature_count
                    last_fixed_size = (chunk.length % feature_count) + fixed_size
                    learning_data = str()

                    for feature in range(feature_count - 1):
                        hash = hf()
                        hash.update(str(data[feature * fixed_size:(1 + feature) * fixed_size]).encode("utf-8"))
                        learning_data += hash.hexdigest()

                    hash = hf()
                    feature += 1
                    hash.update(str(data[feature * fixed_size:feature * fixed_size + last_fixed_size]).encode("utf-8"))
                    learning_data += hash.hexdigest()

                    # 找到与其数据域最相似的内容，由于模型存放的是十六进制，在这我也将其转换为16进制
                    most_similar = model.wv.most_similar(learning_data)
                    if len(most_similar) > 0:# and most_similar[0][1] >= 1:
                        # 得到最相似的数据域
                        similar_data = most_similar[0][0]
                        a = bytes.fromhex(similar_data)
                        b = data
                        # 利用a 和b 生成diff文件
                        delta.get_diff_file(a, b, dir_name + "/" + "diff" + str(diff_count))
                    else:
                        k = open(dir_name + "/" + "origin" + str(origin_count), "ab+")
                        k.write(data)
                    try:
                        if os.path.getsize(dir_name + "/" + "diff" + str(diff_count)) > 10 * (2 ** 20):
                            diff_count += 1
                        if os.path.getsize(dir_name + "/" + "origin" + str(origin_count)) > 10 * (2 ** 30):
                            origin_count += 1
                    except BaseException:
                        # 可能首次暂时不存在文件，os.path.getsize()会报错
                        pass
    after_size = 0
    for root, dirs, files in os.walk(dir_name):
        after_size += sum([os.path.getsize(os.path.join(root, name)) for name in files])
    print("before size : " + str(before_size))
    print("after size : " + str(after_size))
    try:
        DCR = before_size/after_size
    except ZeroDivisionError:
        DCR = 0
    print("DCR : " + str(DCR))


if __name__ == '__main__':
    import argparse
    import sys
    import time

    # import the pkg
    sys.path.extend(["../"])
    import fasttext_chunk.usemodel_predect_vsim.delta_v3 as delta
    import fasttext_chunk.usemodel_predect_vsim.log_v1 as log
    # parser
    parser = argparse.ArgumentParser(description='from dataset get corpus and model')
    parser.add_argument('--dataset_dir', type=str, help=" the dataset path")
    parser.add_argument('--average_size', type=int, default=4096, help="the chunk's average size (unit: B)")
    parser.add_argument('--model', type=str, help="result model path(include name)")
    args = parser.parse_args()

    # get params
    dataset_dir = args.dataset_dir
    average_size = args.average_size
    model = args.model

    l = log.LogSystem()
    l.print_log("dataset_dir : " + dataset_dir)
    l.print_log("average_size : " + str(average_size))
    l.print_log("model : " + model)
    start = time.clock()

    # train
    pre_dir_operation(dataset_dir, model, avg_size=average_size)
    end = time.clock()
    print("cpu run time : " + str(end - start))
