import os
from fastcdc import fastcdc
from hashlib import md5
import shutil

def get_corpus(dir_name, avg_size=4096,package_count=4, fat=True, hf=md5, maxsize=None, feature_count=12):
    file_count = 0
    folder = os.path.exists(dir_name + "_origin_file/")
    if not folder:
        os.mkdir(dir_name + "_origin_file/")
    else:
        shutil.rmtree(dir_name + "_origin_file/")
        os.mkdir(dir_name + "_origin_file/")
    for home, dirs, files in os.walk(dir_name):
        for filename in files:
            fname = os.path.join(home, filename)
            min_size = avg_size // 2
            if maxsize is None:
                max_size = avg_size * 2
            else:
                max_size = maxsize
            cdc = list(fastcdc(fname, min_size=min_size, avg_size=avg_size, max_size=max_size, fat=fat, hf=hf))
            folder = os.path.exists(dir_name+"_learning")
            if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
                os.makedirs(dir_name+"_learning")
            with open(file=fname, mode="rb") as r:
                vir_count = 0
                with open(dir_name + "_learning/" + "learning_file" + ".txt", "a") as learning_file:
                    for chunk in cdc:
                        r.seek(chunk.offset)
                        data = r.read(chunk.length)
                        fixed_size = chunk.length // feature_count
                        last_fixed_size = (chunk.length % feature_count) + fixed_size
                        learning_data = str()

                        for feature in range(feature_count - 1):
                            hash = hf()
                            hash.update(str(data[feature*fixed_size:(1+feature) * fixed_size]).encode("utf-8"))
                            learning_data += hash.hexdigest()

                        hash = hf()
                        feature += 1
                        hash.update(str(data[feature * fixed_size:feature * fixed_size + last_fixed_size]).encode("utf-8"))
                        learning_data += hash.hexdigest()
                        hash = hf()
                        hash.update(learning_data.encode("utf-8"))
                        origin_filename = hash.hexdigest()
                        with open(dir_name + "_origin_file/" + origin_filename, "wb") as origin_file:
                            origin_file.write(data)
                        if vir_count % package_count == 0 and vir_count != 0:
                            learning_file.write(learning_data + "\n")
                        else:
                            learning_file.write(learning_data + "  ")
                        vir_count += 1
            file_count += 1


if __name__ == '__main__':
    get_corpus("E:/ISO/test_small_data")

