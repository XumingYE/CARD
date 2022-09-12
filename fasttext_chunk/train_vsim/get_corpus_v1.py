import os
from fastcdc import fastcdc
from hashlib import md5


def get_corpus(dir_name, avg_size=4096, fat=True, hf=md5, package_count=4, maxsize=None):
    file_count = 0
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
            with open(file=fname, mode="rb", buffering=2 ** 20) as r:
                vir_count = 0
                learning_file = open(dir_name + "_learning/" + "learning_file" + ".txt", "a+")
                for chunk in cdc:
                    r.seek(chunk.offset)
                    data = r.read(chunk.length)
                    learning_data = data.hex()
                    if vir_count % package_count == 0 and vir_count != 0:
                        learning_file.write(learning_data + "\n")
                    else:
                        learning_file.write(learning_data + "  ")
                        # print("same")
                    vir_count += 1
                learning_file.write("\n")
                learning_file.flush()
                learning_file.close()
            file_count += 1


if __name__ == '__main__':
    get_corpus("E:/ISO/train_data")

