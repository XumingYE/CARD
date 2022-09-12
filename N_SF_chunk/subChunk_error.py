from rabin_origin.utils import RabinFingerprint
import os
from hashlib import md5


def extracting_N_sf(string, n):
    length = len(string)
    subChunkSize = length // n
    window = 256 if subChunkSize < 2 ** 10 else subChunkSize // (2 ** 3)
    rabin = RabinFingerprint(window)
    feature = [0] * n
    for m in range(length):
        FP = rabin.feed_many(string[:m], is_str=True)
        for i in range(n):
            if feature[i] <= FP:
                feature[i] = FP
    return feature

def get_fixed_chunks_by_bytes(data, method=md5, chunk_count=12):
    features = extracting_N_sf(data, chunk_count)
    file_size = len(data)
    avg_size = file_size // chunk_count
    over_size = file_size % chunk_count
    for i in range(chunk_count):
        sha_256 = method()
        start = i * avg_size
        end = (i + 1) * avg_size
        if i == chunk_count:
            end += over_size
        sub_data = data[start:end]
        sha_256.update(str(features[i]).encode("UTF-8"))
        yield SubChunk(data=sub_data, feature=sha_256.hexdigest())


def get_fixed_chunks_by_file(file, method=md5, chunk_count=12):
    data = open(file, "r").read()
    features = extracting_N_sf(data, chunk_count)
    file_size = os.path.getsize(file)
    avg_size = file_size // chunk_count
    over_size = file_size % chunk_count

    with open(file, "r") as f:
        for i in range(chunk_count):
            sha_256 = method()
            if i == chunk_count:
                avg_size += over_size
            f.seek(i * avg_size)
            data = f.read(avg_size)
            sha_256.update(str(features[i]).encode("UTF-8"))
            yield SubChunk(data=data, feature=sha_256.hexdigest())


class SubChunk:
    identify = 1

    def __init__(self, data, feature):
        self.id = SubChunk.identify
        self.data = data
        self.feature = feature
        SubChunk.identify += 1

    def __str__(self):
        return "{\nid: " + str(self.id) + "\ndata: " + self.data + "\nfeature: " + str(self.feature) + "\n}"

    @staticmethod
    def get_features(fixed_chunks, method=md5, chunk_count=12, group_count=4):
        features = [[""] for _ in range(group_count)]
        sub_count = chunk_count // group_count
        for group in range(group_count):
            tmp = []
            for sub in range(sub_count):
                chunk = next(fixed_chunks)
                tmp.append(chunk.feature)
            features[group] = tmp
            features[group].sort()
        SFs = []
        for i in range(sub_count):
            hash_method = method()
            for k in range(group_count):
                hash_method.update(str(features[k].pop()).encode("UTF-8"))
            SFs.append(hash_method.hexdigest())
        return SFs


if __name__ == '__main__':

    # for i in range(191):
    #     sub_chunks = get_fixed_chunks_by_file("chinese" + str(i) + ".txt")
    #     result = SubChunk.get_features(sub_chunks, method=md5)
    #     print(result)
    # f = open("../test4.txt", "r")
    # content = f.read()
    # sfs = extracting_N_sf(content, 12)

    chunks = get_fixed_chunks_by_file("../test4.txt", chunk_count=16)
    sfs = SubChunk.get_features(chunks, chunk_count=16, group_count=8)
    for s in sfs:
        print(s)

