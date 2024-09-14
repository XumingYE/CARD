import torch
from model import Fasttext
from annoy import AnnoyIndex
def get_similar_chunk_index(data, model_path, annoy_path, max_vocab, max_ngram, hidden, n=1):
    """
    主函数，接收数据块，返回最相似的index
    :param hidden:
    :param max_ngram:
    :param max_vocab:
    :param n: 找最近的n个
    :param model_path: 模型的加载路径
    :param data: 数据块内容
    :param annoy_path: annoy 索引的加载路径
    :return: 最相似的数据块的index
    """
    # print("define")
    model = Fasttext(max_vocab, max_ngram, hidden)
    # print("load")
    model.load_state_dict(torch.load(model_path))
    # print("eval")
    model.eval()
    vector = model.predict(data)
    return get_most_similar_index(vector, annoy_path, n=n)
def get_most_similar_index(vector, annoy_path, n):
    """
    :param n: 找最近的n个
    :param vector: 向量
    :param annoy_path: annoy 索引保存的路径
    :return: 当前数据块的index, 最相似的数据块的index
    """
    table = AnnoyIndex(50, 'angular')
    table.load(annoy_path)
    most_similar_index = table.get_nns_by_vector(n=n, vector=vector)
    table.add_item(table.get_n_items(), vector) # table.get_n_items() 为当前的annoy树中最后一个索引 + 1
    table.save(annoy_path)
    return table.get_n_items()-1, most_similar_index
if __name__ == '__main__':
    import argparse
    import sys

    sys.path.extend(["../../"])
    # parser
    parser = argparse.ArgumentParser(description='return the most similar item of vector to the chunk')
    parser.add_argument('--chunk', type=str, help="chunk bytes")
    parser.add_argument('--max_vocab', type=int, default=50000, help='max vocab size')
    parser.add_argument('--max_ngram', type=int, default=400000, help='max n-gram bag size')
    parser.add_argument('--hidden', type=int, default=128, help='hidden embedding dimension')
    parser.add_argument('--model_path', type=str, help="model path")
    parser.add_argument('--annoy_path', type=str, help="annoy path")
    parser.add_argument('--n', default=1, type=int, help="n most similar index")
    args = parser.parse_args()
    print(get_similar_chunk_index(args.chunk, args.model_path, args.annoy_path, args.max_vocab, args.max_ngram, args.hidden, args.n)[0])