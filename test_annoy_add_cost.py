from gensim.models import FastText
from gensim.similarities.annoy import AnnoyIndexer
import time
model = FastText.load("/home/yxm/model/kernel_512_150dim_6epoch.model")


vector = model.wv.get_vector("1234")
model.wv.add_vector(key="1234", vector=vector)
start = time.clock()
annoy_index = AnnoyIndexer(model, model.wv.vector_size)
end = time.clock()
print(end - start)
vector = model.wv.get_vector("asdsa")
model.wv.add_vector(key="asdsa", vector=vector)
vector = model.wv.get_vector("hjfkas")
model.wv.add_vector(key="hjfkas", vector=vector)
vector = model.wv.get_vector("jfksdfbb")
model.wv.add_vector(key="jfksdfbb", vector=vector)
vector = model.wv.get_vector("qwdhsuihdff")
model.wv.add_vector(key="qwdhsuihdff", vector=vector)
start_2 = time.clock()
annoy_index_2 = AnnoyIndexer(model, model.wv.vector_size)
end_2 = time.clock()
print(end_2 - start_2)