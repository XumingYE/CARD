from gensim.models import FastText
model = FastText.load("/home/yxm/model/think_8192_150dim_12S.model")
print(model.wv.most_similar("hello")[0][0])