def get_n_grams(sentence, n=3, horizon=True, single_word=False):
    """
    :param n: n 为窗口大小
    :param single_word: 是否是单个字符串
    :param horizon: 是否为二维数组
    :param sentence: 一个单词数组
    :return: 字符级别的n-gram
    """
    # if type(sentence) is not str:
    #     import re
    #     sentence = [re.findall(r'[a-zA-Z]', sentence, re.S)]
    # prefix_sentence = ['<' + sentence[i] + '>' for i in range(len(sentence))]
    if horizon:
        char_grams = [word[i:i + n] for word in sentence for i in range(len(word) - n + 1)]
    elif single_word:
        char_grams = [sentence[i:i + n] for i in range(len(sentence) - n + 1)]
    else:
        char_grams = []
        for word in sentence:
            # char_grams = [word[i:i + n] for word in prefix_sentence for i in range(len(word) - n + 1)]
            char_grams.append([word[i:i + n] for i in range(len(word) - n + 1)])
    return char_grams