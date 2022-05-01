import pickle
# import config
class ConvertToNum:
    """
    将字符串转化为数字
    """
    UNK_TAG = 'UNK'
    PAD_TAG = 'PAD'
    SOS_TAG = 'SOS'
    EOS_TAG = 'EOS'

    UNK = 0
    PAD = 1
    SOS = 2
    EOS = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.SOS_TAG: self.SOS,
            self.EOS_TAG: self.EOS,
        }
        self.fited = False

    def fit(self, sequence, min_count=1, max_count=None, max_feature=None):
        """
        fit数据进入词典
        :param sequence: [word1,word3,wordn..]
        :param min_count:最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """
        count = {}
        for a in sequence:
            if a not in count:
                count[a] = 0
            count[a] += 1


        # 词频限制
        if min_count is not None:
            count = {k: v for k, v in count.items() if v >= min_count}

        if max_count is not None:
            count = {k: v for k, v in count.items() if v <= max_count}

        # 特征限制
        if isinstance(max_feature, int):
            count = dict(sorted(count.items(), key=lambda x: x[1])[:max_feature])
        # print(count)
        for k in count:
            self.dict[k] = len(self.dict)

        self.fited = True



    def transform(self, sentence, max_len=None, add_eos=False):
        """
        :param sentence: [word1,word3,wordn..]
        :return: [1,3,n..]
        """
        # print(self.dict)
        if max_len:
            r = [self.PAD]*max_len
        else:
            r = [self.PAD]*len(sentence)
        if max_len and len(sentence) > max_len:
            sentence = sentence[:max_len]
        for index, word in enumerate(sentence):
            r[index] = self.dict.get(word, self.UNK)
        if add_eos:
            if r[-1] == self.PAD:
                # 句子中有PAD，在PAD之前添加EOS
                pad_index = r.index(self.PAD)
                r[pad_index] = self.EOS
            else:
                # 句子中没有PAD，在最后添加EOS
                r[-1] = self.EOS


        return r

    def inverse_transform(self, indices):
        """
        :param indices: [1,3,n..]
        :return: [word1,word3,wordn..]
        """
        # print(self.dict)
        self.inverse_dict = {v: k for k, v in self.dict.items()}
        # print(self.inverse_dict)
        sentence = []
        for i in indices:
            word = self.inverse_dict.get(i)
            if i != self.EOS:  # 把EOS之后的内容删除，123---》1230EOS，predict 1230EOS123
                sentence.append(word)
            else:
                break
        return sentence

    # def build_dict(self):
    #     """
    #     存储词典
    #     :return:
    #     """
    #     convert_to_num = ConvertToNum()
    #     convert_to_num.fit(list('0123456789'))
    #     pickle.dump(convert_to_num, open('./utils/convert_to_num.pkl', 'wb'))

    def __len__(self):
        return len(self.dict)


if __name__ == "__main__":
    convert_to_num = ConvertToNum()
    convert_to_num.fit(list('01234'))
    pickle.dump(convert_to_num, open('./utils/convert_to_num.pkl', 'wb'))
    convert_to_num = pickle.load(open('./utils/convert_to_num.pkl', 'rb'))
    ret = convert_to_num.transform(list("023"))
    print(ret)
    print(convert_to_num.inverse_transform(ret))

