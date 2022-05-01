import config
import predict
import pandas as pd
import numpy as np
import pickle
from multiprocessing import Process, Queue, Pool, Manager, Pipe
# from multiprocessing import Process
# 商家预算

budget = config.budget

print(budget.head())

# 商家位置
location = config.location
print(location.head())


# 结果表的遍历和写入
result_df = config.df4

result_df2 = pd.DataFrame()

def func(line):
    global budget
    global result_df2
    user_id = line['use_ID']
    loc_id = line['loc_ID']
    # 召回附近的商店

    recall_loc_ls = location.loc[location.loc_list == loc_id].mer_ID.values.tolist()
    # 推荐并排序
    d = {}
    for i in recall_loc_ls:
        b_line = budget.loc[budget.mer_ID == i].to_dict(orient="records")[0]
        # print(b_line.to_dict(orient="records"))
        # print(b_line['budget'])
        b = '1' if b_line['budget'] >= 1 else '-1'
        s = b_line['star_level']
        score = predict.predict([b, s])
        d[i] = score
    recommend_ls = sorted(d.items(), key=lambda kv:kv[1], reverse=True)[:10]
    recommend_ls = pd.DataFrame(recommend_ls)

    # print(type(recommend_ls[[0]].values))
    recommend_ls = recommend_ls[[0]].values
    # 去掉最后一个维度
    recommend_ls = recommend_ls[:, 0]
    # if len(recommend_ls) > 1:
    #     recommend_ls = np.squeeze(recommend_ls).tolist()
    # print(recommend_ls)
    # print(':'.join(recommend_ls))
    line['Merchant_id_list'] = ':'.join(recommend_ls)
    # 推荐后budget减少1
    for i in recommend_ls:
        budget.loc[budget.mer_ID == i, 'budget'] -= 1
    # print(recall_loc_ls)
    # raise Exception
    result_df2 = result_df2.append(line)
    print('{}'.format(len(result_df2)))
    pickle.dump(result_df2, open('./utils/resultsss.pkl', 'wb'))
    return line
#
r = result_df.apply(func, axis=1)
pickle.dump(r, open('./utils/resultsss.pkl', 'wb'))
# r = pickle.load(open('./utils/result.pkl', 'rb'))
print(r)



