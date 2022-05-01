import config
import predict
import pandas as pd
import numpy as np
import pickle
import torch
from multiprocessing import Process


# 商家预算

budget = config.budget

print(budget.head())

# 商家位置
location = config.location
print(location.head())


# 结果表的遍历和写入
result_df = config.df4

result_df2 = pd.DataFrame()

def func(line,num):
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
    pickle.dump(result_df2, open('./utils/result{}.pkl'.format(num), 'wb'))
    return line
#
def sub_task(num, df):
    r = df.apply(func, num=num, axis=1)
    pickle.dump(r, open('./utils/result{}.pkl'.format(num), 'wb'))
    # r = pickle.load(open('./utils/result.pkl', 'rb'))
    # print(r)

# r = df.apply(func, num=num, axis=1)
# pickle.dump(r, open('./utils/result{}.pkl'.format(num), 'wb'))
# # r = pickle.load(open('./utils/result.pkl', 'rb'))
# print(r)


if __name__ == '__main__':
    print("---主进程开始执行---")
    torch.multiprocessing.set_start_method('forkserver', force=True)
    df1 = result_df.iloc[:140000]
    p1 = Process(target=sub_task, args=(1,df1))
    df2 = result_df.iloc[140000:280000]
    p2 = Process(target=sub_task, args=(2,df2))
    df3 = result_df.iloc[280000:]
    p3 = Process(target=sub_task, args=(3,df3))
    # df4 = result_df.iloc[195000:260000]
    # p4 = Process(target=sub_task, args=(4,df4))
    # df5 = result_df.iloc[260000:320000]
    # p5 = Process(target=sub_task, args=(5,df5))
    # df6 = result_df.iloc[320000:]
    # p6 = Process(target=sub_task, args=(6,df6))

    # df1 = result_df.iloc[:333]
    # p1 = Process(target=sub_task, args=(1,df1))
    # df2 = result_df.iloc[333:666]
    # p2 = Process(target=sub_task, args=(2,df2))
    # df3 = result_df.iloc[666:]
    # p3 = Process(target=sub_task, args=(3,df3))
    # df4 = result_df.iloc[498:664]
    # p4 = Process(target=sub_task, args=(4,df4))
    # df5 = result_df.iloc[664:830]
    # p5 = Process(target=sub_task, args=(5,df5))
    # df6 = result_df.iloc[830:]
    # p6 = Process(target=sub_task, args=(6,df6))

    p1.start()
    p2.start()
    p3.start()
    # p4.start()
    # p5.start()
    # p6.start()
    p1.join()
    p2.join()
    p3.join()
    # p4.join()
    # p5.join()
    # p6.join()
    print("--主进程执行结束---")



