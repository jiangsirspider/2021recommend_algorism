# 计算准确率 召回率 F值

import config
import predict
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report

# 商家预算

budget = config.budget

# print(budget.head())

# 商家位置
location = config.location
# print(location.head())

# 表二线下购物记录
df2 = config.df2[['use_ID','mer_ID', 'loc_ID']].sort_values('use_ID').drop_duplicates(['use_ID','mer_ID','loc_ID'], keep='first')
# print(df2.head())

# 召回率考察表格
recall_df = pd.DataFrame()
# print(config.df2.head())

# 结果表的遍历和写入
result_df = config.df4
def func(line):
    global budget
    user_id = line['use_ID']
    loc_id = line['loc_ID']
    target = line['mer_ID']
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
    # print(recommend_ls[[1]] > 0.5)
    # recommend_ls=recommend_ls.loc[recommend_ls[[1]] > 0.5]
    print(recommend_ls)

    # print(type(recommend_ls[[0]].values))
    recommend_ls = recommend_ls[[0]].values
    # 去掉最后一个维度
    recommend_ls = recommend_ls[:, 0]
    # if len(recommend_ls) > 1:
    #     recommend_ls = np.squeeze(recommend_ls).tolist()
    # print(recommend_ls)
    print(':'.join(recommend_ls))
    line['Merchant_id_list'] = ':'.join(recommend_ls)
    if target in recommend_ls:
        line['target'] = 1
        line['predict'] = 1
    else:
        line['target'] = 1
        line['predict'] = 0

    # 推荐后budget减少1
    for i in recommend_ls:
        budget.loc[budget.mer_ID == i, 'budget'] -= 1
    # print(recall_loc_ls)
    # raise Exception
    return line

if __name__ == "__main__":
    # r = df2.apply(func, axis=1)
    # pickle.dump(r, open('./utils/predict.pkl', 'wb'))
    # print(r)
    r = pickle.load(open('./utils/predict.pkl', 'rb'))
    print(r.head())
    #     # precision:预测结果为正例样本中真实为正例的比例（查得准）
    #     # recall: 真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力
    #     # F1-score: 反映了模型的稳健型
    print(r[['Merchant_id_list', 'mer_ID']].head())
    report = classification_report(r.target, r.predict)
    print(report)

# # 十次抽样，每次在df2中抽取1000样本，计算 f1score  precision  recall
# for i in range(10):
#     # 在df2数据集中一个商家平均对应539个用户，故单次采样1000
#     df2 = config.df2[['use_ID', 'mer_ID', 'loc_ID']].sample(1000)
#     r = df2.apply(func, axis=1)
#     # pickle.dump(r, open('./utils/predict.pkl', 'wb'))
#     # r = pickle.load(open('./utils/predict.pkl', 'rb'))
#     # print(r)
#     # precision:预测结果为正例样本中真实为正例的比例（查得准）
#     # recall: 真实为正例的样本中预测结果为正例的比例（查的全，对正样本的区分能力
#     # F1-score: 反映了模型的稳健型
#     report = classification_report(r.target, r.predict)
#     # print("每个类别的分类报告：", report)
#     # print(type(report))
#     report = report.strip(' ').replace('\t',';').split('\n')
#     avg = report[-2].split('      ')
#     precision = np.float(avg[1])
#     recall = np.float(avg[2])
#     f1score = np.float(avg[3])
#     recall_df = recall_df.append([{'precision':precision, 'recall':recall, 'f1score':f1score}])
# print(recall_df)

# 十次抽样，每次在df2中抽取1000样本，计算 f1score  precision  recall
#    f1score  precision  recall
# 0     0.93        1.0    0.86
# 0     0.91        1.0    0.83
# 0     0.91        1.0    0.84
# 0     0.91        1.0    0.83
# 0     0.91        1.0    0.84
# 0     0.92        1.0    0.84
# 0     0.90        1.0    0.82
# 0     0.90        1.0    0.82
# 0     0.91        1.0    0.83
# 0     0.89        1.0    0.81

