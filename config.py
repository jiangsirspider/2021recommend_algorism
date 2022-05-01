# from convert_to_num2 import ConvertToNum
import pandas as pd
import numpy as np
import torch


import pickle

"""
随着移动设备在我们的日常生活中无处不在，基于位置的服务 (LBS) 变得越来越重要。人们越来越习惯于通过各种基于位置的服务分享他们的实时位置，
例如导航、叫车、餐厅/酒店预订等。 因此，积累了大量的用户数据，点燃了人们的热情来自机器学习/数据挖掘社区的联合力量，
揭示了我们日常生活中潘多拉魔盒的魔力，其中需要探索高维时空复杂性。在本次比赛中，当用户进入他们过去很少访问的新区域时，
我们将重点关注附近的商店推荐。比赛有两个新奇之处：首先，您应该调查在线和现场偏好之间的相关性是否有助于推荐附近的商店。
阿里巴巴集团拥有中国最大的在线零售平台淘宝网和天猫网，为超过 1000 万商家和超过 3 亿客户提供服务。
同时，蚂蚁金服的支付宝为众多客户提供名为口碑的餐厅和零售店推荐和支付服务。享受这两个群体提供的服务的用户往往拥有一个统一的在线账户。
虽然淘宝和天猫已经运行多年，积累了大量的消费者行为数据，但支付宝提供的就近推荐/支付服务相对较新，数据较少。其次，
推荐系统会受到一组预算限制，例如，由于服务能力或商店可用的优惠券数量。据我们所知，这种竞赛设置对研究界来说是新奇的，
尽管它对于蓬勃发展的基于位置的业务至关重要。

在本次比赛中，我们的目标是根据用户在 2015 年 7 月 1 日至 2015 年 11 月 30 日之间的在线/现场行为（表 1、2）预测用户在
 2015 年 12 月的偏好（表 4） .此外，每个商家都有预算限制（表 3），模拟可用的有限折扣/优惠券。
该数据集涉及以下在天猫/淘宝网和支付宝应用上积累的数据。
备注1：出于业务和噪音的考虑，我们在大促销期间删除了数据。也就是说，11 月 1 日至 11 月。表 1 中的 20 和表 4 中的 12 月 12。
备注 2：数据是从每日日志中抽取的有偏差的，因此其分布将与我们整个业务的分布不同。尽管如此，我们相信它不会对用户的偏好预测产生太大影响。 
"""

# 表 1：2015 年 12 月之前的在线用户行为。（ijcai2016_taobao）
"""
User_id:唯一用户id
Seller_id:唯一卖家id
Item_id:唯一商品id
Category_id:唯一类别 ID , 无任何作用
Online_Action_id:“0”表示“点击”，“1”表示“购买” 
Time_Stamp:日期
"""
df1 = pd.read_csv('ijcai2016_taobao.csv',header=1,usecols=['use_ID','sel_ID','ite_ID','cat_ID','act_ID','time'])


# 2015 年 12 月之前用户在实体店的购物记录。 (ijcai2016_koubei_train)
"""
User_id:唯一用户id
Merchant_id：唯一卖家id
Location_id：唯一位置id
Time_Stamp:日期
"""

df2 = pd.read_csv('ijcai2016_koubei_train.csv')


#  表 3：商家信息。 (ijcai2016_merchant_info)
"""
Merchant_id:唯一商户id
Budget:强加于商家的预算限制
Location_id_list:可用位置列表，例如 1:356:89 
"""

df3 = pd.read_csv('ijcai2016_merchant_info.csv')


#  表 4：预测结果。 (ijcai2016_koubei_test)
"""
User_id:唯一用户id
Location_id:唯一位置id
Merchant_id_list:您最多可以在这里推荐 10 个商家，以“:”分隔，例如 1:5:69 
"""
df4 = pd.read_csv('ijcai2016_koubei_test.csv')



df1.use_ID = df1.use_ID.astype(str)
df1.ite_ID = df1.ite_ID.astype(str)
df1.act_ID = df1.act_ID.astype(str)
df1.sel_ID = df1.sel_ID.astype(str)
df1 = df1.rename(columns={'time\n':'time'})

df2.use_ID = df2.use_ID.astype(str)
df2.mer_ID = df2.mer_ID.astype(str)
df2.loc_ID = df2.loc_ID.astype(str)
df2 = df2.rename(columns={'time\n':'time'})

df3 = df3.rename(columns={'loc_list\n':'loc_list'})
df3.mer_ID = df3.mer_ID.astype(str)
df3.loc_list = df3.loc_list.astype(str)

df4 = df4.rename(columns={'loc_ID\n':'loc_ID'})
df4.use_ID = df4.use_ID.astype(str)
df4.loc_ID = df4.loc_ID.astype(str)

# 将df1中的'use_ID', 'mer_ID'与df2中的mer_ID': 'sel_ID拼接
df = df2[['use_ID', 'mer_ID']].rename(columns={'mer_ID': 'sel_ID'})
df['act_ID'] = 1
df = pd.concat([df1[['use_ID', 'sel_ID', 'act_ID']], df], axis=0)

# convert_to_num = pickle.load(open('./utils/convert_to_num.pkl', 'rb'))

records = pickle.load(open('./utils/records6.pkl', 'rb'))

budget = pickle.load(open('./utils/budget.pkl', 'rb'))

location = pickle.load(open('./utils/mer_location.pkl', 'rb'))

batch_size = 100

# 使用GPU计算
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:0")

# 循环神经网络
HIDDEN_SIZE = 32
NUM_LAYER = 2
BIDIRECTIONAL = False
BITCH_FIRST = True
DROPOUT = 0.3

EMBEDDING_DIM = 100