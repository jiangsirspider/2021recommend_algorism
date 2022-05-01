"""
数据情况
# 表 1：2015 年 12 月之前的在线用户行为。（ijcai2016_taobao） 
User_id:唯一用户id
Seller_id:唯一卖家id
Item_id:唯一商品id
Category_id:唯一类别 ID 
Online_Action_id:“0”表示“点击”，“1”表示“购买” 
Time_Stamp:日期
# 表2：2015 年 12 月之前用户在实体店的购物记录。 (ijcai2016_koubei_train)
User_id:唯一用户id
Merchant_id：唯一卖家id
Location_id：唯一位置id
Time_Stamp:日期
#  表 3：商家信息。 (ijcai2016_merchant_info)
Merchant_id: 唯一商户id
Budget: 强加于商家的预算限制
Location_id_list: 可用位置列表，例如 1:356:89
# 表 4：预测结果。 (ijcai2016_koubei_test)
User_id: 唯一用户id
Location_id: 唯一位置id
Merchant_id_list: 您最多可以在这里推荐 10 个商家，以:分隔，例如 1:5:69


数据发现
1.用户更愿意在购物人数多的商家购买，对店铺人气分级 1-4；分级依据四分位数
2.商家有预算限制，一个商家推荐给用户数不应该超过这个限制。商家有预算为1，没有预算为-1


采用逻辑回归进行推荐
1.根据用户位置召回附近的商家
2.对商家进行打分，推荐十个得分最高的商家

"""

import pickle
# 商家信息表（商家id-商家地理位置-商家预算）
# 注意一个商家可能对应多个位置
# 这个表用于根据地理位置查询所有商家：一个位置下的所有商家; 预算在此并无作用
# 根据用户的地理位置召回附近的商店
mer_location = pickle.load(open('./utils/mer_location.pkl', 'rb'))
print(mer_location.head())

# 商家预算表(商家id-商家预算-店铺人气-店铺人气的评级)
# 商家id唯一
# 这个表用于更新商家预算，推荐一次，商家预算减少1，当商家预算为0时候则不应该推荐给用户
budget = pickle.load(open('./utils/budget.pkl', 'rb'))
print(budget.head())
print(budget.mer_ID.is_unique)  # True

# 综合信息表-数据发现
# 对用户线上线下购物行为按时间进排序，构成 act_ID（0为浏览，1为购买）
# target 取值为用户最近一次购买行为,(act_ID最后一个值)
records = pickle.load(open('./utils/records6.pkl', 'rb'))
print(records.head())
#   use_ID	sel_ID	act_ID	len_act	cut_act	target	star	star_level	act_level	mer_ID	budget	bd_level
# 0	1000004	4876	11111	5	    111	    1	    919.0	2	        3	        4876	100.0	1
# 1	1000005	2606	11111	5	    111	    1	    22988.0	3	        3	        2606	8968.0	1
# 2	100002	4264	1111111	7	    11111	1	    45554.0	3	        3	        4264	6356.0	1
# 3	100002	4796	1	    1		        1	    1226.0	2	        1	        4796	3281.0	1
# ....
# 6	100051	1757	010	    3	    0	    0	    156.0	2	        2	        1757	100.0	1

# 扩展数据集 len(records)*2
# 为了让模型识别到当商家预算为0时，不推荐该商家
# 将records中的budget修正为0，bd_level修正为-1，target修正为0；append到原来的records中得到新的records

# 到此得到用于逻辑回归的特征值（star_level，bd_level）和目标值（target）
# 模型可以很好识别，当bd_level=-1时候，target趋近0，
# 当bd_level=1时候，target取值随着star_level提高而提高

# 思路
# 测试采用df2
# 取其中[use_ID	mer_ID	loc_ID]
# 预测时加入[use_ID	mer_ID	loc_ID recomend_ls] 如果mer_id在recomend_ls中，则为1，匹配成功


