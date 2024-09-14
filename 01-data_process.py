import pandas as pd
from tqdm import tqdm
import numpy as np
import string
import os
import math

studentVle = pd.read_csv('data/OULAD_Dataset/studentVle.csv')
studentInfo = pd.read_csv('data/OULAD_Dataset/studentInfo.csv')
vle = pd.read_csv('data/OULAD_Dataset/vle.csv')
studentVle = studentVle[studentVle['date']>-1]
course_type = {'AAA':1,'BBB':1,'CCC':2,'DDD':2,'EEE':2,'FFF':2,'GGG':1}
studentInfo['course_type'] = studentInfo['code_module'].map(course_type)
studentInfo['final_result'] = studentInfo['final_result'].replace('Distinction','Pass')
studentInfo = studentInfo[studentInfo['final_result'].isin(['Pass','Fail'])]
studentInfo['final_result'] = studentInfo['final_result'].map({'Pass':1,'Fail':0})
stie_2_type = dict(zip(vle['id_site'],vle['activity_type']))
studentVle['activity_type'] = studentVle['id_site'].map(stie_2_type)
all_act_type = list(studentVle['activity_type'].value_counts().index)
studentInfo['stu_id'] = studentInfo['code_module'] + studentInfo['code_presentation']+studentInfo['id_student'].astype(str)
studentVle['stu_id'] = studentVle['code_module'] + studentVle['code_presentation']+studentVle['id_student'].astype(str)
vle_stu = list(set(list(studentVle['stu_id'])))
need_stu = []
for one in list(studentInfo['stu_id']):
    if one in vle_stu:
        need_stu.append(one)
studentVle = studentVle[studentVle['stu_id'].isin(need_stu)]
studentInfo = studentInfo[studentInfo['stu_id'].isin(need_stu)]

stu_id_map = dict(zip(list(studentInfo['stu_id']),range(len(list(studentInfo['stu_id'])))))
studentInfo['stu_id_index'] = studentInfo['stu_id'].map(stu_id_map)
studentVle['stu_id_index'] = studentVle['stu_id'].map(stu_id_map)
studentVle.index = list(studentVle['stu_id_index'])
studentInfo.index = list(studentInfo['stu_id_index'])
studentVle = studentVle[['date','sum_click','activity_type','stu_id','stu_id_index']]

studentInfo['gender'] = studentInfo['gender'].map({'F':0,'M':1})
studentInfo['disability_num'] = studentInfo['disability'].map({'N':0,'Y':1})
studentInfo['age_band_num'] = studentInfo['age_band'].map({'0-35':1, '35-55':2, '55<=':3})
highest_education_num = {'No Formal quals':1,'Lower Than A Level':2,'A Level or Equivalent':3,  'HE Qualification':4,'Post Graduate Qualification':5 }
studentInfo['highest_education_num'] = studentInfo['highest_education'].map(highest_education_num)
studentInfo = studentInfo[['course_type','gender','highest_education_num','disability_num','age_band_num','num_of_prev_attempts','final_result','stu_id','stu_id_index']]
studentInfo.head()

ALL_ACT_TYPE = ['forumng', 'oucontent', 'subpage', 'homepage', 'quiz', 'resource',
       'url', 'ouwiki', 'oucollaborate', 'externalquiz', 'questionnaire',
       'page', 'ouelluminate', 'glossary', 'dataplus', 'dualpane', 'folder',
       'htmlactivity', 'sharedsubpage', 'repeatactivity']
       
STEP_DAY = 7
STEP_NUM = max(studentVle['date'])//STEP_DAY
print(STEP_NUM)
ALL_STU_ID = list(studentInfo['stu_id_index'])


def get_count_data(one_set):
    all_stu_data = []
    one_set.index = list(one_set['stu_id_index'])
    one_set_stu = one_set['stu_id_index'].value_counts()
    one_stu_data_0 = dict(zip(ALL_ACT_TYPE,[0]*len(ALL_ACT_TYPE)))
    for one_stu in ALL_STU_ID:
        one_stu_data = one_stu_data_0.copy()
        if one_stu in one_set_stu.index:
            one_stu_set = one_set.loc[one_stu]
            if one_set_stu[one_stu] ==1:
                one_stu_data[one_stu_set['activity_type']] = one_stu_set['sum_click']
            else:
                one_stu_data.update(dict(one_stu_set.groupby('activity_type').sum()['sum_click']))
        all_stu_data.append(one_stu_data)
    one_re = pd.DataFrame(all_stu_data,index=ALL_STU_ID)
    return one_re

def get_tf_idf(count_data):
    all_stu_len = sum(count_data.sum(axis=1)!=0)
    all_idf = {}
    for one_act_type in ALL_ACT_TYPE:
        one_num = sum(count_data[one_act_type]!=0)
        all_idf[one_act_type] = math.log(all_stu_len/(one_num+1),10)
    all_stu_data = []
    one_tf_0 = dict(zip(ALL_ACT_TYPE,[0]*len(ALL_ACT_TYPE)))
    for one_stu in ALL_STU_ID:
        one_stu_data = count_data.loc[one_stu]
        one_sum = sum(one_stu_data)
        if one_sum!=0:
            one_tf = dict(one_stu_data/one_sum) 
        else:
            one_tf = one_tf_0.copy() 
        one_tf_idf = [one_tf[one_a]*all_idf[one_a] for one_a in ALL_ACT_TYPE]
        one_tf_idf = dict(zip(ALL_ACT_TYPE,one_tf_idf))
        all_stu_data.append(one_tf_idf)
    one_re = pd.DataFrame(all_stu_data,index=ALL_STU_ID)
    return one_re

def save_data(stpe_day,is_single,is_count,step_num,one_data_set):
    data_path = f'data/{STEP_DAY}/{is_single}/{is_count}'
    os.makedirs(data_path, exist_ok=True)
    flie_path = f'data/{STEP_DAY}/{is_single}/{is_count}/{one_step}.pkl'
    data_set = one_data_set.copy()
    data_set['stu_id_index'] = ALL_STU_ID
    data_set.to_pickle(flie_path)

one_step_conut_add = []
for one_step in tqdm(range(1,STEP_NUM+1)):
    one_step_set = studentVle[(studentVle['date']>one_step*(STEP_DAY-1)) & (studentVle['date']<=one_step*STEP_DAY)].copy()
    one_step_count_single = get_count_data(one_step_set)
    if one_step == 1:
        one_step_conut_add = one_step_count_single.copy()
    else:
        one_step_conut_add = one_step_conut_add+one_step_count_single
    one_step_tf_single = get_tf_idf(one_step_count_single)
    one_step_tf_add = get_tf_idf(one_step_conut_add)
    save_data(STEP_DAY,'single','count',one_step,one_step_count_single)
    save_data(STEP_DAY,'add','count',one_step,one_step_conut_add)
    save_data(STEP_DAY,'single','tf',one_step,one_step_tf_single)
    save_data(STEP_DAY,'add','tf',one_step,one_step_tf_add)