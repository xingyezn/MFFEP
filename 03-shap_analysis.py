import pandas as pd
from tqdm import tqdm
import numpy as np
import string
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error
from torch.utils import data
from sklearn.model_selection import train_test_split
import logging
import datetime
import os
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import r2_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
import shap
from IPython.display import display, Javascript
import matplotlib.pyplot as plt

def performance(ground_truth, prediction):
    auc = metrics.roc_auc_score(ground_truth, prediction) * 100
    prediction = np.round(prediction)
    f1 = metrics.f1_score(ground_truth, prediction) * 100
    recall = metrics.recall_score(ground_truth, prediction) * 100
    precision = metrics.precision_score(ground_truth, prediction) * 100
    acc = metrics.accuracy_score(ground_truth, prediction) * 100
    rmse = np.sqrt(mean_squared_error(ground_truth,prediction))
    one_re_t = [auc, f1, recall, precision, acc,rmse]
    re_col = ['Auc', 'F1', 'Recall', 'Precision', 'Acc','Rmse']
    one_re_dict = dict(zip(re_col,one_re_t))
    return one_re_dict

class Shap_LSTM_DataSet(Dataset):
    def __init__(self,all_stu_data,week_need):
        self.all_stu_data = all_stu_data
        self.week_need = week_need
        
    def __len__(self):
        return len(self.all_stu_data)

    def __getitem__(self, index):
        one_data = self.all_stu_data[index]
        x = torch.FloatTensor(one_data['week_data'][:self.week_need])
        comm_data = torch.FloatTensor(one_data['common_data'])
        y = torch.as_tensor(one_data['final_result'], dtype=torch.float)
        inputs = torch.cat((x.view(-1), comm_data), dim=0)
        return inputs,y

class Shap_LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim,is_info,info_len,model_name):
        super(Shap_LSTMModel, self).__init__()        
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.layer_dim = layer_dim
        self.is_info = is_info
        self.info_len = info_len
        self.model_name = model_name
        predict_dim = hidden_dim
        if model_name == 'LSTM':
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True) 
        elif model_name == 'BiLSTM':
            predict_dim += hidden_dim
            self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True) 
        elif model_name == 'RNN':
            self.lstm = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True)
        if is_info:
            predict_dim += hidden_dim
            self.info_layer = nn.Sequential(nn.Linear(self.info_len, hidden_dim),
                                             nn.ReLU(True),
                                             nn.Linear(hidden_dim, hidden_dim))     
        self.class_classifier = nn.Sequential(nn.Linear(predict_dim, predict_dim),
                                             nn.ReLU(True),
                                             nn.Linear(predict_dim, 1),
                                             nn.Sigmoid())
    def forward(self, inputs):
        batch_size = int(inputs.size(0))
        common_data = inputs[:,-self.info_len:]
        reshaped_part = inputs[:, :-self.info_len]
        week = int(reshaped_part.size(1)/self.input_dim)
        x = reshaped_part.reshape(batch_size, week, self.input_dim)
        if self.model_name == 'LSTM':
            hidden_len = self.layer_dim
            h0 = torch.zeros(hidden_len, x.size(0), self.hidden_dim).to(DEVICE)
            c0 = torch.zeros(hidden_len, x.size(0), self.hidden_dim).to(DEVICE)
            out, (hn, cn) = self.lstm(x, (h0, c0))
        elif self.model_name == 'BiLSTM':
            hidden_len = self.layer_dim *2
            h0 = torch.zeros(hidden_len, x.size(0), self.hidden_dim).to(DEVICE)
            c0 = torch.zeros(hidden_len, x.size(0), self.hidden_dim).to(DEVICE)
            out, (hn, cn) = self.lstm(x, (h0, c0))
        elif self.model_name == 'RNN':
            hidden_len = self.layer_dim
            h0 = torch.zeros(hidden_len, x.size(0), self.hidden_dim).to(DEVICE)
            out, _ = self.lstm(x, h0)
        out = out[:, -1, :]
        if self.is_info:
            out = torch.cat([out, self.info_layer(common_data)], dim=1)
        out = self.class_classifier(out)
        return out

from sklearn.preprocessing import StandardScaler
def get_stand_data(one_data_set,use_info):
    scaler = StandardScaler()
    need_data = one_data_set[use_info]
    df_standardized = pd.DataFrame(scaler.fit_transform(need_data), columns=use_info)
    df_standardized['stu_id_index'] = list(one_data_set['stu_id_index'])
    df_standardized.index = list(df_standardized['stu_id_index'])
    return df_standardized

def get_time_data(need_stu_id,data_path,step_num,stu_info_set,stu_info,act_type,is_front,is_stand,is_merge,count_type):
    all_week_data = []
    for one_week in range(1,step_num+1):
        one_data = pd.read_pickle(f'{data_path}/{one_week}.pkl')
        if is_stand:
            one_data = get_stand_data(one_data,act_type)
        all_week_data.append(one_data)
    if is_merge:
        a_conut = 'count' if count_type == 'tf' else 'tf'
        new_path = data_path.replace(count_type,a_conut)
        all_week_data_a = []
        for one_week in range(1,step_num+1):
            one_data = pd.read_pickle(f'{new_path}/{one_week}.pkl')
            if is_stand:
                one_data = get_stand_data(one_data,act_type)
            all_week_data_a.append(one_data)
    all_stu_data = []
    for one_stu in tqdm(need_stu_id,desc='read_data'):
        one_stu_data = {'final_result':stu_info_set.loc[one_stu]['final_result'],'week_data':[],'stu_id':one_stu}
        one_stu_data['common_data'] = list(stu_info_set.loc[one_stu][stu_info])
        for one_week in range(step_num):
            one_week_data = list(all_week_data[one_week].loc[one_stu][act_type])
            if is_merge:
                one_week_data.extend(list(all_week_data_a[one_week].loc[one_stu][act_type]))
            if is_front:
                one_week_data.extend(one_stu_data['common_data'].copy())
            one_stu_data['week_data'].append(one_week_data)
        all_stu_data.append(one_stu_data)
    return all_stu_data

StudentInfo=pd.read_pickle('data/studentInfo.pkl')
StudentInfo.index = list(StudentInfo['stu_id_index'])
all_stu_id = list(StudentInfo['stu_id_index'])
need_stu_id = list(StudentInfo[StudentInfo['final_result']==0]['stu_id_index'])
fail_len = len(need_stu_id)
pass_data = list(StudentInfo[StudentInfo['final_result']==1].sample(fail_len,random_state=10,replace=False)['stu_id_index'])
need_stu_id.extend(pass_data)
StudentInfo = StudentInfo.loc[need_stu_id]
StudentInfo.head()


import copy
ALL_ACT_TYPE = ['forumng', 'oucontent', 'subpage', 'homepage', 'quiz', 'resource',
       'url', 'ouwiki', 'oucollaborate', 'externalquiz', 'questionnaire',
       'page', 'ouelluminate', 'glossary', 'dataplus', 'dualpane', 'folder',
       'htmlactivity', 'sharedsubpage', 'repeatactivity']
STU_INFO = ['course_type','gender', 'highest_education_num', 'disability_num', 'age_band_num','num_of_prev_attempts']

EPOCH_NUM = 10
BATCH_SIZE = 64
DEVICE = 'cpu'
STEP_DAY = 7 
STEP_NUM = 10
IS_COUNT = 1 
IS_ADD = 0 
USE_INFO = 1 
IS_MERGE = 0 
IS_FRONT = 0 
IS_STAND = 0 
COUNT_TYPE = 'count' if IS_COUNT else 'tf'
ADD_TYPE = 'add'if IS_ADD else 'single'
ALL_ML_MODEL = ['RFC','KNN','DT','GNB','SVM','GBT','LR']
DEEP_MODEL = ['BiLSTM','LSTM','RNN']
MODEL_NAME = 'RNN'
config = {
    'Model': MODEL_NAME,
    'Add_type': ADD_TYPE,
    'Count_type': COUNT_TYPE,
    'USE_INFO': USE_INFO,
    'BATCH_SIZE': BATCH_SIZE,
    'IS_FRONT': IS_FRONT,
    'IS_STAND': IS_STAND,
    'IS_MERGE': IS_MERGE
}
logging.shutdown()
time_now = datetime.datetime.now()
now_str = time_now.strftime('%Y-%m-%d-%H-%M-%S')
log_dir = f'log/{STEP_DAY}/{MODEL_NAME}/{COUNT_TYPE}'
log_file_name = f'{log_dir}/{ADD_TYPE}-{now_str}.log'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(message)s')
file_logger = logging.getLogger(log_file_name)
file_handler = logging.FileHandler(log_file_name, mode='a', encoding='utf-8')
file_handler.setFormatter(formatter)
file_logger.addHandler(file_handler)
file_logger.info(f"{config}")
TRAIN_DATA, TEST_DATA = train_test_split(StudentInfo,test_size=0.2, random_state=42, stratify=StudentInfo['final_result'])
TRAIN_DATA = list(TRAIN_DATA['stu_id_index'])
print('TRAIN_DATA len',len(TRAIN_DATA))
TEST_DATA = list(TEST_DATA['stu_id_index'])
print('TEST_DATA len',len(TEST_DATA))
ALL_STEP_RE = []
for ONE_STEP in [10]: 
    DATA_PATH = f'data/{STEP_DAY}/{ADD_TYPE}/{COUNT_TYPE}'
    if MODEL_NAME in ALL_ML_MODEL:
        one_data = pd.read_pickle(f'{DATA_PATH}/{ONE_STEP}.pkl')
        ML_data = pd.concat([StudentInfo,one_data],axis=1)
        ML_col = ALL_ACT_TYPE.copy()
        ML_col.extend(STU_INFO)
        X_train,y_train = ML_data.loc[TRAIN_DATA][ML_col].values,ML_data.loc[TRAIN_DATA]['final_result']
        X_test,y_test = ML_data.loc[TEST_DATA][ML_col].values,ML_data.loc[TEST_DATA]['final_result']
        if MODEL_NAME=='RFC':
            model = RandomForestClassifier(n_estimators=40,max_depth=18,random_state=10)
        if MODEL_NAME=='KNN':
            model = KNeighborsClassifier()
        if MODEL_NAME=='DT':
            model = DecisionTreeClassifier(max_depth=7,random_state=2023)
        if MODEL_NAME=='GNB':
            model =  GaussianNB()
        if MODEL_NAME=='SVM':
            model = SVC(kernel='rbf')
        if MODEL_NAME=='GBT':
            model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        if MODEL_NAME=='LR':
            model = LogisticRegression(max_iter=50)
        model = model.fit(X_train,y_train)
        one_re_dict = performance(y_train,model.predict(X_train))
        file_logger.info(f"STEP {ONE_STEP} Train: {one_re_dict}")
        best_one_re_dict = performance(y_test,model.predict(X_test))
        file_logger.info(f"STEP {ONE_STEP} Test: {best_one_re_dict}")
    if MODEL_NAME in ['LSTM','BiLSTM','RNN']:
        file_logger.info(f'Start Week :{ONE_STEP}')
        data_params = {
            'data_path': DATA_PATH,
            'step_num': ONE_STEP,
            'stu_info_set': StudentInfo,
            'stu_info': STU_INFO,
            'act_type': ALL_ACT_TYPE,
            'is_front': IS_FRONT,
            'is_stand': IS_STAND,
            'is_merge': IS_MERGE,
            'count_type': COUNT_TYPE
        }
        train_data = get_time_data(TRAIN_DATA,**data_params)
        train_data = Shap_LSTM_DataSet(train_data,ONE_STEP)
        test_data = get_time_data(TEST_DATA,**data_params)
        test_data = Shap_LSTM_DataSet(test_data,ONE_STEP)
        train_laoder = data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
        test_laoder = data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
        input_dim = len(ALL_ACT_TYPE)
        if IS_FRONT:
            input_dim += len(STU_INFO)
        if IS_MERGE:
            input_dim += len(ALL_ACT_TYPE)
        hidden_dim = 64
        layer_dim = 2 
        output_dim = 1 
        info_len = len(STU_INFO)
        LR = 0.0001
        model = Shap_LSTMModel(input_dim, hidden_dim, layer_dim, output_dim,USE_INFO,info_len,MODEL_NAME)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
        file_logger.info(f'input_dim:{input_dim}, hidden_dim:{hidden_dim}, layer_dim:{hidden_dim}, output_dim:{output_dim},info_len:{info_len}')
        model.to(DEVICE)
        best_auc = 0
        best_one_re_dict = []
        best_ecoch = 0
        best_model = copy.deepcopy(model)
        for one_epoch in range(EPOCH_NUM):
            model.train()
            ground_truth = torch.Tensor([])
            prediction = torch.Tensor([])
            for inputs, labels in tqdm(train_laoder, desc=f'Week {ONE_STEP} Train-{one_epoch}:',):
                inputs.to(DEVICE)
                labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(inputs)
                outputs = outputs.squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                prediction = torch.cat([prediction, outputs.to('cpu')])
                ground_truth = torch.cat([ground_truth, labels.to('cpu')])
            one_re_dict = performance(ground_truth.detach().numpy(), prediction.detach().numpy())
            file_logger.info(f'Week {ONE_STEP} Train-{one_epoch} Preformance:{one_re_dict}')
            model.eval()
            ground_truth = torch.Tensor([])
            prediction = torch.Tensor([])
            for inputs, labels in tqdm(test_laoder, desc=f'Week {ONE_STEP} Test-{one_epoch}::',):
                inputs.to(DEVICE)
                with torch.no_grad():
                    outputs = model(inputs)
                    outputs = outputs.squeeze()
                    prediction = torch.cat([prediction, outputs.to('cpu')])
                    ground_truth = torch.cat([ground_truth, labels.to('cpu')])
            one_re_dict = performance(ground_truth.detach().numpy(), prediction.detach().numpy())
            file_logger.info(f'Week {ONE_STEP}  Test-{one_epoch}Preformance:{one_re_dict}')
            if one_re_dict['Auc'] > best_auc:
                best_auc = one_re_dict['Auc']
                best_ecoch = one_epoch
                file_logger.info(f'Best_acu in Valid:【{best_auc}】 at epoch 【{one_epoch}】')
                best_one_re_dict = one_re_dict.copy()
                best_model = copy.deepcopy(model)
                torch.save(best_model, 'best_model.pth')
            if one_epoch-best_ecoch>10:
                print('Train stop')
                break
    best_one_re_dict['STEP'] = ONE_STEP
    file_logger.info(f'Test【{ONE_STEP}】:{best_one_re_dict}')
    ALL_STEP_RE.append(best_one_re_dict)
    del model
file_logger.info(f"{config}")
ALL_STEP_RE_set = pd.DataFrame(ALL_STEP_RE)
print(ALL_STEP_RE_set)
ALL_STEP_RE_set.to_csv(f'{log_dir}/{MODEL_NAME}-{ADD_TYPE}-{now_str}.csv')
logging.shutdown()


ALL_stu_id = StudentInfo['stu_id_index']
test_data_shap = get_time_data(ALL_stu_id,**data_params)
test_data_shap = Shap_LSTM_DataSet(test_data_shap,ONE_STEP)
test_laoder_shap = data.DataLoader(test_data_shap, batch_size=8, shuffle=False)
need_data = torch.Tensor([])
true_lable = torch.Tensor([])
for inputs, labels in test_laoder_shap:
    need_data = torch.concat([need_data,inputs])
    true_lable = torch.concat([true_lable,labels])
need_data = [need_data]
best_model.eval()
explainer = shap.GradientExplainer(best_model,data=need_data)
shap_values = explainer.shap_values(need_data)


best_model.eval()
with torch.no_grad():
    predictions = best_model(need_data[0])
    expected_value = predictions.mean().item()
    

ALL_ACT_TYPE = ['forumng', 'oucontent', 'subpage', 'homepage', 'quiz', 'resource',
       'url', 'ouwiki', 'oucollaborate', 'externalquiz', 'questionnaire',
       'page', 'ouelluminate', 'glossary', 'dataplus', 'dualpane', 'folder',
       'htmlactivity', 'sharedsubpage', 'repeatactivity']
STU_INFO = ['*course type*','*gender*', '*highest education*', '*disability*', '*age*','*prev attempts*']

def get_one_week_shap(shap_values,need_data,week_num):
    one_week_len = len(ALL_ACT_TYPE)
    week_shap = shap_values[:,one_week_len*(week_num-1):one_week_len*week_num]
    info_shap = shap_values[:,-len(STU_INFO):]
    combined_shap = np.concatenate((week_shap, info_shap), axis=1)
    week_shap_2 = need_data[:,one_week_len*(week_num-1):one_week_len*week_num]
    info_shap_2 = need_data[:,-len(STU_INFO):]
    combined_shap_2 = np.concatenate((week_shap_2, info_shap_2), axis=1)
    return (combined_shap,combined_shap_2)

week_shap_val, week_data = get_one_week_shap(shap_values,need_data[0],10)
week_feature = ALL_ACT_TYPE.copy()
week_feature.extend(STU_INFO)
shap.summary_plot(week_shap_val, week_data, week_feature,max_display=10,show=False)
plt.savefig('ten_week.png', dpi=300, bbox_inches='tight')
plt.show()


shap.initjs()
stu_id = 10015
one_stu_data = [dict(zip(week_feature,list(np.array(week_data[stu_id]))))]
one_stu_data = pd.DataFrame(one_stu_data)
shap.force_plot(base_value=0.49,
    shap_values=week_shap_val[stu_id], 
    features=one_stu_data.loc[0], 
    feature_names=week_feature, 
    matplotlib=True,
    show=False,out_names='Predict Value',figsize=(20, 3))
plt.savefig(f'student_{stu_id}.png', dpi=300, bbox_inches='tight')
plt.show()