import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import csv

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=12)
fig = plt.figure(figsize=(5.5, 4.5), dpi=100)

# 1. 데이터 준비
# read csv file in a list
cam_list = [1,2,3,4,5]
model_list = ['rbf_svr', 'linear_reg', 'sgd_reg', 'voting_reg', 'rf_reg', 'mlp_reg', 'knn_reg', 'dt_reg', 'ada_reg', 'bag_reg']
model_list = ['rbf_svr', 'linear_reg', 'knn_reg', 'dt_reg', 'ada_reg', 'voting_reg']

model_name_for_print = {
    'rbf_svr': 'SVR',
    'linear_reg': 'Linear',
    'knn_reg': 'KNN',
    'dt_reg': 'DT',
    'ada_reg': 'AdaBoost',
    'voting_reg': 'Voting',
    'sgd_reg': 'SGD',
    'rf_reg': 'Random Forest',
    'mlp_reg': 'MLP',
    'bag_reg': 'Bagging'
}

my_colors = ['#4285f4', '#ea4335', '#fbbc04', '#34a853', '#ff6d01', '#46bdc6', '#7baaf7', '#f07b72', '#fcd04f', '#71c287', '#ff994d', '#7ed1d7', '#b3cefb', '#f7b4ae']

csv_dir = './pred_vmaf/trained_models/'

dfs = {}

for model in model_list:
    each_data = []
    for cam in cam_list:
        cam_str = str(cam)
        for cam2 in cam_list:
            cam2_str = str(cam2)
            if cam2 != cam:
               continue
            csv_file = f'{csv_dir}{cam_str}/prediction_error_{cam2_str}_{model}_csvmodule.csv'
            with open(csv_file, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    each_data.append(float(row[0]))  # 첫 번째 열의 값을 float로 변환하여 리스트에 추가

    total_dp = len(each_data)
    df = pd.DataFrame({
        'Model': [model_name_for_print[model]] * total_dp,   # 이름 5개 생성
        'Score': each_data           # 값 5개 매핑
    })
    dfs[model] = df

# draw boxplot for each model
all_df = pd.concat(dfs.values(), ignore_index=True)
sns.boxplot(x='Model', y='Score', data=all_df, width=0.6, palette=my_colors, linewidth=1.5, linecolor='black')

# plt.title(f"Model Performance Comparison", fontsize=16)
plt.ylabel("Error")
plt.ylim(0, 55)
plt.grid(axis='y', linestyle='--', alpha=0.5) # y축
plt.show()
fig.savefig('cv_models.pdf', bbox_inches='tight')
exit()



csv_file = './pred_vmaf/trained_models/1/prediction_error_1_dt_reg_csvmodule.csv'  # CSV 파일 경로
scores = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        scores.append(float(row[0]))  # 첫 번째 열의 값을 float로 변환하여 리스트에 추가

total_dp = len(scores)

# 데이터프레임으로 변환 (Seaborn은 DataFrame을 좋아합니다)
df = pd.DataFrame({
    'Model': ['dtreg'] * total_dp,   # 이름 5개 생성
    'Score': scores           # 값 5개 매핑
})

# 2. 그래프 그리기
plt.figure(figsize=(6, 7))  # 그래프 크기 (가로, 세로)

# (1) Box Plot 그리기 (전체 분포)
# width: 박스 가로 폭 (디자인)
sns.boxplot(x='Model', y='Score', data=df, width=0.4, color='skyblue')

# (2) Strip Plot 그리기 (실제 데이터 점)
# jitter=False: 점을 일렬로 정렬 (점 개수가 적을 때 보기 좋음)
# size: 점 크기, color: 점 색상 (빨간색으로 강조)
# sns.stripplot(x='Model', y='Score', data=df, color='red', size=8, jitter=False)
# visualize outliers clearly
sns.stripplot(x='Model', y='Score', data=df[df['Score'] > df['Score'].quantile(0.75) + 1.5 * (df['Score'].quantile(0.75) - df['Score'].quantile(0.25))], color='darkred', size=8, jitter=False)

# 3. 통계 수치 계산 (제목에 표시하기 위함)
mean_score = df['Score'].mean()
std_score = df['Score'].std()

# 4. 꾸미기
plt.title(f"[dtreg] Performance\nMean: {mean_score:.2f} ± {std_score:.2f}", fontsize=14)
plt.ylabel("Score")
plt.grid(axis='y', linestyle='--', alpha=0.5) # y축 눈금선 추가

plt.show()
