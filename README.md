# modify_data

##data_process
数据清洗方式
###初始化数据集
###格式化数据集
### 
##model
集合了大部分常用的分类算法类
###树模型

####决策树
`DecisionTreeClassifier(max_depth=8,random_state=0)`
####随机森林
`RandomForestClassifier(oob_score=False, random_state=0)`

###回归模型
`LogisticRegression(penalty = 'l2', tol = 0.0001, C = 1.0,solver = 'liblinear', max_iter = 100, multi_class = 'ovr',
                                 verbose = 0, warm_start = False, n_jobs = 1
                                 )`
###SVM模型
###贝叶斯模型
##log
重定向日志类
##plot
绘制类
###