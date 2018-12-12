from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import pandas as pd
import pickle
# import matplotlib.pyplot as plt

# Train on the whole dataset
# Test on the vallidation part
data = pd.read_csv('../consolidated.csv')
features = data.drop(['articleId','hyperpartisan','orientation','article_date'], axis=1)
truth = data['hyperpartisan']

# Use a data normalizer
scaler = preprocessing.Normalizer()

# train_X, test_X, train_y, test_y = train_test_split(features, truth, test_size=0.20)
train_X = scaler.fit_transform(features[:800000])
train_X = pd.DataFrame(train_X)
test_X = scaler.fit_transform(features[800000:])
test_X = pd.DataFrame(test_X)
train_y = truth[:800000]
test_y = truth[800000:]

# error_rate = []
# for i in range(1,50):
# 	model = KNeighborsClassifier(n_neighbors=i)
# 	model.fit(train_X,train_y)
# 	wee = model.predict(test_X)
# 	acc = accuracy_score(wee, test_y)
# 	error_rate.append(acc)
model = LinearSVC(tol=1e-1)
model.fit(train_X,train_y)
# wee = model.predict(test_X)
# print(accuracy_score(wee, test_y))
print(model.score(test_X, test_y))
with open('svm_model-linear.pkl', 'wb') as f:
	pickle.dump(model, f)
# plt.figure()
# plt.plot(range(1,50), error_rate)
# plt.title('accuracy_score vs knn input k')
# plt.xlabel('K')
# plt.ylabel('accuracy_score')
# plt.show()
