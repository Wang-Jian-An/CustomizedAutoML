# 導入所需的庫
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from AutoML_Flow.tabular_LIME import tabular_LIME

# 加載鳶尾花資料集
iris = load_iris()
data = pd.DataFrame(
    iris.data,
    columns = iris.feature_names
)
data["Target"] = iris.target.astype("int")

# 分割資料集為訓練集和測試集
train, test = train_test_split(data, test_size=0.3, random_state=42)

# 創建隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型
clf.fit(train[iris.feature_names], train["Target"])

# 進行預測
y_pred = clf.predict(test[iris.feature_names])

# 評估模型
accuracy = accuracy_score(test["Target"], y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

lime_obj = tabular_LIME(
    original_model  = clf,
    target_label = iris.target_names, 
    features_list = [iris.feature_names]
)
lime_obj.fit(
    explain_instance = pd.Series(test.iloc[0, :])
)
print(lime_obj.output_explanation_model_weight())
ax = lime_obj.draw_forest_plot()
plt.savefig("sample_plot.png")