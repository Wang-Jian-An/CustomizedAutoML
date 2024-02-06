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
X = pd.DataFrame(
    iris.data,
    columns = iris.feature_names
)
y = iris.target

# 分割資料集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 創建隨機森林分類器
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# 訓練模型
clf.fit(X_train, y_train)

# 進行預測
y_pred = clf.predict(X_test)

# 評估模型
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

lime_obj = tabular_LIME(
    original_model  = clf
)
lime_obj.fit(
    explain_instance = pd.Series(X_test.iloc[0, :])
)
ax = lime_obj.draw_forest_plot()
plt.show()