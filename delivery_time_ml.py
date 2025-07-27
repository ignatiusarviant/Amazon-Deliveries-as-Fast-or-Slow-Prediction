from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, recall_score, auc, roc_curve
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

file = pd.read_csv("C:\\Users\\HP\\Downloads\\Python Only\\Delivery Time ML\\Data Training & Test.csv")
df = pd.DataFrame(file)

x_column = ['Agent_Age', 'Agent_Rating', 'Store_Latitude', 'Store_Longitude',
       'Drop_Latitude', 'Drop_Longitude', 'Weather', 'Traffic', 'Vehicle', 'Area',
       'Category', 'Distance_km']
y_column = 'fast_or_not'
x = df[x_column]
y = df[y_column]

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
model = RandomForestClassifier(n_estimators=300, random_state=42, oob_score=True, class_weight="balanced")
model.fit(x_train, y_train)
prediction = model.predict(x_test)

f1_score = f1_score(y_test, prediction)
accuracy_score = accuracy_score(y_true=y_test, y_pred=prediction)
roc_auc_score = roc_auc_score(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
recall_score = recall_score(y_true=y_test, y_pred=prediction)
fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=model.predict_proba(x_test)[:, 1])
auc = auc(x=fpr, y=tpr)
diff = tpr-fpr

new_df = pd.DataFrame({"FPR": fpr, "TPR": tpr, "Difference": diff})

sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#171515")
# sns.lineplot(data=new_df, x=fpr, y=tpr, color="#03FFA7", marker="o", linestyle='-')

ax.plot(fpr, tpr, color="#03FFA7", linewidth=2.5, alpha=0.9, label="Line Title", marker="o")
ax.tick_params(colors='white', which='both')
ax.set_facecolor("#171515")
ax.grid(color = "#4D4E4E", linestyle= "-", linewidth=0.5)
ax.xaxis.label.set_color('white')
ax.yaxis.label.set_color('white')
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_color('#777777')
ax.spines['bottom'].set_color('#777777')
ax.fill_between(fpr, tpr, color="#03FFA7", alpha=0.2)

# plt.plot(fpr, tpr, label="ROC line", marker="o", linestyle='-', alpha=0.7, color="#01FF0ED4")
plt.title("ROC Curve", color="white")
plt.xlabel("False Positive Rate")
plt.ylabel("Total Positive Rate")
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

stats_text = (f"F1 Score: {f1_score:.2f}  |  Accuracy Score: {accuracy_score:.2f} |  Recall Score: {recall_score:.2f}  |  OOB Score: {model.oob_score_:.2f} | AUC: {auc:.2f}")
plt.figtext(0.5, 0.05, stats_text, ha="center", fontsize=10, color="white",
            bbox={"facecolor":"whitesmoke", "alpha":0.5, "pad":5})

plt.subplots_adjust(bottom=0.2) 

print(new_df)
plt.show()