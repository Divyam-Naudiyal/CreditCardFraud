import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA, TruncatedSVD

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objects as go
import plotly

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,recall_score,precision_recall_curve,average_precision_score
from sklearn.utils import resample
import csv

with open(r"C:\Users\lenovo\PycharmProjects\CreditCardFraud\creditcard.csv", 'r', errors='replace') as csvfile:
    reader = csv.reader(csvfile)
    try:
        datafr = pd.read_csv(csvfile)
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
        # For simple data display
        print(datafr)

        # For data visualization
        import matplotlib.pyplot as plt

        # Your data visualization code here
        plt.show()

print(datafr.shape)

hours = (datafr['Time']/3600).astype(int)
datafr['Hours'] = hours

days = (datafr['Time']/86400).astype(int)
datafr['Days'] = days

bins = [0,100,1000,5000,10000,20000, 30000]
labels = [1,2,3,4,5,6]
datafr['binned'] = pd.cut(datafr['Amount'], bins=bins, labels=labels)
datafr.head(10)

f, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
sns.boxplot(x="binned", y="Amount", hue="Class", data=datafr[datafr['Class']==0], palette='Blues', ax=axes[0])
axes[0].set_title('BoxPlot for {}'.format("Class 0: Not Fraudulent"))
sns.boxplot(x="binned", y="Amount", hue="Class", data=datafr[datafr['Class']==1], palette='Purples', ax=axes[1])
axes[1].set_title('BoxPlot for {}'.format("Class 1: Fraudulent"))

plt.figure(figsize=(14,6))
sns.set(style="darkgrid")
sns.countplot(x='binned',data = datafr, hue = 'Class',palette='BuPu')
plt.title("Count Plot of Transactions per each amount bin\n", fontsize=16)
sns.set_context("paper", font_scale=1.4)
plt.show()


plt.figure(figsize=(14,6))
sns.set(style="darkgrid")
sns.countplot(x='Hours',data = datafr, hue = 'Class',palette='BuPu')
plt.title("Count Plot of Transactions per each Hour\n", fontsize=16)
sns.set_context("paper", font_scale=1.4)
plt.show()

print("Fraudulent Transactions:", len(datafr[datafr['Class']==1]))
print("Usual Transactions:", len(datafr[datafr['Class']==0]))

fraud =len(datafr[datafr['Class']==1])
notfraud = len(datafr[datafr['Class']==0])

# Data to plot
labels = 'Fraud','Not Fraud'
sizes = [fraud,notfraud]

# Plot
plt.figure(figsize=(7,6))
plt.pie(sizes, explode=(0.1, 0.1), labels=labels, colors=sns.color_palette("BuPu"),
autopct='%1.1f%%', shadow=True, startangle=0)
plt.title('Pie Chart Ratio of Transactions by their Class\n', fontsize=16)
sns.set_context("paper", font_scale=1.2)

y = datafr['Class']
X = datafr.drop(['Time','Class', 'binned'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

model = XGBClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))
import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")
model = XGBClassifier()
model.fit(X_train[['V1','V2','V3']], y_train)

y_pred = model.predict(X_test[['V1','V2','V3']])
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy is {:.2f}%".format(accuracy * 100.0))

# assign cnf_matrix with result of confusion_matrix array
cnf_matrix = confusion_matrix(y_test,y_pred)
#create a heat map
sns.heatmap(pd.DataFrame(cnf_matrix), annot = True, cmap = 'Blues', fmt = 'd')
plt.xlabel('Predicted')
plt.ylabel('Expected')
plt.show()
# separate minority and majority classes
not_fraud = X[X.Class==0]
fraud = X[X.Class==1]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=2727) # reproducible results

# combine majority and oversampled minority
oversampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
oversampled.Class.value_counts()
# trying xgboost again with the balanced dataset
y_train = oversampled.Class
X_train = oversampled.drop('Class', axis=1)

upsampled = XGBClassifier()
upsampled.fit(X_train, y_train)

# Predict on test
upsampled_pred = upsampled.predict(X_test)

# predict probabilities
probs = upsampled.predict_proba(X_test)
# keep probabilities for the positive outcome only
probs = probs[:, 1]
explainer = shap.TreeExplainer(upsampled)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test, plot_type="bar")

