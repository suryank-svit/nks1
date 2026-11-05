"""Educational ML comparison for BITS Assignment 2. Run in BITS Virtual Lab."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef

HERE = Path(__file__).resolve().parent
(HERE / 'model').mkdir(exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
X = dataset.data.copy()
# Positive class = malignant, for interpretable binary precision/recall.
y = pd.Series(np.where(dataset.target == 0, 1, 0), index=X.index, name='target')
full_data = X.copy()
full_data['target'] = y.map({0:'benign', 1:'malignant'})
full_data.to_csv(HERE / 'dataset.csv', index=False)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=42, stratify=y)
test_data = X_test.copy()
test_data['target'] = y_test.map({0:'benign',1:'malignant'})
test_data.to_csv(HERE / 'test_data.csv', index=False)
models = {
    'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)),
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True, random_state=42)),
}
filenames = {'Logistic Regression':'logistic_regression.joblib', 'Decision Tree':'decision_tree.joblib', 'KNN':'knn.joblib', 'Naive Bayes':'naive_bayes.joblib', 'Support Vector Machine':'svm.joblib'}
rows=[]
for name, clf in models.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    scores = clf.predict_proba(X_test)[:, list(clf.classes_).index(1)]
    rows.append({'Model':name, 'Accuracy':accuracy_score(y_test,pred), 'AUC':roc_auc_score(y_test,scores), 'Precision':precision_score(y_test,pred,zero_division=0), 'Recall':recall_score(y_test,pred,zero_division=0), 'F1':f1_score(y_test,pred,zero_division=0), 'MCC':matthews_corrcoef(y_test,pred)})
    joblib.dump(clf, HERE/'model'/filenames[name])
result=pd.DataFrame(rows)
result.to_csv(HERE / 'model_comparison.csv', index=False)
(HERE/'model_config.json').write_text(json.dumps({'feature_names':X.columns.tolist(), 'model_files':filenames, 'positive_class':'malignant', 'test_rows':len(y_test)}, indent=2), encoding='utf-8')
print('Training:',len(X_train),'Test:',len(X_test))
print(result.round(4).to_string(index=False))
print('\nBest test F1:',result.loc[result.F1.idxmax(),'Model'])
