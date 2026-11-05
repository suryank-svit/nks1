"""Streamlit app: evaluate pre-trained models only; never fit on uploaded test data."""
from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, recall_score,
                             f1_score, matthews_corrcoef, confusion_matrix, classification_report,
                             ConfusionMatrixDisplay)

HERE = Path(__file__).resolve().parent
st.set_page_config(page_title='ML Classifier Evaluation', page_icon='📊', layout='wide')
st.title('Classification Model Evaluation')
st.caption('BITS Pilani M.Tech AIML • Five pretrained classifiers • Test data only')
st.info('Upload a CSV with the original 30 numeric features and target column (benign/malignant). The application does not retrain models.')
config = json.loads((HERE/'model_config.json').read_text(encoding='utf-8'))
choice = st.selectbox('Choose a trained classification model', list(config['model_files']))
file = st.file_uploader('Upload test_data.csv',type=['csv'])
if file is not None:
    try:
        test = pd.read_csv(file)
        expected=config['feature_names']+['target']
        missing=[name for name in expected if name not in test.columns]
        if missing:
            st.error('Missing required columns: '+', '.join(missing))
            st.stop()
        if test.empty:
            st.error('The CSV contains no rows.')
            st.stop()
        X=test[config['feature_names']].apply(pd.to_numeric,errors='raise')
        if X.isna().any().any():
            st.error('Input features must not contain missing values.')
            st.stop()
        text=test['target'].astype(str).str.strip().str.lower()
        if not text.isin(['benign','malignant']).all():
            st.error('The target must contain only benign and malignant labels.')
            st.stop()
        y=(text=='malignant').astype(int)
        clf=joblib.load(HERE/'model'/config['model_files'][choice])
        prediction=clf.predict(X)
        scores=clf.predict_proba(X)[:,list(clf.classes_).index(1)]
        metrics={'Accuracy':accuracy_score(y,prediction),'AUC':roc_auc_score(y,scores) if y.nunique()==2 else None,
                 'Precision':precision_score(y,prediction,zero_division=0),'Recall':recall_score(y,prediction,zero_division=0),
                 'F1':f1_score(y,prediction,zero_division=0),'MCC':matthews_corrcoef(y,prediction)}
        cols=st.columns(3)
        for i,(name,value) in enumerate(metrics.items()):
            cols[i%3].metric(name,'N/A' if value is None else f'{value:.4f}')
        st.caption(f'{len(y)} test rows • Positive class: malignant • AUC needs both classes')
        left,right=st.columns(2)
        with left:
            st.subheader('Confusion matrix')
            cm=confusion_matrix(y,prediction,labels=[0,1])
            fig,ax=plt.subplots()
            ConfusionMatrixDisplay(cm,display_labels=['Benign','Malignant']).plot(ax=ax,cmap='Blues',colorbar=False)
            st.pyplot(fig)
        with right:
            st.subheader('Classification report')
            report=classification_report(y,prediction,labels=[0,1],target_names=['Benign','Malignant'],output_dict=True,zero_division=0)
            st.dataframe(pd.DataFrame(report).T)
    except Exception as error:
        st.error(f'Could not evaluate CSV: {error}')
else:
    st.write('Upload the supplied test_data.csv to display metrics and the confusion matrix.')
