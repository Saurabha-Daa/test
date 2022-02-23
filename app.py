#Import libraries
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import streamlit as st

#Function named dataframe_optimizer is defined. This will reduce space consumption by dataframes.
#Credit - https://www.kaggle.com/rinnqd/reduce-memory-usage and 
#https://www.analyticsvidhya.com/blog/2021/04/how-to-reduce-memory-usage-in-python-pandas/
def dataframe_optimizer(df):
  '''This is a dataframe optimizer'''
  start_mem=np.round(df.memory_usage().sum()/1024**2,2)    
  for col in df.columns:
    col_type=df[col].dtype        
    if col_type!=object:
      c_min=df[col].min()
      c_max=df[col].max()
      if str(col_type)[:3]=='int':
        if c_min>np.iinfo(np.int8).min and c_max<np.iinfo(np.int8).max:
            df[col]=df[col].astype(np.int8)
        elif c_min>np.iinfo(np.int16).min and c_max<np.iinfo(np.int16).max:
            df[col]=df[col].astype(np.int16)
        elif c_min>np.iinfo(np.int32).min and c_max<np.iinfo(np.int32).max:
            df[col]=df[col].astype(np.int32)
        elif c_min>np.iinfo(np.int64).min and c_max<np.iinfo(np.int64).max:
            df[col]=df[col].astype(np.int64)  
      else:
        if c_min>np.finfo(np.float16).min and c_max<np.finfo(np.float16).max:
            df[col]=df[col].astype(np.float16)
        elif c_min>np.finfo(np.float32).min and c_max<np.finfo(np.float32).max:
            df[col]=df[col].astype(np.float32)
        else:
            df[col]=df[col].astype(np.float64)
  end_mem=np.round(df.memory_usage().sum()/1024**2,2)
  return df

#Import saved data and pickle files
bureau_numerical_merge = dataframe_optimizer(pd.read_csv('bureau_numerical_merge.csv'))
bureau_categorical_merge = dataframe_optimizer(pd.read_csv('bureau_categorical_merge.csv'))
previous_numerical_merge = dataframe_optimizer(pd.read_csv('previous_numerical_merge.csv'))
previous_categorical_merge = dataframe_optimizer(pd.read_csv('previous_categorical_merge.csv'))
query_template = pd.read_csv('query_template.csv')
filename = open('columns_input.pkl', 'rb')
columns_input = pickle.load(filename)
filename.close()
filename1 = open('model.pkl', 'rb')
model = pickle.load(filename1)
filename1.close()
filename2 = open('imputer.pkl', 'rb')
imputer = pickle.load(filename2)
filename2.close()
filename3 = open('scaler.pkl', 'rb')
scaler = pickle.load(filename3)
filename3.close()
filename4 = open('imputer_constant.pkl', 'rb')
imputer_constant = pickle.load(filename4)
filename4.close()
filename5 = open('ohe.pkl', 'rb')
ohe = pickle.load(filename5)
filename5.close()
filename6 = open('selected_features.pkl', 'rb')
selected_features = pickle.load(filename6)
filename6.close()
filename7 = open('columns_ohe.pkl', 'rb')
columns_ohe = pickle.load(filename7)
filename7.close()

#Define a function to create a pipeline for prediction
def inference(query): 
    #Add columns titled DEBT_INCOME_RATIO, LOAN_VALUE_RATIO & LOAN_INCOME_RATIO to a copy of query data
    #query_with_additinal_features = query.copy()    
    query['DEBT_INCOME_RATIO'] = query['AMT_ANNUITY']/query['AMT_INCOME_TOTAL']
    query['LOAN_VALUE_RATIO'] = query['AMT_CREDIT']/query['AMT_GOODS_PRICE']
    query['LOAN_INCOME_RATIO'] = query['AMT_CREDIT']/query['AMT_INCOME_TOTAL']

    #Merge numerical features from bureau to query data
    query_bureau = query.merge(bureau_numerical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_BUREAU'))
    #Merge categorical features from bureau to query data
    query_bureau = query_bureau.merge(bureau_categorical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_BUREAU'))
    #Drop SK_ID_BUREAU
    query_bureau = query_bureau.drop(columns = ['SK_ID_BUREAU'])  
    #Merge numerical features from previous_application to query_bureau
    query_bureau_previous = query_bureau.merge(previous_numerical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_PREVIOUS'))
    #Merge categorical features from previous_application to query_bureau
    query_bureau_previous = query_bureau_previous.merge(previous_categorical_merge, on='SK_ID_CURR', how='left', suffixes=('', '_PREVIOUS'))
    #Drop SK_ID_PREV
    query_bureau_previous = query_bureau_previous.drop(columns = ['SK_ID_PREV'])  
    #Drop SK_ID_PREV
    query_bureau_previous = query_bureau_previous.drop(columns = ['SK_ID_CURR'])

    query_numerical = query_bureau_previous.select_dtypes(exclude=object)
    query_categorical = query_bureau_previous.select_dtypes(include=object)

    columns_numerical = query_numerical.columns
    columns_categorical = query_categorical.columns

    query_numerical_imputed_scaled_df = imputer.transform(query_numerical)
    query_numerical_imputed_scaled_df = scaler.transform(query_numerical_imputed_scaled_df)
    query_numerical_imputed_scaled_df = pd.DataFrame(data = query_numerical_imputed_scaled_df, columns = columns_numerical)

    query_data = imputer_constant.transform(query_categorical)
    query_data = ohe.transform(query_data)
    query_data = pd.DataFrame(data = query_data.toarray(), columns = columns_ohe)

    query_data = pd.concat([query_numerical_imputed_scaled_df, query_data], axis = 1)
    query_data = query_data[selected_features]

    predictions = model.predict(query_data)
    return predictions
    
def main():
    st.sidebar.write("This predictor is based on a Kaggle competition. This competition and datasets can be accessed from https://www.kaggle.com/c/home-credit-default-risk/overview. The source code for this predictor can be accessed from https://github.com/Saurabha-Daa/test.")
    st.write('LOAN DEFAULT TENDENCY PREDICTOR')
    template = query_template.to_csv().encode('utf-8')
    st.download_button("Download template for query data", template, "query_template.csv", key='text/csv')
    uploaded_file = st.file_uploader("Choose a query data file")       
    if uploaded_file is not None:
        query = dataframe_optimizer(pd.read_csv(uploaded_file))
        #query_prediction = inference(query)
        #st.write(query_prediction)        
        columns_query = list(query.columns)
        if columns_query == columns_input:
            query_data_with_prediction = query.copy()
            query_prediction = inference(query)            
            query_data_with_prediction['LABEL'] = query_prediction
            conditions = [(query_data_with_prediction['LABEL'] == 0), (query_data_with_prediction['LABEL'] == 1)]
            values = ['NO', 'YES']
            query_data_with_prediction['DEFAULT TENDENCY'] = np.select(conditions, values)
            query_data_with_prediction = query_data_with_prediction.drop(columns = ['LABEL']).to_csv().encode('utf-8')
            st.write('Default tendency of a loan applicant can be seen under column titled DEFAULT TENDENCY')
            st.write(query_data_with_prediction)
            st.download_button("Download query data with predictions as CSV", query_data_with_prediction, "prediction.csv", key='text/csv')
        else:
          print("Query columns do not match the columns of required format as given in template. Please upload query data in the given format.")

if __name__=='__main__':
    main()
