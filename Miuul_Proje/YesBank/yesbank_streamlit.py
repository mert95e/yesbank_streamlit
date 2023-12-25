# Importing Required Libraries
import streamlit as st
import plotly.express as px
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import math
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import zscore
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn import metrics
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(layout="wide")


# using pandas library and 'read_csv' function to read YesBank_StockPrices csv file
dataset = pd.read_csv("Miuul_Proje/YesBank/data_YesBank_StockPrices.csv")

#tab_home, tab_vis, tab_model = st.tabs(["Ana Sayfa", "Grafikler", "Model"])
tabs = ["Ana Sayfa", "Görsellestirme"]
selected_tab = st.sidebar.radio("Select Tab", tabs)

# TAB HOME
if selected_tab == "Ana Sayfa":
    st.header(" Yes Bank'ın Hisse Fiyatlarının Analizi ")
    st.subheader("Veri Seti Hakkında:")
    st.markdown("Yes Bank, Hindistan finans alanında tanınmış bir bankadır. 2018'den beri Rana Kapoor'un karıştığı dolandırıcılık davası nedeniyle haberlerde yer alıyor. 
Yes Bank'ın son yıllardaki finansal zorlukları, hisse senedi piyasasında önemli dalgalanmalara neden oldu. Bu projede, Yes Bank'ın hisse fiyatlarını derinlemesine analiz ediyoruz.

Bu projenin amacı, Yes Bank'ın hisse fiyatlarındaki tarihsel eğilimleri anlamak ve fiyat hareketlerini tahmin etmektir.")

    fig = go.Figure()

    # Add traces for 'Close', 'Open', and 'High'
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset['Close'], mode='lines', name='Close'))
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=dataset.index, y=dataset['High'], mode='lines', name='High'))

    # Set layout parameters
    fig.update_layout(title="Stock Prices",
                      xaxis_title="Month",
                      yaxis_title="Price",
                      legend=dict(title="Legend"),
                      height=500,
                      width=800)

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)


    # dataframe'i gösteriyor ekranda.
    st.dataframe(dataset, width=750)
if selected_tab == "Görsellestirme":
#Linear Regression Model
#Normalization
# Splitting our data into Dependent and Independent Variables
    X = dataset.drop(columns=['Close','Date']).apply(zscore)
    y = np.log10(dataset['Close'])

# Creating Testing and Training Datasets
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state = 1)

#Linear Regression
    reg = LinearRegression()
    reg_model = reg.fit(X_train,y_train)

    reg.score(X_train,y_train)

    y_test_pred = reg.predict(X_test)
    y_train_pred = reg.predict(X_train)

    #Linear Regression Predication vs Actual

    import plotly.graph_objects as go
    import plotly.express as px
    import streamlit as st
    import numpy as np

    # Assuming y_test and y_test_pred are your data arrays
    y_test = np.array([y_test])
    y_test_pred = np.array([y_test_pred])

    # Create a Plotly figure
    fig = go.Figure()

    # Add traces for actual and predicted data
    fig.add_trace(go.Scatter(x=np.arange(len(y_test[0])), y=10 ** y_test[0], mode = 'lines', name = 'Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_test_pred[0])), y=10 ** y_test_pred[0], mode = 'lines', name = 'Predicted'))

    # Update layout with desired properties
    fig.update_layout(
        title="Linear Regression Prediction vs Actual",
        xaxis_title="No of Test Data",
        yaxis_title="Close Stock Price",
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500,
        width=800
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    #Evaluation Metrics

    # Test Performance Metrics
    mse = mean_squared_error(y_test[0], y_test_pred[0])
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test[0], y_test_pred[0])
    r2 = r2_score(y_test[0], y_test_pred[0])

    # Display metrics in Streamlit
    st.write("Test Performance Metrics:")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R-squared: {r2}")

    #Lasso Regression

    from sklearn.linear_model import Lasso
    #Create and fit the lasso model
    lasso = Lasso(alpha=0.005,max_iter=3000)
    lasso_model = lasso.fit(X_train,y_train)

    #Calculate and display the R-squared score on the training set
    lasso.score(X_train,y_train)
    y_lasso_pred = lasso.predict(X_test)

    #Cross Validation

    # Hyper-parameter Tuning

    lasso_cv = Lasso()
    parameters = {'alpha':[1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100,0.0014]}
    lasso_model = GridSearchCV(lasso_cv,parameters,scoring = 'neg_mean_squared_error',cv = 3)

    lasso_model.fit(X_train,y_train)

    #st.write("The best fit alpha value is found out to be :" ,lasso_model.best_params_)
    #st.write("\nUsing ",lasso_model.best_params_, " the negative mean squared error is: ", lasso_model.best_score_)

    y_pred_lasso = lasso_model.predict(X_test)

# Add traces for actual and predicted data
    fig.add_trace(go.Scatter(x=np.arange(len(y_test[0])), y=10 ** y_test[0], mode = 'lines', name = 'Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred_lasso)), y=10 ** y_pred_lasso, mode = 'lines', name = 'Predicted'))

    # Update layout with desired properties
    fig.update_layout(
        title="Lasso Plotting Prediction vs Actual",
        xaxis_title="No of Test Data",
        yaxis_title="Close Stock Price",
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500,
        width=800
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    #Evaluation Metrics

    # Test Performance Metrics
    mse = mean_squared_error(y_test[0], y_pred_lasso)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test[0], y_pred_lasso)
    r2 = r2_score(y_test[0], y_pred_lasso)

    # Display metrics in Streamlit
    st.write("Test Performance Metrics:")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R-squared: {r2}")

#Ridge Linear Regression

    ridge  = Ridge(alpha=0.1)
    ridge.fit(X_train,y_train)

    ridge.score(X_train, y_train)
    y_ridge_pred = ridge.predict(X_test)

    #Cross Validation

    # Hyper-parameter Tuning
    ridge_cv = Ridge()
    parameters = {'alpha':[1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100]}
    ridge_model = GridSearchCV(ridge_cv,parameters,scoring='neg_mean_squared_error',cv=3)
    ridge_model.fit(X_train,y_train)

    # Model Prediction
    y_pred_ridge  = ridge_model.predict(X_test)

    #Ridge Predication vs Actual (After Validification)

    # Add traces for actual and predicted data
    fig.add_trace(go.Scatter(x=np.arange(len(y_test[0])), y=10 ** y_test[0], mode = 'lines', name = 'Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_pred_ridge)), y=10 ** y_pred_ridge, mode = 'lines', name = 'Predicted'))

    # Update layout with desired properties
    fig.update_layout(
        title="Ridge Plotting Prediction vs Actual",
        xaxis_title="No of Test Data",
        yaxis_title="Close Stock Price",
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500,
        width=800
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    #Evaluation Metrics

    # Test Performance Metrics
    mse = mean_squared_error(y_test[0], y_pred_ridge)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test[0], y_pred_ridge)
    r2 = r2_score(y_test[0], y_pred_ridge)

    # Display metrics in Streamlit
    st.write("Test Performance Metrics:")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R-squared: {r2}")

#Elastic Net Linear Regression

    from sklearn.linear_model import ElasticNet

    elastic = ElasticNet(alpha=0.1,l1_ratio=0.5)
    elastic.fit(X_train,y_train)

    y_elastic_pred = elastic.predict(X_test)

    #Cross Validification

    elastic_cv = ElasticNet()
    parameters = {'alpha':[1e-15,1e-13,1e-10,1e-8,1e-5,1e-4,1e-3,1e-2,1e-1,1,5,10,20,30,40,45,50,55,60,100],'l1_ratio':[0.3,0.4,0.5,0.6,0.7,0.8,1,2]}
    elastic_model = GridSearchCV(elastic_cv,parameters,scoring='neg_mean_squared_error',cv=3)
    elastic_model.fit(X_train,y_train)

    y_elastic_pred = elastic_model.predict(X_test)

    #ElasticNet Predication vs Actual (After Validification)

    # Add traces for actual and predicted data
    fig.add_trace(go.Scatter(x=np.arange(len(y_test[0])), y=10 ** y_test[0], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=np.arange(len(y_elastic_pred)), y=10 ** y_elastic_pred, mode='lines', name='Predicted'))

    # Update layout with desired properties
    fig.update_layout(
        title="Elastic Net Plotting Prediction vs Actual",
        xaxis_title="No of Test Data",
        yaxis_title="Close Stock Price",
        legend=dict(x=0, y=1, traceorder='normal'),
        height=500,
        width=800
    )

    # Display the Plotly figure in Streamlit
    st.plotly_chart(fig)

    #Evaluation Metrics

    # Test Performance Metrics
    mse = mean_squared_error(y_test[0], y_elastic_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_test[0], y_elastic_pred)
    r2 = r2_score(y_test[0], y_elastic_pred)

    # Display metrics in Streamlit
    st.write("Test Performance Metrics:")
    st.write(f"MSE: {mse}")
    st.write(f"RMSE: {rmse}")
    st.write(f"MAE: {mae}")
    st.write(f"R-squared: {r2}")

