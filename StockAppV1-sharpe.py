#!/usr/bin/env python
# coding: utf-8

# In[1]:


import yfinance as yf
import streamlit as st
import datetime 
import pandas as pd
import requests
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
yf.pdr_override()
import altair as alt
from yahoofinancials import YahooFinancials
import requests
from PIL import Image


# In[2]:


stocks=['AALB.AS','ABN.AS','ACCEL.AS','ADYEN.AS','AGN.AS','AJAX.AS','AKZA.AS','ALFEN.AS','ALLFG.AS','AMG.AS','ACOMO.AS','APAM.AS','ARCAD.AS','MT.AS','ASM.AS','ASML.AS','ASRNL.AS','BSGR.AS','BFIT.AS','BESI.AS','BBED.AS','BRNL.AS','CMCOM.AS','CCEP.AS','CRBN.AS','DPA.AS','ECMPA.AS','FLOW.AS','FFARM.AS','FUR.AS','GLPG.AS','HEIJM.AS','HDG.AS','HYDRA.AS','IMCD.AS','INGA.AS','INTER.AS','JDEP.AS','TKWY.AS','KENDR.AS','AD.AS','BAMNB.AS','DSM.AS','KPN.AS','PHIA.AS','VPK.AS','NEDAP.AS','NEWAY.AS','NN.AS','OCI.AS','ORDI.AS','PSH.AS','PHARM.AS','PNL.AS','RAND.AS','BOKA.AS','RDSA.AS','SBMO.AS','SIFG.AS','LIGHT.AS','SLIGR.AS','STRN.AS','TFG.AS','TWEKA.AS','TOM2.AS','URW.AS','UMG.AS','VLK.AS','WHA.AS','WKL.AS']


# In[3]:


st.title("MNB's stock analysis app")
st.markdown("The graphs and tables below can be used to determine an optimal stock portfolio based on the Sharpe ratio and will give you detailed information on any given stock")
#image = Image.open(r'C:\Users\melis\Pictures\wallstphoto.jpg')
#st.image(image)

st.sidebar.header('User Input Parameters')

today = datetime.date.today()
def user_input_features():
    start_date = st.sidebar.text_input("Start Date", '2021-10-10')
    end_date = st.sidebar.text_input("End Date", f'{today}')
    tickerlist=st.sidebar.multiselect('Which stocks would you like to include in the Sharpe ratio analysis?',stocks, ['ASML.AS','DSM.AS','ADYEN.AS'])
    return start_date, end_date, tickerlist

start, end, tickerlist = user_input_features()


start1 = pd.to_datetime(start)
end1 = pd.to_datetime(end)


# In[ ]:


yahoo_financials = YahooFinancials(tickerlist)

data = yahoo_financials.get_historical_price_data(start, 
                                                  end, 
                                                  time_interval='daily')

prices_df = pd.DataFrame({
    a: {x['formatted_date']: x['adjclose'] for x in data[a]['prices']} for a in tickerlist
})


# In[ ]:


returns =prices_df.pct_change()
meanDailyreturns=returns.mean()
daily_cumu_returns=(1+returns).cumprod()
#covariance matrix from daily returns
cov_matrix_d=(returns.cov())*252


# In[ ]:


Sigma = risk_models.sample_cov(prices_df)
mu = expected_returns.mean_historical_return(prices_df)
ef=EfficientFrontier(mu, Sigma)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()


# In[ ]:


weightdf=pd.DataFrame(cleaned_weights.items(), columns=['Stock', 'Weight'])
weightdf= weightdf.set_index('Stock')
weightdf1=weightdf[weightdf['Weight']!=0]


# In[ ]:


stats=(ef.portfolio_performance(verbose=True))


# In[ ]:


# Plot
st.title("Optimized Sharpe ratio portfolio")
weightdf1

st.bar_chart(weightdf1['Weight'])
st.header("Portfolio stats")
st.markdown("Below you can see the expected annual return (x100%), annual volatility (x100%) and the sharpe ratio")
st.write(ef.portfolio_performance(verbose=False))


# In[ ]:




