# -*- coding: utf-8 -*-
"""

@Team 4 : 
Jacobo Valderrama 
Cecilia Prada 
Juliana Leaño 
Sofia Herrera 
Jacqueline Xu 
Xuefei Wang

"""

import pandas as pd
import pandas
import missingno
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from plotnine import *
from plotnine import ggplot, aes, geom_bar, geom_text, labs, theme_bw, ggtitle, after_stat
import datetime
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import statsmodels.formula.api as smf#used for fitting models
import stepwise as stepwise
from sklearn.metrics import roc_curve#for performance
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from mizani.formatters import percent_format
import sys#maxsize
import plotnine as pn

#Loading all data
CustomerData = pd.read_csv("customerdata.csv",parse_dates=['Registrationdate'])
print(CustomerData.info())
SummerFinTRX= pd.read_csv("summerfintrx.csv",parse_dates=['Date'])
print(SummerFinTRX.info())
SummersessTRX= pd.read_csv("summersesstrx.csv",parse_dates=['Date'])
print(SummersessTRX.info())
FallFinTRX= pd.read_csv("fallfintrx.csv",parse_dates=['Date'])
print(FallFinTRX.info())
FallSesstrx=pd.read_csv("fallsesstrx.csv",parse_dates=['Date'])
print(FallSesstrx.info())

#Checking for missing data
missingno.matrix(CustomerData)
plt.show()
missingno.matrix(SummerFinTRX)
plt.show()
missingno.matrix(SummersessTRX)
plt.show()
missingno.matrix(FallFinTRX)
plt.show()
missingno.matrix(FallSesstrx)
plt.show()

#Renaming Date column of transaction not to mix them
SummerFinTRX = SummerFinTRX.rename(columns={'Date': 'TransactionDate'})
SummerFinTRX.head()



"""Question 1 – Creation of a basetable + profiling"""
   
#Creating first basetable with RFM based on last session as recency, number of sessions as frequency and time as value:
SummersessTRXMerged=SummersessTRX.merge(CustomerData, how='left', left_on='CustomerID', right_on='CustomerID')
SummersessTRXMerged.info()

endindependent=datetime.datetime.strptime('30/08/2022','%d/%m/%Y')
SessionsBasetable=SummersessTRXMerged.groupby('CustomerID').agg({'CustomerID':'mean','SessionID':'count','Duration':'sum','Date': lambda date: (endindependent - date.max()).days,
                                        'fallbonus':'mean',
                                        'Registrationdate': 'min',
                                        'Pokestops':'sum',
                                        'Gyms':'sum','Raids':'sum','Social':'sum',
                                        'Pokemons':'sum','Distance':'sum','Experience':'sum',
                                        'Gender':'mean','Age':'mean',
                                        'Income':'mean','CustomerType':'mean',
                                        })
#Changing names of columns to identify RFM
SessionsBasetable=SessionsBasetable.rename(columns={'SessionID': 'Frequency(sessions)','Duration':'Value (totalduration)','Date':'Recency'})

#Creating second basetable with RFM based on last transaction as recency, number of transactions as frequency and money as value:
SummerFinTRXMerged=SummerFinTRX.merge(CustomerData, how='left', left_on='CustomerID', right_on='CustomerID')
SummerFinTRXMerged.info()

FinancialsBasetable=SummerFinTRXMerged.groupby('CustomerID').agg({'CustomerID':'mean','TransactionID':'count','Value':'sum','TransactionDate': lambda date: (endindependent - date.max()).days,
                                        'fallbonus':'mean',
                                        'Registrationdate': 'min',
                                        'Gender':'mean','Age':'mean',
                                        'Income':'mean','CustomerType':'mean',
                                        })
#Changing names of columns to identify RFM
FinancialsBasetable=FinancialsBasetable.rename(columns={'TransactionID': 'Frequency(transactions)','Value':'Value (totalspend)','TransactionDate':'Recency'})

#Checking for missing values
missingno.matrix(FinancialsBasetable)
plt.show()
missingno.matrix(SessionsBasetable)
plt.show()



"""Data preparation for the next steps & questions"""
#Gathering info for seniority
SessionsBasetable['Days_Joined'] = (endindependent - SessionsBasetable['Registrationdate']).dt.days
SessionsBasetable['Years_Joined'] = (SessionsBasetable['Days_Joined']/360)
FinancialsBasetable['Days_Joined'] = (endindependent - FinancialsBasetable['Registrationdate']).dt.days
FinancialsBasetable['Years_Joined'] = (FinancialsBasetable['Days_Joined']/360)

#Making sure CustomerType data are integers
SessionsBasetable['CustomerType'] = SessionsBasetable['CustomerType'].astype(int)

#Defining bins & labels for segments
freq_bins = [0, 3, 7, 11, float('inf')]
freq_labels = ['1-3', '4-7', '8-11', '>11']

freq_bins_fin = [0, 1, 2, 3, float('inf')]
freq_labels_fin = ['1', '2', '3', '>3']

rec_bins = [-0.5, 21, 42, 63, 84, 105, float('inf')]  # Represents days
rec_labels = ['0-21d', '22-42d', '43-63d', '64-84d', '84-105d', '>105d']

years_bins = [-0.5, 1, 3, 6, float('inf')]
years_labels = ['0-1 years', '1-3 years', '4-6 years', '>6 years']

age_bins = [0, 18, 25, 32, 39, 48, float('inf')]
age_labels = ['0-18yo', '19-25yo', '26-32yo', '33-39yo', '40-48yo', '>48yo']

spend_bins_fin = [0, 
                  FinancialsBasetable['Value (totalspend)'].quantile(0.33),
                FinancialsBasetable['Value (totalspend)'].quantile(0.66),
                float('inf')]
spend_labels_fin = ['Low', 'Medium', 'High']

duration_bins = [0, 
                 SessionsBasetable['Value (totalduration)'].quantile(0.33),
                 SessionsBasetable['Value (totalduration)'].quantile(0.66),
                 float('inf')]
duration_labels = ['Low', 'Medium', 'High']

#Creating segments for frequency (sessions), recency (sessions) and age
#Using pd.cut() to assign each user a segment based on their data
SessionsBasetable['segmfreq'] = pd.cut(SessionsBasetable['Frequency(sessions)'], 
                                      bins=freq_bins,
                                      labels=freq_labels)
FinancialsBasetable['segmfreq'] = pd.cut(FinancialsBasetable['Frequency(transactions)'], 
                                      bins=freq_bins_fin,
                                      labels=freq_labels_fin)
SessionsBasetable['segmrec'] = pd.cut(SessionsBasetable['Recency'], 
                                     bins=rec_bins,
                                     labels=rec_labels)
FinancialsBasetable['segmrec'] = pd.cut(FinancialsBasetable['Recency'], 
                                     bins=rec_bins,
                                     labels=rec_labels)
SessionsBasetable['segmage'] = pd.cut(SessionsBasetable['Age'], 
                                     bins=age_bins,
                                     labels=age_labels)
FinancialsBasetable['segmage'] = pd.cut(FinancialsBasetable['Age'], 
                                     bins=age_bins,
                                     labels=age_labels)
SessionsBasetable['seniority_segment'] = pd.cut(SessionsBasetable['Years_Joined'], 
                                                bins=years_bins, 
                                                labels=years_labels)
FinancialsBasetable['seniority_segment'] = pd.cut(FinancialsBasetable['Years_Joined'], 
                                                bins=years_bins, 
                                                labels=years_labels, 
                                                include_lowest=True)
SessionsBasetable['value_duration'] = pd.cut(SessionsBasetable['Value (totalduration)'], 
                                             bins=duration_bins,
                                             labels=duration_labels)
FinancialsBasetable['value_spent'] = pd.cut(FinancialsBasetable['Value (totalspend)'], 
                                            bins=spend_bins_fin,
                                            labels=spend_labels_fin)

#Convert to categorical for proper ordering
SessionsBasetable['seniority_segment'] = pd.Categorical(SessionsBasetable['seniority_segment'], 
                                                        categories=years_labels, 
                                                        ordered=True)
FinancialsBasetable['seniority_segment'] = pd.Categorical(FinancialsBasetable['seniority_segment'], 
                                                        categories=years_labels, 
                                                        ordered=True)

#Mapping CustomerType to descriptive labels, match numbers with names for niantic segments
type_mapping = {0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'}
SessionsBasetable['PlayerType'] = SessionsBasetable['CustomerType'].map(type_mapping)
FinancialsBasetable['PlayerType'] = FinancialsBasetable['CustomerType'].map(type_mapping)

gender_mapping = {0: 'Male', 1: 'Female'} 
SessionsBasetable['Gender'] = SessionsBasetable['Gender'].map(gender_mapping)
FinancialsBasetable['Gender'] = FinancialsBasetable['Gender'].map(gender_mapping)
#here gender as binary disappear

#Income mapping
income_mapping = {0: 'Low', 1: 'Medium', 2: 'High'}
SessionsBasetable['Income'] = SessionsBasetable['Income'].map(income_mapping)
FinancialsBasetable['Income'] = FinancialsBasetable['Income'].map(income_mapping)

#Sort Income in right order
income_order = ['Low', 'Medium', 'High']
SessionsBasetable['Income'] = pd.Categorical(SessionsBasetable['Income'],
                                               categories=income_order,
                                               ordered=True)
FinancialsBasetable['Income'] = pd.Categorical(FinancialsBasetable['Income'],
                                               categories=income_order,
                                               ordered=True)

#Sort value spent in right order
spend_order = ['Low', 'Medium', 'High']
SessionsBasetable['value_duration'] = pd.Categorical(SessionsBasetable['value_duration'], 
                                                    categories=spend_order, 
                                                    ordered=True)
FinancialsBasetable['value_spent'] = pd.Categorical(FinancialsBasetable['value_spent'], 
                                                    categories=spend_order, 
                                                    ordered=True)

"""end of data preparation"""



#Calculating CLV

#Assumptions
T = 3  # Forecasting over 3 years
retention_rate = 0.4  # Retention probability per year
discount_rate = 0.12  # Discount rate (12%)
acquisition_cost = 5  # Assume acquisition cost
profit_margin = 0.3

#Computing Average Order Value (AOV)
FinancialsBasetable["AOV"] = FinancialsBasetable["Value (totalspend)"] / FinancialsBasetable["Frequency(transactions)"]
#Computing Annual Purchase Frequency
FinancialsBasetable["Annual_Purchase_Frequency"] = FinancialsBasetable["Frequency(transactions)"] / ((FinancialsBasetable["Recency"]+1) / 365)  # Convert to yearly frequency
#Computing Annual Revenue per Customer
FinancialsBasetable["Annual_Revenue"] = FinancialsBasetable["AOV"] * FinancialsBasetable["Annual_Purchase_Frequency"]
#Computing Annual Margin per Customer
FinancialsBasetable["Annual_Margin"] = FinancialsBasetable["Annual_Revenue"] * 0.3

#Function to compute CLV using annual revenue
def calc_clv(margin, r, d, acquisition, t):
    clv = -acquisition  # Start with acquisition cost
    for i in range(0, t + 1):  # Iterate over time periods
        clv += (pow(r, i) * margin) / (pow((1 + d), i))  # Discounted future value
    return clv

#Applying the function to calculate CLV using Annual Margin
FinancialsBasetable["Annual_CLV"] = FinancialsBasetable["Annual_Margin"].apply(lambda x: calc_clv(x, retention_rate, discount_rate, acquisition_cost, T))

FinancialsBasetable.to_excel("Customer_CLV_Output.xlsx", index=False)

#Printing the first few rows
print(FinancialsBasetable[["CustomerID", "Annual_Revenue", "Annual_CLV"]].head())

#Analyzing the impact of Age, Gender, Customer Type, and Income on CLV
impact_factors = ["Age", "Gender", "CustomerType", "Income"]
impact_results = FinancialsBasetable.groupby(impact_factors)["Annual_CLV"].mean().reset_index()

#Displaying insights
print(impact_results)

#Insight 1: Customer Type Impact
plt.figure(figsize=(8, 4))
sns.boxplot(x=FinancialsBasetable["CustomerType"], y=FinancialsBasetable["Annual_CLV"])
plt.xlabel("Customer Type")
plt.ylabel("Annual CLV")
plt.title("Insight 1: Customer Type Strongly Affects CLV")
plt.show()

#Insight 2: Income Level Impact
plt.figure(figsize=(8, 4))
sns.boxplot(x=FinancialsBasetable["Income"], y=FinancialsBasetable["Annual_CLV"])
plt.xlabel("Income Level")
plt.ylabel("Annual CLV")
plt.title("Insight 2: Not Higher Income Customers Have Higher CLV")
plt.show()

#Insight 3: Gender Impact
plt.figure(figsize=(8, 4))
sns.boxplot(x=FinancialsBasetable["Gender"], y=FinancialsBasetable["Annual_CLV"])
plt.xlabel("Gender")
plt.ylabel("Annual CLV")
plt.title("Insight 3: Gender Has Minimal Impact on CLV")
plt.show()

#Insight 4: Age Impac
#Defining age bins and labels
plt.figure(figsize=(10, 10))
sns.boxplot(x=FinancialsBasetable["segmage"], y=FinancialsBasetable["Annual_CLV"])
plt.xlabel("Age Segment")
plt.ylabel("Annual CLV")
plt.title("Insight 4: Age Group Impact on CLV")
plt.xticks(rotation=45)
plt.show()



"""Question 2 – Lifecycle grids (& histograms)"""

#Histograms - General plots for all players:
    
#Sessions histograms :

#Frequency distribution:
histfreq=(ggplot(SessionsBasetable, aes(x='Frequency(sessions)')) +
  theme_bw() +
  geom_bar(alpha=0.6) +  
  ggtitle("Distribution of frequency - Sessions") +
  labs(y="Number of Customers", x="Frequency (sessions)") +
  geom_text(aes(label='..count..'), stat="count", va='bottom'))
histfreq.show()

histrec = (ggplot(SessionsBasetable, aes(x='segmfreq')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Frequency - Sessions") +
  labs(y="Percentage of Customers", x="Frequency segment (sessions)") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',  # Format as percentage
            va='bottom'))
histrec.show()

#Recency distribution:
histrec=(ggplot(SessionsBasetable, aes(x='Recency')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Recency - Sessions") +
  labs(y="Number of Customers", x="Recency (sessions)"))
histrec.show()

histrec = (ggplot(SessionsBasetable, aes(x='segmrec')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Recency - Sessions") +
  labs(y="Percentage of Customers", x="Recency segment (sessions)") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',
            va='bottom'))
histrec.show()

#Gender distribution
histgender = (ggplot(SessionsBasetable, aes(x='Gender')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Gender - Sessions") +
  labs(y="Percentage of Customers") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%', 
            va='bottom'))
histgender.show()

#Age distribution :
histage = (ggplot(SessionsBasetable, aes(x='segmage')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Age - Sessions") +
  labs(y="Percentage of Customers", x="Age segment") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%', 
            va='bottom'))
histage.show()

#Income distribution:  
histinc = (ggplot(SessionsBasetable, aes(x='Income')) +
    theme_bw() +
    geom_bar(alpha=0.6) +
    ggtitle("Distribution of Income - Sessions") +
    labs(y="Percentage of Customers", x="Income level") +
    geom_text(aes(label=after_stat('count / sum(count) * 100')), 
              stat="count", 
              format_string='{:.1f}%', 
              va='bottom'))
histinc.show()

#Seniority distribution
histsen=(ggplot(SessionsBasetable, aes(x='seniority_segment')) +
  theme_bw() +
  geom_bar(alpha=0.6) +  
  ggtitle("Players seniority - Sessions") +
  labs(y="Number of Customers", x="Seniority segment") +
  geom_text(aes(label='..count..'), stat="count", va='bottom'))
histsen.show()

histsenper = (ggplot(SessionsBasetable, aes(x='seniority_segment')) +
    theme_bw() +
    geom_bar(alpha=0.6) +
    ggtitle("Players seniority - Sessions") +
    labs(y="Percentage of Customers", x="Seniority segment") +
    geom_text(aes(label=after_stat('count / sum(count) * 100')), 
              stat="count", 
              format_string='{:.1f}%', 
              va='bottom'))
histsenper.show()

#Stacked column chart
histstack = (ggplot(SessionsBasetable, aes(x='seniority_segment', fill='PlayerType')) +
    theme_bw() +
    geom_bar(stat="count", alpha=0.6, position="stack") +  # Stacked bars
    ggtitle("Players seniority & Player Type - Sessions") +
    labs(y="Number of Customers", x="Seniority Segment") +
    geom_text(aes(label=after_stat('count')), stat="count", position="stack", va='center'))
histstack.show()

#Histogram for duration level
histdur = (ggplot(SessionsBasetable, aes(x='value_duration')) +
    theme_bw() +
    geom_bar(alpha=0.6) +
    ggtitle("Distribution of Duration") +
    labs(y="Percentage of Customers", x="Duration value") +
    geom_text(aes(label=after_stat('count / sum(count) * 100')), 
              stat="count", 
              format_string='{:.1f}%', 
              va='bottom'))
histdur.show()



#Financials histograms:

#Frequency distribution
histfreqfin=(ggplot(FinancialsBasetable, aes(x='Frequency(transactions)')) +
  theme_bw() +
  geom_bar(alpha=0.6) +  
  ggtitle("Distribution of frequency - Transactions") +
  labs(y="Number of Customers", x="Frequence (transactions)") +
  geom_text(aes(label='..count..'), stat="count", va='bottom'))
histfreqfin.show()

histfreqsegfin = (ggplot(FinancialsBasetable, aes(x='Frequency(transactions)')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Frequency - Transactions") +
  labs(y="Percentage of Customers", x="Frequence (transactions)") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',
            va='bottom'))
histfreqsegfin.show()

#Recency distribution:
histrecfin = (ggplot(FinancialsBasetable, aes(x='segmrec')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Recency - Transactions") +
  labs(y="Percentage of Customers", x="Recency segment (transactions)") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',
            va='bottom'))
histrecfin.show()

#Age distribution:
histagefin = (ggplot(FinancialsBasetable, aes(x='segmage')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Age - Transactions") +
  labs(y="Percentage of Customers", x="Age segment") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',
            va='bottom'))
histagefin.show()

#Gender distribution
histgenfin = (ggplot(FinancialsBasetable, aes(x='Gender')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of Gender - Transactions") +
  labs(y="Percentage of Customers") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%', 
            va='bottom'))
histgenfin.show()

#Seniority distribution
histsenfin=(ggplot(FinancialsBasetable, aes(x='seniority_segment')) +
  theme_bw() +
  geom_bar(alpha=0.6) +  
  ggtitle("Players seniority - Transactions") +
  labs(y="Number of Customers", x="Seniority segment") +
  geom_text(aes(label='..count..'), stat="count", va='bottom'))
histsenfin.show()

histsenperfin = (ggplot(FinancialsBasetable, aes(x='seniority_segment')) +
    theme_bw() +
    geom_bar(alpha=0.6) +
    ggtitle("Players seniority - Transactions") +
    labs(y="Percentage of Customers", x="Seniority segment") +
    geom_text(aes(label=after_stat('count / sum(count) * 100')), 
              stat="count", 
              format_string='{:.1f}%', 
              va='bottom'))
histsenperfin.show()

#Stacked column chart
histstackfin = (ggplot(FinancialsBasetable, aes(x='seniority_segment', fill='PlayerType')) +
    theme_bw() +
    geom_bar(stat="count", alpha=0.6, position="stack") +  # Stacked bars
    ggtitle("Players seniority & Player Type - Transactions") +
    labs(y="Number of Customers", x="Seniority Segment") +
    geom_text(aes(label=after_stat('count')), stat="count", position="stack", va='center'))
histstackfin.show()

#Value spent distribution:    
histspent = (ggplot(FinancialsBasetable, aes(x='value_spent')) +
  theme_bw() +
  geom_bar(alpha=0.6) +
  ggtitle("Distribution of spend") +
  labs(y="Percentage of Customers", x="Spent value") +
  geom_text(aes(label=after_stat('count / sum(count) * 100')), 
            stat="count", 
            format_string='{:.1f}%',
            va='bottom'))
histspent.show()

#Income distribution:  
histincfin = (ggplot(FinancialsBasetable, aes(x='Income')) +
    theme_bw() +
    geom_bar(alpha=0.6) +
    ggtitle("Distribution of Income - Transactions") +
    labs(y="Percentage of Customers", x="Income level") +
    geom_text(aes(label=after_stat('count / sum(count) * 100')), 
              stat="count", 
              format_string='{:.1f}%', 
              va='bottom'))
histincfin.show()



#Lifecycle Grids:

#Converting to categorical and reversing order
SessionsBasetable['segmfreq'] = pd.Categorical(SessionsBasetable['segmfreq'], 
                                              categories=freq_labels[::-1])
FinancialsBasetable['segmfreq'] = pd.Categorical(FinancialsBasetable['segmfreq'], 
                                             categories=freq_labels_fin[::-1])
SessionsBasetable['segmrec'] = pd.Categorical(SessionsBasetable['segmrec'], 
                                              categories=rec_labels[::-1])
FinancialsBasetable['segmrec'] = pd.Categorical(FinancialsBasetable['segmrec'], 
                                             categories=rec_labels[::-1])
SessionsBasetable['segmage'] = pd.Categorical(SessionsBasetable['segmage'], 
                                              categories=age_labels[::-1])
FinancialsBasetable['segmage'] = pd.Categorical(FinancialsBasetable['segmage'], 
                                            categories=age_labels[::-1])
FinancialsBasetable['value_spent'] = pd.Categorical(FinancialsBasetable['value_spent'], 
                                             categories=spend_labels_fin[::-1])


#Sessions Lifecycle Grids

#Basic RF Quantity Grid
lcg = SessionsBasetable.groupby(['segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='client', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    geom_text(aes(y='number_customers', label='number_customers'), size=7) +
    ggtitle("RF Grid - Player Distribution - Sessions")+
    labs(x=" ", y="Number of Customers"))
rfgrid.show()

#Basic RF Quantity Grid + gender
lcg = SessionsBasetable.groupby(['Gender','segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='Gender', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    geom_text(aes(y='number_customers', label='number_customers'), size=7) +
    ggtitle("RF Grid - Gender & Player Distribution - Sessions")+
    labs(y="Number of Customers"))
rfgrid.show()

#Basic RF Quantity Grid with player types (bars)
lcg = SessionsBasetable.groupby(['CustomerType','segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='CustomerType', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player type Distribution - Sessions")+
    labs(x="Customer Type - 0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'", y="Number of Customers"))
rfgrid.show()


#RF Grid by Player Type:
total_per_segment = SessionsBasetable.groupby(['segmfreq', 'segmrec'], observed=True, as_index=False).agg(
    total=('CustomerID', 'count'))

lcg = SessionsBasetable.groupby(['segmfreq', 'segmrec', 'PlayerType'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

lcg = lcg.merge(total_per_segment, on=['segmfreq', 'segmrec'])
lcg['percentage'] = (lcg['quantity'] / lcg['total']) * 100

rfgrid = (ggplot(lcg, aes(x='1', y='percentage', fill='PlayerType')) +  # Use '1' as a dummy x-axis value
    theme_bw() +
    theme(panel_grid=element_blank(), axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +  
    geom_text(aes(label=round(lcg['percentage'], 1)), size=7, position=position_stack(vjust=0.5)) +  
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player type Distribution % - Sessions") +
    labs(x=" ", y="Percentage of Player Types"))
rfgrid.show()

#Seniority graph with %s for simpler interpretation
total_per_segment = SessionsBasetable.groupby(['segmfreq', 'segmrec'], observed=True, as_index=False).agg(
    total=('CustomerID', 'count'))

lcg = SessionsBasetable.groupby(['segmfreq', 'segmrec', 'seniority_segment'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

lcg = lcg.merge(total_per_segment, on=['segmfreq', 'segmrec'])
lcg['percentage'] = (lcg['quantity'] / lcg['total']) * 100

rfgrid = (ggplot(lcg, aes(x='1', y='percentage', fill='seniority_segment')) +  # Use '1' as a dummy x-axis value
    theme_bw() +
    theme(panel_grid=element_blank(), axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +  
    geom_text(aes(label=round(lcg['percentage'], 1)), size=7, position=position_stack(vjust=0.5)) +  
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player Seniority % Distribution - Sessions") +
    labs(x="", y="Percentage of Seniority Segments"))
rfgrid.show()

#Seniority grids & player types
lcg = SessionsBasetable.groupby(['CustomerType', 'segmfreq', 'segmrec', 'seniority_segment'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

rfgrid = (ggplot(lcg, aes(x='CustomerType', y='quantity', fill='seniority_segment')) +
    theme_bw() +
    theme(panel_grid=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player Distribution by Seniority & Player type - Sessions") +
    labs(x="Customer Type - 0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'", y="Number of Customers"))
rfgrid.show()



#Financials Lifecycle grids

#Basic RF Quantity Grid
lcg = FinancialsBasetable.groupby(['segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='client', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    geom_text(aes(y='number_customers', label='number_customers'), size=7) +
    ggtitle("RF Players Grid - Transactions")+
    labs(x=" ", y="Number of Customers"))
rfgrid.show()

#Basic RF Quantity Grid + gender
lcg = FinancialsBasetable.groupby(['Gender','segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='Gender', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    geom_text(aes(y='number_customers', label='number_customers'), size=7) +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grids - Gender & Player Distribution - Transactions") +
    labs(x="Gender", y="Number of Customers"))
rfgrid.show()

# Basic RF Quantity Grid + player type
lcg = FinancialsBasetable.groupby(['CustomerType','segmfreq','segmrec'], observed=True, as_index=False).agg(
    number_customers=('CustomerID','count'))
lcg['client'] = 'client'

rfgrid = (ggplot(lcg, aes(x='CustomerType', y='number_customers', fill='number_customers')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity', alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player type Distribution - Transactions")+
    labs(x="Customer Type - 0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'", y="Number of Customers"))
rfgrid.show()

#RF Quantity Grid with % of player types
total_per_segment = FinancialsBasetable.groupby(['segmfreq', 'segmrec'], observed=True, as_index=False).agg(
    total=('CustomerID', 'count'))

lcg = FinancialsBasetable.groupby(['segmfreq', 'segmrec', 'PlayerType'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

lcg = lcg.merge(total_per_segment, on=['segmfreq', 'segmrec'])
lcg['percentage'] = (lcg['quantity'] / lcg['total']) * 100

rfgrid = (ggplot(lcg, aes(x='1', y='percentage', fill='PlayerType')) +  # Use '1' as a dummy x-axis value
    theme_bw() +
    theme(panel_grid=element_blank(), axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +  
    geom_text(aes(label=round(lcg['percentage'], 1)), size=7, position=position_stack(vjust=0.5)) +  
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player type Distribution % - Transactions")+
    labs(x="", y="Percentage of Player types"))
rfgrid.show()

#Percentage seniority graph with financials
total_per_segment = FinancialsBasetable.groupby(['segmfreq', 'segmrec'], observed=True, as_index=False).agg(
    total=('CustomerID', 'count'))

lcg = FinancialsBasetable.groupby(['segmfreq', 'segmrec', 'seniority_segment'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

lcg = lcg.merge(total_per_segment, on=['segmfreq', 'segmrec'])
lcg['percentage'] = (lcg['quantity'] / lcg['total']) * 100

rfgrid = (ggplot(lcg, aes(x='1', y='percentage', fill='seniority_segment')) +  # Use '1' as a dummy x-axis value
    theme_bw() +
    theme(panel_grid=element_blank(), axis_text_x=element_blank(), axis_ticks_major_x=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +  
    geom_text(aes(label=round(lcg['percentage'], 1)), size=7, position=position_stack(vjust=0.5)) +  
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player Seniority Distribution % - Transactions") +
    labs(x="", y="Percentage of Seniority Segments"))
rfgrid.show()

#Seniority grids & player types
lcg = FinancialsBasetable.groupby(['CustomerType', 'segmfreq', 'segmrec', 'seniority_segment'], observed=True, as_index=False).agg(
    quantity=('CustomerID', 'count'))

rfgrid = (ggplot(lcg, aes(x='CustomerType', y='quantity', fill='seniority_segment')) +
    theme_bw() +
    theme(panel_grid=element_blank(), plot_title=element_text(size=10)) +  
    geom_bar(stat='identity', alpha=0.6, position='stack') +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("RF Grid - Player Distribution by Seniority & Player type - Transactions") +
    labs(x="Customer Type - 0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'", y="Number of Customers"))
rfgrid.show()

#Income distribution
lcg_income_fin = FinancialsBasetable.groupby(['CustomerType','Income', 'segmfreq','segmrec'], 
                                        observed=True, as_index=False).agg(
    quantity=('CustomerID','count'))
lcg_income_fin['client'] = 'client'

income_grid_fin = (ggplot(lcg_income_fin, aes(x='CustomerType', y='quantity', fill='Income')) +
    theme_bw() +
    theme(panel_grid = element_blank(), plot_title=element_text(size=10)) +
    geom_bar(stat='identity',  alpha=0.6) +
    facet_grid('segmfreq ~ segmrec') +
    ggtitle("LCG by Income (proportion)") +
    labs(x="Customer Type - 0: 'Walker', 1: 'Misc', 2: 'Social', 3: 'Catcher'", y="Number of Customers"))
income_grid_fin.show()




'''Question 3: churn analysis'''

#Putting back the numerical data
gender_mapping = {'Male' : 0, 'Female' : 1} 
SessionsBasetable['Gender'] = SessionsBasetable['Gender'].map(gender_mapping)
FinancialsBasetable['Gender'] = FinancialsBasetable['Gender'].map(gender_mapping)
SessionsBasetable['Gender'] = SessionsBasetable['Gender'].astype(int)
FinancialsBasetable['Gender'] = FinancialsBasetable['Gender'].astype(int)
income_mapping = {'Low' : 0, 'Medium' : 1, 'High' : 2}
SessionsBasetable['Income'] = SessionsBasetable['Income'].map(income_mapping)
FinancialsBasetable['Income'] = FinancialsBasetable['Income'].map(income_mapping)
SessionsBasetable['Income'] = SessionsBasetable['Income'].astype(int)
FinancialsBasetable['Income'] = FinancialsBasetable['Income'].astype(int)
SessionsBasetable = SessionsBasetable.drop(columns=['Days_Joined', 'Years_Joined']) #dropping overly correlated columns


#Creating the base table to perform the churn analysis
#Group by the sum_value fall by customer
sum_value_fall=FallFinTRX.groupby('CustomerID').agg(monetary_value=('Value','sum'))
#Merged summer and fall finance tables
FinancialsBasetableChurn = FinancialsBasetable.copy()
FinancialsBasetableChurn = FinancialsBasetableChurn.drop(columns='CustomerID')
ChurnBasetable=FinancialsBasetableChurn.merge(sum_value_fall, how='left', left_on='CustomerID', right_on='CustomerID')
ChurnBasetable['monetary_value']=ChurnBasetable['monetary_value'].fillna(0)
#Churn = played and payed and summer but did not payed in in fall
ChurnBasetable['churn'] = ChurnBasetable['monetary_value'].apply(lambda x: 1 if x == 0 else 0)
#Calculating the number of churn
ChurnBasetable['churn'].sum()

#Calculating Churn rate
churnrate = round((ChurnBasetable['churn'].sum())/(ChurnBasetable['Registrationdate'].count())*100,2)

#Merging the churn with the database table, first dropping the columns that are repeated
ChurnBasetable=ChurnBasetable.drop(columns=['fallbonus','Registrationdate','Gender','Age','Income','CustomerType','monetary_value'])

#renaming Date column of transaction not to mix them
ChurnBasetable = ChurnBasetable.rename(columns={'Recency': 'Recency(financial)'})

SessionsBasetable2 = SessionsBasetable.copy()
SessionsBasetable2 = SessionsBasetable2.drop(columns='CustomerID')
ChurnTotalbase=SessionsBasetable2.merge(ChurnBasetable, how='left', left_on='CustomerID', right_on='CustomerID')
#Deleting nan to only keep the ones relevant for the analysis
ChurnTotalbase = ChurnTotalbase.dropna(subset=['churn'])


#renaming recency column of transaction not to mix them
ChurnTotalbase = ChurnTotalbase.rename(columns={'Recency': 'Recency(sessions)'})

#First drop registration date
ChurnTotalbase.drop('Registrationdate', axis=1, inplace=True)
ChurnTotalbase = ChurnTotalbase.drop(columns=['Years_Joined', 'AOV', 
                                        'Annual_Purchase_Frequency', 'Annual_Revenue', 
                                        'Annual_Margin', 'Annual_CLV'])

#Finding the CLV from the financial table 
CustomerCLV = FinancialsBasetable
#contains CLV related information
print(CustomerCLV.info())
CustomerCLV = CustomerCLV.drop(columns='CustomerID')

CustomerCLV2=CustomerCLV.groupby('CustomerID').agg(Annual_CLV=('Annual_CLV','sum'))
clvpayers = CustomerCLV2['Annual_CLV'].mean()
#CLV of the player paid in summer = 7

'''Descriptive statistics'''

ChurnTotalbase = ChurnTotalbase.select_dtypes(exclude=['object', 'category'])
# Compute overall mean
total_stats = ChurnTotalbase.mean()

# Compute mean by churn group
churn_stats = ChurnTotalbase.groupby('churn').mean()

# Combine into a single DataFrame
summary_table = pd.DataFrame({
    'Overall Mean': total_stats,
    'Churn = 1 Mean': churn_stats.loc[1],
    'Churn = 0 Mean': churn_stats.loc[0]
})

# Display the table
print(summary_table)
summary_table.to_excel("summary_table.xlsx", index=False)

#To better understand the variables, we run a statistical test, to formally test if a feature is significantly different between churners and non-churner
from scipy.stats import ttest_ind

# Compare means for a few key features
features_to_test = ChurnTotalbase.drop(columns='churn')  # Replace with relevant features

for feature in features_to_test:
    churned = ChurnTotalbase[ChurnTotalbase['churn'] == 1][feature]
    non_churned = ChurnTotalbase[ChurnTotalbase['churn'] == 0][feature]

    # Perform t-test
    stat, p_value = ttest_ind(churned, non_churned, equal_var=False)
    
    print(f"{feature}: t-statistic = {stat:.4f}, p-value = {p_value:.4f}")
    
#Results: Gender, age and Recency (financial) are not significantly different


#Descritive statistics by customer type churners
#Churn by customer type
churners_customer_type = ChurnTotalbase.groupby('CustomerType').mean()


# filter churners
churners_df = ChurnTotalbase[ChurnTotalbase['churn'] == 1]

churners_avg_by_type = churners_df.groupby('CustomerType').mean()

# Display the table
print(churners_avg_by_type)

# filter no churners
no_churners_df = ChurnTotalbase[ChurnTotalbase['churn'] == 0]
no_churners_avg_by_type = no_churners_df.groupby('CustomerType').mean()


#Calculate the churn percentage for fall bonus
churn_percentage_bonus = ChurnTotalbase.groupby('fallbonus')['churn'].mean() * 100
print(churn_percentage_bonus)

'''Logistic regression using backward selection for more important variables'''
#Creating the dummies for the non-categorical variables

ChurnTotalbase2 = ChurnTotalbase.copy()
ChurnTotalbase2 = pd.get_dummies(ChurnTotalbase2, columns=['Income', 'CustomerType'], drop_first=True)
ChurnTotalbase2 = ChurnTotalbase2.astype(float)

#Sample creation
train, test = train_test_split(ChurnTotalbase2, test_size=0.3,random_state=1, stratify=ChurnTotalbase2['churn'])

# Exploratory analysis
corrmatrixtrain=train.corr()
corrmatrixtest=test.corr()
sns.heatmap(train.corr())

#Defining the independant and dependant variables
X=train.drop(columns='churn')
Y=train['churn']

#Stepwise function to find the most relevant independant variables
bidirectional=stepwise.BidirectionalStepwiseSelection(X, Y,model_type ="logistic",elimination_criteria = "aic", varchar_process = "dummy_dropfirst", senter=0.05, sstay=0.05)
print(bidirectional[2].summary())

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Extract the coefficients
coef = bidirectional[2].params

# Create a DataFrame with coefficients
coef_df = pd.DataFrame({'Feature Importance': coef.abs()})

# Drop the intercept if it exists
coef_df = coef_df.drop('intercept', errors='ignore')

# Sort by importance in descending order (from highest to lowest)
coef_df = coef_df.sort_values(by='Feature Importance', ascending=False)

# Plot feature importance (with features on the y-axis)
plt.figure(figsize=(8, 6))  # Adjust the size of the figure

ax = sns.barplot(
    x='Feature Importance', 
    y=coef_df.index, 
    data=coef_df, 
    color='blue',  # Set bars to blue
    width=0.5  # Adjust the bar width to make them thinner
)

# Add labels and title
plt.title('Feature Importance (Logistic Regression)', fontsize=14)
plt.xlabel('Importance', fontsize=12)
plt.ylabel('Features', fontsize=12)

# Add tags (values) on top of the bars
for p in ax.patches:
    ax.annotate(f'{p.get_width():.2f}',  # Format to two decimal places
                (p.get_width(), p.get_y() + p.get_height() / 2),  # Position the label at the bar end
                xytext=(5, 0),  # Adjust text position slightly
                textcoords='offset points', 
                ha='left', va='center', fontsize=10, color='black')

plt.tight_layout()  # Adjust layout to ensure the chart fits

# Display the plot
plt.show()


Xtrain=train[np.intersect1d(train.columns,bidirectional[0])]#only select var retained from backbward selection
Xtrain=Xtrain.reindex(columns=bidirectional[0])#sort them according to the sequence used in the regression
Xtrain['intercept']=1#add constant variable
train['predict']=bidirectional[2].predict(Xtrain)#store prediction in corresponding dataset

Xtest=test[np.intersect1d(test.columns,bidirectional[0])]#only select var retained from backbward selection
Xtest=Xtest.reindex(columns=bidirectional[0])#sort them according to the sequence used in the regression
Xtest['intercept']=1#add constant variable
test['predict']=bidirectional[2].predict(Xtest)#store prediction in corresponding dataset



#confusion matrix related performance measures
def evaluate_model(actual, pred,cutoff=0.5):
    #confusion matrix
    cm_df = pandas.DataFrame(confusion_matrix(actual,pred>cutoff))
    cm_df.columns = ['Predicted 0', 'Predicted 1']
    cm_df = cm_df.rename(index={0: 'Actual 0',1: 'Actual 1'})
    print("Cutoff used: ",cutoff)
    print(cm_df)
    # Accuracy Score
    acc_score = round(accuracy_score(actual, pred>cutoff)*100,2)
    print('Accuracy Score : ',acc_score)
    # ROC AUC score
    roc_score = round(roc_auc_score(actual, pred)*100,2)
    print('ROC AUC score : ',roc_score)
    # Precision score
    prec_score = round(precision_score(actual, pred>cutoff)*100,2)
    print('Precision score : ', prec_score)
    # Recall score
    sens_score = round(recall_score(actual, pred>cutoff)*100,2)#recall=sensitivity
    print('sensitivity score : ', sens_score)
    tn, fp, fn, tp = confusion_matrix(actual,pred>cutoff).ravel()
    spec_score = round((tn / (tn + fp))*100,2)
    print('specificity score : ', spec_score)
    fpr, tpr, _ = roc_curve(actual,pred)
    auc = roc_auc_score(actual,pred)
    roccurve=(pn.ggplot(pn.aes(x=fpr,y=tpr))+
                 pn.geom_line(color='red')+
                 pn.scale_y_continuous(name='sensitivity,true positive rate' ) +
                 pn.scale_x_continuous(name='1-specificity, false positive rate' ) +
                 pn.ggtitle('Roc curve: '+str(np.round(auc,2))))
    print(roccurve)
    return acc_score, roc_score, prec_score, sens_score, spec_score

performance=evaluate_model(train['churn'],train['predict'])
performance=evaluate_model(test['churn'],test['predict'])

#balanced dataset function
#is this a balanced dataset?
print('percentage churn in train dataset: ',np.round(np.mean(train['churn'])*100,2))
print('percentage churn in test dataset: ',np.round(np.mean(test['churn'])*100,2))
#No, but churn is not a marginal effect here

#how does the accuracy change for different cutoff values
def AccuracyCutoffInfo(training,validation,depvar='churn',pred='predict'):
    # change the cutoff value's range as you please
    cutoff=np.arange(0.1, 0.85, 0.05)
    cm_train=[0]*len(cutoff)
    cm_validation=[0]*len(cutoff)
    for i in range(0,len(cutoff)):  
        cm_train[i]=accuracy_score(training[depvar], training[pred]>cutoff[i])
        cm_validation[i]=accuracy_score(validation[depvar], validation[pred]>cutoff[i])
    accuracytable=pandas.concat([pandas.Series(cutoff).rename('cutoff'),pandas.Series(cm_train).rename('train'),pandas.Series(cm_validation).rename('validation')],axis=1)    
    accplot=(pn.ggplot(accuracytable, pn.aes(x='cutoff')) + 
    pn.geom_line( pn.aes(y = 'train'), color = "darkred")+
    pn.geom_line( pn.aes(y = 'validation'), color = "steelblue")+
    pn.geom_point(pn.aes(y = 'train'))+
    pn.geom_point(pn.aes(y = 'validation'))+
    pn.scale_y_continuous( labels=percent_format() ) +
    pn.ggtitle( "Train/validation accuracy for different cutoff values" ))
    accplot.show()
  
AccuracyCutoffInfo(train,test,depvar='churn',pred='predict')

#confusionmatrixscatterplot
#create classification table first
def confusionmatrixplot(depvar,pred,cutoff=0.5):
    df=pandas.concat([pandas.Series(depvar).rename('depvar'),pandas.Series(pred).rename('pred')],axis=1)
    def categorize(depvar,pred,cutoff):
        if (pred>=cutoff and depvar==1):
            return 'TP'
        elif (pred>=cutoff and depvar==0):
            return 'FP'
        elif(pred<cutoff and depvar==1):
            return 'FN'
        else:
            return 'TN'
    df['type']=df.apply(lambda X: categorize(X.depvar,X.pred,cutoff), axis=1)
    #plot table
    confusionplot= (pn.ggplot(df, pn.aes(x='depvar',y='pred', color = 'type' ) ) + 
        pn.geom_jitter() + 
        pn.geom_hline( yintercept = cutoff, color = "blue", alpha = 0.6 ) + 
        pn.scale_y_continuous( limits = (0.0,1.0) ) + 
        pn.scale_color_discrete( breaks = [ "TP", "FN", "FP", "TN" ] ) + # ordering of the legend 
        pn.ggtitle("Confusion Matrix with cutoff: "+str(cutoff)))
    confusionplot.show()

confusionmatrixplot(test['churn'],test['predict'],cutoff=0.5)

#find the cutoff that leads to the lowest costs
def findoptcutoff(depvar,pred):
    cutoff=np.arange(0.01, 0.99, 0.01)
    cost_fp=2.99#providing a discount to a non churner
    cost_fn= 7 #not identifying a churner and missing out on his clv
    costs=[0]*len(cutoff)
    bestcost=sys.maxsize
    bestindex=0 
    for i in range(0,len(cutoff)): 
        cm_df = pandas.DataFrame(confusion_matrix(depvar,pred>cutoff[i]))
        am_fp=cm_df[1][0]
        am_fn=cm_df[0][1]
        costs[i]=am_fp*cost_fp+am_fn*cost_fn
        if(costs[i]<bestcost):
            bestcost=costs[i]
            bestindex=i       
    cutofftable=pandas.concat([pandas.Series(cutoff).rename('cutoff'),pandas.Series(costs).rename('costs')],axis=1)
    cutoffchart=(pn.ggplot(cutofftable, pn.aes(x = 'cutoff', y = 'costs',color='costs'))+
       pn.theme_bw() +
       pn.theme(panel_grid = pn.element_blank())+
       pn.geom_point()+
       pn.scale_colour_gradient2(low = "green", high = "red", mid = "yellow",midpoint=8000)+
      pn. ggtitle("Total cost for different cutoff values"))
    cutoffchart.show()
    print('best cutoff: ', cutoff[bestindex])
    return cutoff[bestindex]

optcutoff=findoptcutoff(test['churn'],test['predict'])
#optimal cutoff is 0.32

#now that we now our optimal cut off let's calculate the performance measures again
performance=evaluate_model(test['churn'],test['predict'],cutoff=optcutoff)

#confusion matrix for cutoff 0.32
confusionmatrixplot(test['churn'],test['predict'],cutoff=optcutoff)

#decile/lift table and chart
def calc_lift(depvar,pred,groups=10):
        helper=pandas.concat([pandas.Series(depvar).rename('depvar'),pandas.Series(pred).rename('pred')],axis=1)
        #reverse sort dataset based on pred
        helper=helper.sort_values(by=['pred'],ascending=False)
        helper['id']=range(0,len(helper['depvar']))
        #create new bucket variable highest pred in bucket 1
        helper['bucket']=pandas.qcut(helper['id'],q=10,labels=range(1, 10+1))
        #convert categorical bucket variable into a numeric one
        helper['bucket']=pandas.to_numeric(helper['bucket'])
        #create gaintable: count number of churners per bucket and total customers in a bucket
        lifttable=helper.groupby("bucket", as_index=False).agg(total=('depvar','count'),totalresp=('depvar','sum'))
        #calculate cumulative number of churners
        lifttable['cumresp']=lifttable['totalresp'].cumsum()
        #calculate lift: how many churners did you catch out of the total number of customers in a bucket
        lifttable['lift']=lifttable['cumresp']/sum(lifttable['totalresp'])*100
        #calculate cumulative lift
        lifttable['cumlift']=lifttable['lift']/(lifttable['bucket']*(100/groups))
        return lifttable

def plot_liftchart(depvar,pred):
    lifttable=calc_lift(depvar,pred)
    liftchart=(pn.ggplot(lifttable, pn.aes(x = 'bucket', y = 'cumlift'))+
      pn.theme_bw() +
      pn.theme(panel_grid = pn.element_blank())+#removes colored box behind bar chart
      pn.scale_x_continuous(breaks=range(1,11)) +
      pn.geom_point()+
      pn.geom_smooth()+
      pn.geom_text(pn.aes(x='bucket+0.5',y='cumlift', label='np.round(cumlift,2)')) +
      pn.ggtitle("Lift curve"))
    liftchart.show()
    return lifttable 
    
lifttabletest=plot_liftchart(test['churn'],test['predict'])
