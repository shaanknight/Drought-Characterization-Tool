import pandas as pd
from climate_indices import compute, indices
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import datetime
from scipy.stats import kstest, gamma, pearson3, logistic

class Drought:

    def __init__(self):
        self.data_year_start_monthly = 1995
        self.data_year_end_monthly = 2017

    def set_discharge(self, df):
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']],errors='coerce')
        df = df.set_index(df['Date'])
        df = df.drop(df.columns[[0,1,2,4]], axis=1)
        df = df.fillna(0)
        self.df_discharge = df

    def set_precip(self, df):
        df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']],errors='coerce')
        df = df.set_index(df['Date'])
        df = df.drop(df.columns[[0,1,2,4]], axis=1)
        df = df.fillna(0)
        df = df.replace(0, df.mean())
        self.df_precip = df

    def get_discharge(self,start,end):
        df_discharge = self.df_discharge.loc[start: end]
        print(start, end)
        df_discharge_monthly_mean = df_discharge['Item'].resample('M').mean()
        dis = np.nan_to_num(df_discharge_monthly_mean)
        return dis

    def get_precip(self,start,end):
        df_precip = self.df_precip.loc[start: end]
        df_precip_monthly_mean = df_precip['Data'].resample('M').mean()
        precip = np.nan_to_num(df_precip_monthly_mean)
        return precip

    def get_yearly_discharge(self,start,end):
        df_discharge = self.df_discharge.loc[start: end]
        df_discharge_yearly_mean = df_discharge['Item'].resample('A-MAY').sum()
        dis = np.nan_to_num(df_discharge_yearly_mean)
        return dis

    def get_yearly_precip(self,start,end):
        df_precip = self.df_precip.loc[start: end]
        df_precip_yearly_mean = df_precip['Data'].resample('A-MAY').sum()
        precip = np.nan_to_num(df_precip_yearly_mean)
        return precip

    def get_dates(self,start,end):
        Period = pd.date_range(start=start, end=end, freq='M')
        return Period.strftime('%Y-%m').tolist()

    def get_yearly_dates(self,start,end):
        Period = pd.date_range(start=start, end=end, freq='Y')
        return Period.strftime('%Y').tolist()

    def daily_analysis(data):
        Discharge_datewise = pd.DataFrame(data, columns=['Date', 'Discharge'])
        date_discharge = ['Date','Discharge']
        Discharge_datewise = Discharge_datewise[date_discharge]
        #Discharge_datewise.Date = pd.to_datetime(Discharge_datewise.Date)
        Discharge_datewise.plot(x='Date')
        pyplot.xlabel("Daywise Variation")
        pyplot.ylabel("Discharge")
        pyplot.show()

    def monthly_analysis(data):
        Discharge_datewise = pd.DataFrame(data, columns=['Date', 'Discharge'])
        date_discharge = ['Date','Discharge']
        Discharge_datewise = Discharge_datewise[date_discharge]
        Discharge_datewise.set_index('Date', inplace=True)
        Discharge_datewise.index = pd.to_datetime(Discharge_datewise.index)
        Monthly_data = Discharge_datewise.resample('1M').mean()
        Monthly_data.plot()
        pyplot.xlabel("Monthwise Variation")
        pyplot.ylabel("Average Discharge")
        pyplot.show()

    def seasonal_analysis(data):
        dfDischarge = pd.DataFrame(data, columns=['Date', 'Discharge'])
        dfDischarge['Day'] = dfDischarge['Date'].apply(lambda row : row.day)
        dfDischarge['Month'] = dfDischarge['Date'].apply(lambda row : row.month)
        dfDischarge['Year'] = dfDischarge['Date'].apply(lambda row : row.year)
        cols = ['Year', 'Month', 'Day', 'Discharge']
        dfDischarge = dfDischarge[cols]
        x_season = []
        y_season = []
        x_values = []
        c = 0
        days_count = 0
        x_season.append("1980_Monsoon")
        x_values.append(c+1)
        y_season.append(0)
        for index, row in dfDischarge.iterrows():
          if row['Month'] == 6.0 and row['Day'] == 1.0 and days_count > 0:
            y_season[c] = y_season[c]/days_count
            x_season.append(str(int(row['Year']))+"_Monsoon")
            y_season.append(0)
            days_count = 0
            c += 1
            x_values.append(c+1)
          elif row['Month'] == 10.0 and row['Day'] == 1.0:
            y_season[c] = y_season[c]/days_count
            x_season.append(str(int(row['Year']))+"_Non_Monsoon")
            y_season.append(0)
            days_count = 0
            c += 1
            x_values.append(c+1)
          y_season[c] += row['Discharge']
          days_count += 1
        pyplot.plot(y_season)
        pyplot.xticks(x_values[::5],x_season[::5],rotation ='vertical') 
        pyplot.legend(['Discharge']) 
        pyplot.xlabel("Season wise variation - Monsoon and Non-Monsoon")
        pyplot.ylabel("Average Discharge")
        pyplot.show()

    def yearly_analysis(data):
        Discharge_datewise = pd.DataFrame(data, columns=['Date', 'Discharge'])
        Yearly_data = Discharge_datewise.resample('1Y').mean()
        Yearly_data.plot()
        pyplot.xlabel("Yearwise Variation")
        pyplot.ylabel("Average Discharge")
        pyplot.show()

    def test_hypothesis_pearson(self, start, end):
        args = pearson3.fit(self.get_discharge(start,end))
        return kstest(self.get_discharge(start,end), "pearson3", args=args)

    def test_hypothesis_gamma(self, start, end):
        fit_alpha, fit_loc, fit_beta=gamma.fit(self.get_discharge(start,end))
        return kstest(self.get_discharge(start,end), 'gamma', args=(fit_alpha, fit_loc, fit_beta))

    def discharge_test_pearson(self, start, end):
        args = pearson3.fit(self.get_discharge(start,end))
        return kstest(self.get_discharge(start,end), "pearson3", args=args)

    def discharge_test_gamma(self, start, end):
        fit_alpha, fit_loc, fit_beta=gamma.fit(self.get_discharge(start,end))
        return kstest(self.get_discharge(start,end), 'gamma', args=(fit_alpha, fit_loc, fit_beta))

    def discharge_test_logistic(self, start, end):
        args = logistic.fit(self.get_discharge(start,end))
        return kstest(self.get_discharge(start,end), "logistic", args=args)

    def precip_test_pearson(self, start, end):
        args = pearson3.fit(self.get_precip(start,end))
        return kstest(self.get_precip(start,end), "pearson3", args=args)

    def precip_test_gamma(self, start, end):
        fit_alpha, fit_loc, fit_beta=gamma.fit(self.get_discharge(start,end))
        return kstest(self.get_precip(start,end), 'gamma', args=(fit_alpha, fit_loc, fit_beta))

    def precip_test_logistic(self, start, end):
        args = logistic.fit(self.get_precip(start,end))
        return kstest(self.get_precip(start,end), "logistic", args=args)

    def get_indices(self, start, end, period):
        stat_p, pval_p = self.test_hypothesis_pearson(start, end)
        stat_g, pval_g = self.test_hypothesis_gamma(start, end)
        print("pearson:", stat_p, pval_p)
        print("gamma:", stat_g, pval_g)
        if pval_g > pval_p:
            print("using gamma")
            sdi = indices.spi(self.get_discharge(start,end),int(period),indices.Distribution.gamma,self.data_year_start_monthly,self.data_year_start_monthly,self.data_year_end_monthly,compute.Periodicity.monthly)
            spi = indices.spi(self.get_precip(start,end),int(period),indices.Distribution.gamma,self.data_year_start_monthly,self.data_year_start_monthly,self.data_year_end_monthly,compute.Periodicity.monthly)
        else:
            print("using pearson3")
            sdi = indices.spi(self.get_discharge(start,end),int(period),indices.Distribution.pearson,self.data_year_start_monthly,self.data_year_start_monthly,self.data_year_end_monthly,compute.Periodicity.monthly)
            spi = indices.spi(self.get_precip(start,end),int(period),indices.Distribution.pearson,self.data_year_start_monthly,self.data_year_start_monthly,self.data_year_end_monthly,compute.Periodicity.monthly)
        sdi, spi = np.nan_to_num(sdi), np.nan_to_num(spi)
        return sdi, spi, self.get_dates(start,end)
        # return self.get_dates(start,end), self.get_dates(start,end), self.get_dates(start,end)

    def get_periodic_indices(self, start, end):
        df_discharge, df_precip = self.df_discharge.loc[start: end], self.df_precip.loc[start: end]
        print(df_discharge)
        print(df_precip)
        df_discharge_monthly_sum = df_discharge['Item'].resample('M').mean()
        print(df_discharge_monthly_sum)
        df_discharge_yearly_sum = df_discharge_monthly_sum.resample('A-MAY').mean()
        print(df_discharge_yearly_sum)
        df_discharge_yearly_mean = df_discharge_monthly_sum.mean()
        df_discharge_yearly_SD = df_discharge_monthly_sum.std()
        df_precip_monthly_sum = df_precip['Data'].resample('M').mean()
        df_precip_yearly_mean = df_precip_monthly_sum.resample('A-MAY').mean()
        df_precip_monthly_SD = df_precip_monthly_sum.std()
        df_precip_monthly_mean = df_precip_monthly_sum.mean()

        SDI = (df_discharge_yearly_sum[:] - df_discharge_yearly_mean)/df_discharge_yearly_SD
        SPI = (df_precip_yearly_mean[:] - df_precip_monthly_mean)/df_precip_monthly_SD

        SPI = SPI.fillna(SPI.mean())

        return list(SDI.values),list(SPI.values),self.get_yearly_dates(start,end)

    def get_yearly_indices(self, start, end):
        df_discharge, df_precip = self.df_discharge.loc[start: end], self.df_precip.loc[start: end]
        print(df_discharge)
        print(df_precip)
        df_discharge_monthly_sum = df_discharge['Item'].resample('M').mean()
        print(df_discharge_monthly_sum)
        df_discharge_yearly_sum = df_discharge_monthly_sum.resample('A-MAY').mean()
        print(df_discharge_yearly_sum)
        df_discharge_yearly_mean = df_discharge_monthly_sum.mean()
        df_discharge_yearly_SD = df_discharge_monthly_sum.std()
        df_precip_monthly_sum = df_precip['Data'].resample('M').mean()
        df_precip_yearly_mean = df_precip_monthly_sum.resample('A-MAY').mean()
        df_precip_monthly_SD = df_precip_monthly_sum.std()
        df_precip_monthly_mean = df_precip_monthly_sum.mean()

        SDI = (df_discharge_yearly_sum[:] - df_discharge_yearly_mean)/df_discharge_yearly_SD
        SPI = (df_precip_yearly_mean[:] - df_precip_monthly_mean)/df_precip_monthly_SD

        SPI = SPI.fillna(SPI.mean())

        return list(SDI.values),list(SPI.values),self.get_yearly_dates(start,end)

def make_line_intensity(date, x1, x2, t):
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.xaxis.grid()

    ax.plot(x1, marker='o',markersize=0,label='SDI', lw=5,color='red')
    ax.plot(x2, marker='o',markersize=0,label='SPI', lw=5,color='green')
    
    ax.fill_between(date, 2, 3, color='#CCCCFF', label="Extreme wet")
    ax.fill_between(date, 1.5, 2, color='#CCFFFF', label="Very wet")
    ax.fill_between(date, 1, 1.5, color='#CCFFCC', label="Moderate wet")
    ax.fill_between(date, -1, 1, color='#E5FFCC', label="Normal")
    ax.fill_between(date, -1.5, -1, color='#FFFFCC', label="Moderate drought")
    ax.fill_between(date, -2, -1.5, color='#FFE5CC', label="Severe drought")
    ax.fill_between(date, -2, -3, color='#FFCCCC', label="Extreme drought")

    plt.axhline(y = t, color = 'black', linestyle = '-', label='Drought Threshold')

    plt.legend(loc='upper right',fontsize=20,ncol=3)
    plt.grid(True,linestyle='--')
    plt.xlabel('Corresponding dates', fontsize=20)
    plt.ylabel('Drought Index', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(-3.0, 3.0, 0.5))
    plt.xticks(rotation ='vertical') 
    plt.savefig('./static/plot_intensity.png', bbox_inches = 'tight', dpi = 300)

def make_line_frequency(date, x1, x2, t):
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.xaxis.grid()

    ax.plot(x1, marker='o',markersize=0,label='SDI', lw=5,color='red')
    ax.plot(x2, marker='o',markersize=0,label='SPI', lw=5,color='green')
    
    ax.fill_between(date, -1, 3, color='#E5FFCC', label="Rare drought Frequency")
    ax.fill_between(date, -1.5, -1, color='#FFFFCC', label="Moderate drought Frequency")
    ax.fill_between(date, -2, -1.5, color='#FFE5CC', label="Severe drought Frequency")
    ax.fill_between(date, -2, -3, color='#FFCCCC', label="Extreme drought Frequency")

    plt.axhline(y = t, color = 'black', linestyle = '-', label='Drought Threshold')

    plt.legend(loc='upper right',fontsize=20,ncol=3)
    plt.grid(True,linestyle='--')
    plt.xlabel('Corresponding dates', fontsize=20)
    plt.ylabel('Drought Index', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(-3.0, 3.0, 0.5))
    plt.xticks(rotation ='vertical') 
    plt.savefig('./static/plot_frequency.png', bbox_inches = 'tight', dpi = 300)

def make_line_duration(date, x1, x2, t):
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.xaxis.grid()

    ax.plot(x1, marker='o',markersize=0,label='SDI', lw=5,color='red')
    ax.plot(x2, marker='o',markersize=0,label='SPI', lw=5,color='green')
    plt.axhline(y = t, color = 'black', linestyle = '-', label='Drought Threshold')

    plt.legend(loc='upper right',fontsize=20,ncol=3)
    plt.grid(True,linestyle='--')
    plt.xlabel('Corresponding dates', fontsize=20)
    plt.ylabel('Drought Index', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(-3.0, 3.0, 0.5))
    plt.xticks(np.arange(1,len(x1),1),rotation ='vertical') 
    plt.savefig('./static/plot_duration.png', bbox_inches = 'tight', dpi = 300)

def make_line_duration_spi(date, x1, x2, t):
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.xaxis.grid()

    #ax.plot(x1, marker='o',markersize=0,label='SDI', lw=5,color='red')
    ax.plot(x2, marker='o',markersize=0,label='SPI', lw=5,color='green')
    plt.axhline(y = t, color = 'black', linestyle = '-', label='Drought Threshold')

    plt.legend(loc='upper right',fontsize=20,ncol=3)
    plt.grid(True,linestyle='--')
    plt.xlabel('Corresponding dates', fontsize=20)
    plt.ylabel('Drought Index', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(-3.0, 3.0, 0.5))
    plt.xticks(np.arange(1,len(x1),1),rotation ='vertical') 
    x = np.arange(0,len(x1),1)
    plt.fill_between(x,0,x2,facecolor = 'grey')
    plt.savefig('./static/plot_SPI_duration.png', bbox_inches = 'tight', dpi = 300)

def make_line_duration_sdi(date, x1, x2, t):
    fig, ax = plt.subplots(figsize=(50, 15))
    ax.xaxis.grid()

    ax.plot(x1, marker='o',markersize=0,label='SDI', lw=5,color='red')
    #ax.plot(x2, marker='o',markersize=0,label='SPI', lw=5,color='green')
    plt.axhline(y = t, color = 'black', linestyle = '-', label='Drought Threshold')

    plt.legend(loc='upper right',fontsize=20,ncol=3)
    plt.grid(True,linestyle='--')
    plt.xlabel('Corresponding dates', fontsize=20)
    plt.ylabel('Drought Index', fontsize=20)
    plt.tick_params(labelsize=20)
    plt.yticks(np.arange(-3.0, 3.0, 0.5))
    plt.xticks(np.arange(1,len(x1),1),rotation ='vertical') 
    x = np.arange(0,len(x1),1)
    plt.fill_between(x,0,x1,facecolor = 'grey')
    plt.savefig('./static/plot_SDI_duration.png', bbox_inches = 'tight', dpi = 300)