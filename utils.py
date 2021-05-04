import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker


def make_line1(date, x1, x2):
    fig, ax = plt.subplots()
    line1 = ax.plot(date, x1, color='red', marker='o', label = 'Discharge')
    line2 = ax.plot(date, x2, color='blue', marker='o', label = 'Precipitation')
    plt.title('Monthly Discharge and Precipitation', fontsize=14)
    plt.xlabel('Month', fontsize=14)
    plt.ylabel('in mm', fontsize=14)
    plt.grid(True)
    plt.legend()
    #plt.xticks(rotation='vertical')
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 5))
    plt.xticks(rotation='vertical')
    plt.savefig('./static/plot.png', bbox_inches = 'tight', dpi = 300)

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

def make_bar1(date, x1, x2):
    barWidth = 0.1
    fig, ax = plt.subplots() 
    br1 = np.arange(len(x1)) 
    br2 = [x + barWidth for x in br1] 
    plt.bar(br1, x1, color ='r', width = barWidth, 
            edgecolor ='grey', label ='Discharge') 
    plt.bar(br2, x2, color ='g', width = barWidth, 
            edgecolor ='grey', label ='Precipitation') 
    plt.title('Yearly Discharge and Precipitation', fontsize=14)
    plt.xlabel('Year', fontweight ='bold') 
    plt.ylabel('in mm', fontsize=14)
    plt.legend()
    plt.savefig('./static/plot.png', bbox_inches = 'tight', dpi = 300)
