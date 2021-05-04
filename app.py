from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from drought_indices import Drought
import pandas as pd
from utils import *
import matplotlib.pyplot as plt
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


drought = Drought()

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

	
@app.route('/', methods = ['GET', 'POST'])
def home():
   if request.method == 'POST':
      pre_file = request.files['prec_file']
      # pre_file.save(secure_filename(pre_file.filename))

      dis_file = request.files['dis_file']
      # dis_file.save(secure_filename(dis_file.filename)) 
      
      time_checked_op = None
      drought_checked_op = None

      Time_Radio_clicked = request.form["time_options"]
      Drought_Radio_clicked = request.form["drought_options"]

      if Time_Radio_clicked == "Monthly Discharge":
      	time_checked_op = 0
      if Time_Radio_clicked == "Yearly Discharge":
         time_checked_op = 1

      if Drought_Radio_clicked == "Monthly SPI":
      	drought_checked_op = 0
      if Drought_Radio_clicked == "Yearly SPI":
      	drought_checked_op = 1

      interestperiod = request.form['interestperiod']

      start_date = request.form['start']
      temp = start_date.split("-")
      start_date = temp[0] +  "-" + temp[1]

      end_date = request.form['end']
      temp = end_date.split("-")
      end_date = temp[0] +  "-" + temp[1]

      D_threshold = int(request.form['threshold'])
      D_threshold = D_threshold

      # print(start_date,end_date,checked_op,dis_file,pre_file)

      dis_df = pd.read_csv(dis_file,index_col=0, parse_dates=True, squeeze=True)
      drought.set_discharge(dis_df)

      pre_df = pd.read_csv(pre_file,index_col=0, parse_dates=True, squeeze=True)
      drought.set_precip(pre_df)

      if time_checked_op == 0:
      	dis, pre = drought.get_discharge(start_date, end_date), drought.get_precip(start_date, end_date)
      	dates = drought.get_dates(start_date, end_date)
      	dis = dis[0:-1]
      	pre = pre[0:-1]
      	make_line1(dates, list(dis), list(pre))
      else:
      	dis, pre = drought.get_yearly_discharge(start_date, end_date), drought.get_yearly_precip(start_date, end_date)
      	dates = drought.get_yearly_dates(start_date, end_date)
      	dis = dis[0:-1]
      	pre = pre[0:-1]
      	for i in range(len(dis)):
      		dis[i] = dis[i] // 10
      		pre[i] = pre[i] // 10
      	print(dates, list(dis),list(pre))
      	make_bar1(dates, list(dis), list(pre))

      if drought_checked_op == 0:
         sdi, spi, dates = drought.get_indices(start_date, end_date, interestperiod)
         print(dates)
         sdi = sdi[0:-1]
         spi = spi[0:-1]
         make_line_intensity(dates, list(sdi), list(spi), D_threshold)
         make_line_frequency(dates, list(sdi), list(spi), D_threshold)
         make_line_duration_spi(dates, list(sdi), list(spi), D_threshold)
         make_line_duration_sdi(dates, list(sdi), list(spi), D_threshold)
      else:
         sdi, spi, dates = drought.get_yearly_indices(start_date, end_date)
         print(dates)
         sdi = sdi[0:-1]
         spi = spi[0:-1]
         make_line_intensity(dates, list(sdi), list(spi), D_threshold)
         make_line_frequency(dates, list(sdi), list(spi), D_threshold)
         make_line_duration_spi(dates, list(sdi), list(spi), D_threshold)
         make_line_duration_sdi(dates, list(sdi), list(spi), D_threshold)

      sdg,pdg = drought.discharge_test_gamma(start_date, end_date)
      spg,ppg = drought.precip_test_gamma(start_date, end_date)
      sdp,pdp = drought.discharge_test_pearson(start_date, end_date)
      spp,ppp = drought.precip_test_pearson(start_date, end_date)
      sdl,pdl = drought.discharge_test_logistic(start_date, end_date)
      spl,ppl = drought.precip_test_logistic(start_date, end_date)

      return render_template('home.html',sdg=round(sdg, 4),pdg=pdg,spg=round(spg, 4),ppg=ppg,sdp=round(sdp, 4),pdp=pdp,spp=round(spp, 4),ppp=ppp,sdl = round(sdl,4),pdl = pdl, spl = round(spl,4), ppl = ppl, display = "true")
   else:
   	return render_template('home.html',display = "false")
      


if __name__ == '__main__':
   app.run(debug = True)