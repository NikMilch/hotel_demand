# hotel_demand
The purpose of the hotel demand dataset analysis is to describe the hotel occupancy during the year and the main reasons for booking cancelations.
mport numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap
from skimage import data, color

import datetime

import re

import nltk
from nltk.corpus import stopwords

from collections import Counter

import skimage.io
from scipy.ndimage.filters import convolve
from skimage.feature import canny

from skimage.feature import Cascade
from matplotlib import patches
from skimage import data


# ## HOTEL BOOKING DEMAND AND CANCELLATIONS
# 
# ### Hotel data visualization

# ### Author: Nikoleta Milcheva

# ### Abstract
# 
# The following exploratory data analysis visualization presents two datasets - one with the hotel demand data (Resorts and City hotels in Portugal) and the other with hotels reviews (original data from Yelp about hotels in Europe).
# 
# The purpose of the hotel demand dataset analysis is to describe the hotel occupancy during the year and the main reasons for booking cancelations. The datasets contain information about bookings that has arrived and bookings that has been canceled. 
# 
# The hotel reviews datasets analysis describes the best and the worst hotels in Europe. Also, it describes what influence on guest positive reviews. Picture of the best ranked hotel has been thresholded and face recognition has been implemented. 
# 
# The data analysis is based on the hotel real data and it can be used from hotel and revenue managers. 
# 

# ## Hotel bookings demand dataset

# ### Dataset Introduction
# 
# Number of the Resort Hotel in hotel demand dataset is double than the number of the City hotels. Resort hotels and City hotels has different number of guests, different prices and different cancelations during the year. The highest number of the guests is from Portugal and most of the bookings are made by online travel agent. The biggest influence of cancelations has lead time and previous cancelation. The purpose of the analysis is to show what is the trend based on three years data for bookings and cancellations and based on this analysis to be made prediction of what to expect in future year. This is the reason why cancelled bookings are not removed from the data.
# 

# ### Previous Works
# 
# Previous researches show that most of the guests stay in City hotels between 1 and 4 days. For the Resort hotels is also usual guests to stay 7 days. City hotels has higher percent of cancellations than resort hotels and the number of cancellations is more stable during the year. Resort Hotels has more cancelation during the summer period.
# 
# Highest adr has aviation room type E (economy). By segments higher number of cancellations has online TA, offline TA and groups. Reservation status is - higher number checked outs, followed by cancellations and the smallest number are No-showns.

# ### Data source:
# 
# Kaggle - Hotel Bookings Demand and Hotel reviews enriched
# https://www.kaggle.com/jessemostipak/hotel-booking-demand
# https://www.kaggle.com/ycalisar/hotel-reviews-dataset-enriched

# ### Initial Data Exploration

# #### 1. Reading and cleaning the data

# In[3]:


# Read the file csv

hotel_bookings = pd.read_csv("data/hotel_bookings.csv")

