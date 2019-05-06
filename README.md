---

# GENERAL ASSEMBLY CAPSTONE PROJECT  

<font size='3'>
DSI-7 San Francisco
Manu Kalia
</font>  

<br/>
Electricity Price Model Using Recurrent Neural Networks

Capstone Title:	Predicting California Hour-ahead Electricity Prices Based on Weather and  
Day-ahead Market Prices Using a Recursive Neural Network

Description:	A predictive analysis California ISO's locational marginal prices in the hour-ahead (HASP),  
and day-ahead (DAM) markets, and the best mix a power generator should use in selling energy into the CA grid.

Two supervised machine learning models... one each, taking either DAM or HASP as the dependant variable,  
with the other market price as one of several predictive features.  The other features include:  

 - Realtime spot settlement prices (5-minute data, not transactable)
 - Load (elec. demand) forecasts at the 7-day-ahead, 2-day-ahead, 1-day-ahead, and realtime horizons
 - Hourly measurements of water storage levels in all 47 California reservoirs (CA Dept. of Water Resources data)
 - Recent weather measurements from weather stations near the electricity price node in question.  Avail NOAA weather data...
  (typically 3 measurements per day at each station):
 1. temperatures
 2. wind speeds
 3. "solarity" (cloud ceiling + visibility)