# Weather-forecasting-with-Deep-Learninig
Time series forecasting is the process of analyzing time series data using statistics and modeling to make predictions and inform strategic decision-making.
Use Irish Weather dataset from kaggle to experiment various Deep Learning model (LSTM, RNN, CNN - Conv1D, GRU, and Transformer) to forecast Weather in a specific station and many specific features to forecast. Then, choose the best result 

Dataset: https://www.kaggle.com/datasets/conorrot/irish-weather-hourly-data

# Engineering Features
Handle missing value, change type features, scaling, generate sequences to sequences pattern

## Scaling
Instead of Normalization, I extremely recommend use Standardization for time series data

Here's the point, Normalization commonly scale data to range [0;1] or [-1;1] from min and max values but the problem is in time series data (like stock price, weather, many quantitle values,...) values always increasing. In other words, there's no maximum value, maybe in this currently dataset have max value but in fact if we change or update data the range value will change and the weight we scale gonna be different; the value has no upper bound or lower bound either. 

Generally, Normalization for time series data don't affect to the wrong computation but it's a mistake cause it's unboundary data. So, Standardization is more appropriate

## Avoid Data Leakage
One common mistake is to use the entire training data to generate global statistics before splitting it into different splits (training, validation, test), leaking the mean and variance of the test sample into the training process, allowing a model to adjust its predictions for the test samples. This information isn't available in production, so the model's perfornmance will likely degrade. In other words, we learn features from training data and predict test data that we don't know what feature it have,

To avoid leakage, always split data first before scaling, then use the statistics from the train split to scale all the splits. Some even suggest that we split our data before any exploratory data analysis and processing, so that we don't accidentally gain information about the test split

## Generate Seq2Seq pattern
Time series forecasting is the Seq2Seq form. So before we apply model, we have to generate pattern Seq2Seq: historical time stamped data -> future data predicted based on historical data.

For example, if we have data [1, 2, 3, 4, 5, 6, 7, 8] we gonna generate pattern with input length (I call window_size) and the output (batch_size) is the value right after the last value of input. In the next sample, the input will have the output push into the last value of previous input and delete the first value of previous input, the output will take the next value in data. this iteration will stop after the last sample reach to the last value in data. For example: window_size = 5, batch_size = 1
 
 [[[1], [2], [3], [4], [5]]] -> [6]
 
 [[[2], [3], [4], [5], [6]]] -> [7]
 
 [[[3], [4], [5], [6], [7]]] -> [8] and so on.
 
 Or you can use TimeSeriesGenerator instead. https://machinelearningmastery.com/how-to-use-the-timeseriesgenerator-for-time-series-forecasting-in-keras/

# Seasonal_Decompose Analysis
Here is the result I decompose first 1000 sample temperature feaeture equal to 42 days, cause with large sample it can't clearly analyize.
![image](https://user-images.githubusercontent.com/83870939/217588831-d2d5a5e8-bdc5-40ff-8bcd-11787847420d.png)

An additive model suggests that the components are added together as follows: y(t) = Observation + Trend + Seasonality + Residual
- Observation(or level) is the main value that goes on average with time.
- Trend component is a centered moving average of the original series, it's moving up and down periodly seem maybe this sample in a stable temperature
- Seasonal component is the per period average of the detrended series
- Residual(or Noise) component is obtained after removing the trend and seasonal components from the time series.
The data seem very clean and reliable to forecast

# Experiment Generator more sample to predict future

Based on the ideal of generate time series data and model after training. I'm gonna try predict the weather from january 6th 2020 to now (january 2023) and it failed

![image](https://user-images.githubusercontent.com/83870939/217531929-0daff6e7-44a7-4e97-804b-ebed63bdf30c.png)

If you try to predict few sample maybe about in 2 month, you'll see the model predict the next sample and it's seemlike "look fine" cause it's still look like the seasonal_decompose above but how about if we predict further ?. The problem is the feature of next sample based on the previous sample and following the trend of old data, it don't have biased to shuffle data and the data more and more converging to a central line. This is contrary to real-world data.
