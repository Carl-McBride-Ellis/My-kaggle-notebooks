The following represents a selection of my kaggle notebooks

# <center style="background-color:Gainsboro; width:80%;">1. data cleaning / preparation</center>

* [Data anonymization using Faker (Titanic example)](https://www.kaggle.com/carlmcbrideellis/data-anonymization-using-faker-titanic-example)

# <center style="background-color:Gainsboro; width:80%;"> 2. eda (exploratory data analysis) </center>
* [Anscombe's quartet and the importance of EDA](https://www.kaggle.com/carlmcbrideellis/anscombe-s-quartet-and-the-importance-of-eda) (+ [dataset](https://www.kaggle.com/carlmcbrideellis/data-anscombes-quartet))
* [Absolute beginners Titanic 'EDA' using dabl](https://www.kaggle.com/carlmcbrideellis/absolute-beginners-titanic-eda-using-dabl)
* [Exploratory data analysis using pandas pivot table](https://www.kaggle.com/carlmcbrideellis/exploratory-data-analysis-using-pandas-pivot-table)
* Use case example: [Riiid: EDA and feature importance](https://www.kaggle.com/carlmcbrideellis/riiid-eda-and-feature-importance)
* Use case example: [Jane Street: EDA of day 0 and feature importance](https://www.kaggle.com/carlmcbrideellis/jane-street-eda-of-day-0-and-feature-importance)
* Use case example: [Ventilator Pressure: EDA and simple submission](https://www.kaggle.com/carlmcbrideellis/ventilator-pressure-eda-and-simple-submission)


#  <center style="background-color:Gainsboro; width:80%;">  3. baselines</center>
An important practice is to create a baseline with which to compare future work against:
* Titanic: Machine Learning from Disaster: [Titanic baseline: an all zeros csv file](https://www.kaggle.com/carlmcbrideellis/titanic-all-zeros-csv-file)
* House Prices: [Create a House Prices baseline score (0.42577)](https://www.kaggle.com/carlmcbrideellis/create-a-house-prices-baseline-score-0-42577)
* OpenVaccine: COVID-19: [Baseline: mean values + Gaussian noise ≈ 0.495](https://www.kaggle.com/carlmcbrideellis/baseline-mean-values-gaussian-noise-0-495)
* Mechanisms of Action (MoA) Prediction: [MoA Baseline = 0.02398 (∀ 0's = 0.13073)](https://www.kaggle.com/carlmcbrideellis/moa-baseline-0-02398-0-s-0-13073)
* INGV - Volcanic Eruption Prediction: [The mean volcano](https://www.kaggle.com/carlmcbrideellis/baseline-the-mean-volcano)
* Jane Street Market Prediction: [A random walk down Jane Street...](https://www.kaggle.com/carlmcbrideellis/baseline-a-random-walk-down-jane-street)
* Store Sales - Time Series Forecasting: [Store Sales baseline: Just plain mean](https://www.kaggle.com/carlmcbrideellis/store-sales-baseline-just-plain-mean)

#  <center style="background-color:Gainsboro; width:80%;"> 4. feature selection / engineering </center>
* [Feature importance using the LASSO](https://www.kaggle.com/carlmcbrideellis/feature-importance-using-the-lasso)
* [Feature selection using the Boruta-SHAP package](https://www.kaggle.com/carlmcbrideellis/feature-selection-using-the-boruta-shap-package)
* [Recursive Feature Elimination (RFE) example](https://www.kaggle.com/carlmcbrideellis/recursive-feature-elimination-rfe-example)
* [House Prices: Permutation Importance example](https://www.kaggle.com/carlmcbrideellis/house-prices-permutation-importance-example)
* [What is Adversarial Validation?](https://www.kaggle.com/carlmcbrideellis/what-is-adversarial-validation)
* [Jane Street: t-SNE using RAPIDS cuML](https://www.kaggle.com/carlmcbrideellis/jane-street-t-sne-using-rapids-cuml)
* [Synthanic feature engineering: Beware!](https://www.kaggle.com/carlmcbrideellis/synthanic-feature-engineering-beware)


#  <center style="background-color:Gainsboro; width:80%;"> 5. classification / regression </center>

This is a collection of my python example scripts for either classification, using the [Titanic: Machine Learning from Disaster](https://www.kaggle.com/c/titanic) competition data, or regression, for which I use the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition data:

| algorithm | classification | regression |
| :--- | --- | --- |
| Logistic regression | [link](https://www.kaggle.com/carlmcbrideellis/titanic-explainability-why-me-asks-miss-doyle) | --- |
| Generalized Additive Models (GAM) | [link](https://www.kaggle.com/carlmcbrideellis/classification-using-generalized-additive-models) | --- |
| Support Vector Machines (SVM) | [link](https://www.kaggle.com/carlmcbrideellis/support-vector-classifier-minimalist-script) | --- |
| Iterative Dichotomiser 3 (ID3) | [link](https://www.kaggle.com/carlmcbrideellis/titanic-using-the-iterative-dichotomiser-3-id3) | --- |
| Decision tree| [link](https://www.kaggle.com/carlmcbrideellis/titanic-some-sex-a-bit-of-class-and-a-tree) | --- |
| Random forest | [link](https://www.kaggle.com/carlmcbrideellis/beginners-random-forest-classifier-script) | [link](https://www.kaggle.com/carlmcbrideellis/random-forest-regression-minimalist-script) |
| Regularized Greedy Forest (RGF) | [link](https://www.kaggle.com/carlmcbrideellis/introduction-to-the-regularized-greedy-forest) | [link](https://www.kaggle.com/carlmcbrideellis/introduction-to-the-regularized-greedy-forest) |
| Gradient boosting: XGBoost | [link](https://www.kaggle.com/carlmcbrideellis/xgboost-classification-minimalist-python-script) | [link](https://www.kaggle.com/carlmcbrideellis/an-introduction-to-xgboost-regression)|
| Gradient boosting: CatBoost | --- | [link](https://www.kaggle.com/carlmcbrideellis/catboost-regression-minimalist-script)|
| Histogram gradient boosting | [link](https://www.kaggle.com/carlmcbrideellis/histogram-gradient-boosting-classifier-example) | [link](https://www.kaggle.com/carlmcbrideellis/histogram-gradient-boosting-regression-example) |
| TabNet | [link](https://www.kaggle.com/carlmcbrideellis/tabnet-simple-binary-classification-example) | [link](https://www.kaggle.com/carlmcbrideellis/tabnet-a-very-simple-regression-example) |
| Neural networks (using keras) | [link](https://www.kaggle.com/carlmcbrideellis/very-simple-neural-network-for-classification) | [link](https://www.kaggle.com/carlmcbrideellis/very-simple-neural-network-regression) |
| Gaussian process | [link](https://www.kaggle.com/carlmcbrideellis/gaussian-process-classification-sample-script) | [link](https://www.kaggle.com/carlmcbrideellis/gaussian-process-regression-sample-script) |
| Hyperparameter grid search | [link](https://www.kaggle.com/carlmcbrideellis/hyperparameter-grid-search-sample-code) | [link](https://www.kaggle.com/carlmcbrideellis/hyperparameter-grid-search-simple-example) |

* [Classification using TensorFlow Decision Forests](https://www.kaggle.com/carlmcbrideellis/classification-using-tensorflow-decision-forests)
* [Titanic in pure H2O.ai](https://www.kaggle.com/carlmcbrideellis/titanic-in-pure-h2o-ai)
* [Predict house prices using H2O.ai (regression)](https://www.kaggle.com/carlmcbrideellis/predict-house-prices-using-h2o-ai-regression)
* [Tabular binary classification with neural networks: Keras](https://www.kaggle.com/carlmcbrideellis/tabular-classification-with-neural-networks-keras)
* [Automatic tuning of XGBoost with XGBTune](https://www.kaggle.com/carlmcbrideellis/automatic-tuning-of-xgboost-with-xgbtune)


#  <center style="background-color:Gainsboro; width:80%;">  6. time series</center>

* [Time series: A simple moving average (MA) model](https://www.kaggle.com/carlmcbrideellis/time-series-a-simple-moving-average-ma-model)
* [Time series decomposition: Naive example](https://www.kaggle.com/carlmcbrideellis/time-series-decomposition-naive-example)
* [LSTM time series prediction: sine wave example](https://www.kaggle.com/carlmcbrideellis/lstm-time-series-prediction-sine-wave-example)
* [LSTM time series + stock price prediction = FAIL](https://www.kaggle.com/carlmcbrideellis/lstm-time-series-stock-price-prediction-fail)
* [Interrupted time series analysis: Causal Impact](https://www.kaggle.com/carlmcbrideellis/interrupted-time-series-analysis-causal-impact)
* [Temporal Convolutional Network using Keras-TCN](https://www.kaggle.com/carlmcbrideellis/temporal-convolutional-network-using-keras-tcn)
* [Probabilistic forecasting using GluonTS: Bitcoin](https://www.kaggle.com/carlmcbrideellis/probabilistic-forecasting-using-gluonts-bitcoin)
* [Plotting OHLC and V ticker data using mplfinance](https://www.kaggle.com/carlmcbrideellis/plotting-ohlc-and-v-ticker-data-using-mplfinance)
* [Correlograms of 14 cryptocurrencies (1 day)](https://www.kaggle.com/carlmcbrideellis/correlograms-of-14-cryptocurrencies-1-day)
* [Granger causality testing for 1 day](https://www.kaggle.com/carlmcbrideellis/granger-causality-testing-for-1-day) along with [Granger causality Part II: The Movie](https://www.kaggle.com/carlmcbrideellis/granger-causality-part-ii-the-movie)

# <center style="background-color:Gainsboro; width:80%;">  7. ensemble methods </center>

* [Ensemble methods: majority voting example](https://www.kaggle.com/carlmcbrideellis/ensemble-methods-majority-voting-example)
* [ML-Ensemble example using House Prices data](https://www.kaggle.com/carlmcbrideellis/ml-ensemble-example-using-house-prices-data)
* [Stacking ensemble using the House Prices data](https://www.kaggle.com/carlmcbrideellis/stacking-ensemble-using-the-house-prices-data)

#  <center style="background-color:Gainsboro; width:80%;"> 8. deployment </center>
* [All in a pickle: Saving the Titanic](https://www.kaggle.com/carlmcbrideellis/all-in-a-pickle-saving-the-titanic) - Saving our machine learning model to a file using pickle

# <center style="background-color:Gainsboro; width:80%;"> 9. explainability  </center>
* [Explainability, collinearity and the variance inflation factor (VIF)](https://www.kaggle.com/carlmcbrideellis/explainability-collinearity-and-the-vif)
* [KISS: Small and simple Titanic models](https://www.kaggle.com/carlmcbrideellis/kiss-small-and-simple-titanic-models)
* [House Prices: my score using only 'OverallQual'](https://www.kaggle.com/carlmcbrideellis/house-prices-my-score-using-only-overallqual) and also a [simple two variable model](https://www.kaggle.com/carlmcbrideellis/simple-two-variable-model)
* [Titanic explainability: Why me? asks Miss Doyle](https://www.kaggle.com/carlmcbrideellis/titanic-explainability-why-me-asks-miss-doyle)
* [TabNet and interpretability: Jane Street example](https://www.kaggle.com/carlmcbrideellis/tabnet-and-interpretability-jane-street-example)
* [GPU accelerated SHAP values: Jane Street example](https://www.kaggle.com/carlmcbrideellis/gpu-accelerated-shap-values-jane-street-example)

#  <center style="background-color:Gainsboro; width:80%;">  10. causality</center>
* [Causal Forests Double ML example using EconML](https://www.kaggle.com/carlmcbrideellis/causal-forests-double-ml-example-using-econml)
* [Interrupted time series analysis: Causal Impact](https://www.kaggle.com/carlmcbrideellis/interrupted-time-series-analysis-causal-impact)

#  <center style="background-color:Gainsboro; width:80%;"> 11. statistics </center>
* [Animated histogram of the central limit theorem](https://www.kaggle.com/carlmcbrideellis/animated-histogram-of-the-central-limit-theorem)
* [Hypothesis testing: The two sample t-test, p-value and power](https://www.kaggle.com/carlmcbrideellis/hypothesis-testing-the-t-test-p-values-and-power)
* [Explainability, collinearity and the variance inflation factor (VIF)](https://www.kaggle.com/carlmcbrideellis/explainability-collinearity-and-the-vif)

#  <center style="background-color:Gainsboro; width:80%;">12. didactic notebooks  </center>
* [Beautiful math in your notebook](https://www.kaggle.com/carlmcbrideellis/beautiful-math-in-your-notebook): a guide to using $\LaTeX$ math markup in kaggle notebooks.
* [Titanic: In all the confusion...](https://www.kaggle.com/carlmcbrideellis/titanic-in-all-the-confusion) which looks at the confusion matrix, ROC curves, $F_1$ scores etc.
* [Classification: How imbalanced is "imbalanced"?](https://www.kaggle.com/carlmcbrideellis/classification-how-imbalanced-is-imbalanced) - (mentioned in ["*Notebooks of the week: Hidden Gems*"](https://www.kaggle.com/general/273618))
* [Overfitting and underfitting the Titanic](https://www.kaggle.com/carlmcbrideellis/overfitting-and-underfitting-the-titanic)
* [False positives, false negatives and the discrimination threshold](https://www.kaggle.com/carlmcbrideellis/discrimination-threshold-false-positive-negative)
* [Introduction to the Regularized Greedy Forest](https://www.kaggle.com/carlmcbrideellis/introduction-to-the-regularized-greedy-forest) (using [rgf_python](https://github.com/RGF-team/rgf/tree/master/python-package))
* [Extrapolation: Do not stray out of the forest!](https://www.kaggle.com/carlmcbrideellis/extrapolation-do-not-stray-out-of-the-forest)
* [Titanic: some sex, a bit of class, and a tree...](https://www.kaggle.com/carlmcbrideellis/titanic-some-sex-a-bit-of-class-and-a-tree)


#  <center style="background-color:Gainsboro; width:80%;"> 13. miscellaneous </center>
* [Titanic leaderboard: a score > 0.8 is great!](https://www.kaggle.com/carlmcbrideellis/titanic-leaderboard-a-score-0-8-is-great)
* [House Prices: How to work offline](https://www.kaggle.com/carlmcbrideellis/house-prices-how-to-work-offline) (+ [dataset](https://www.kaggle.com/carlmcbrideellis/house-prices-how-to-work-offline))
* [Pandas one-liners](https://www.kaggle.com/carlmcbrideellis/pandas-one-liners)
* [The latest trends in data science](https://www.kaggle.com/carlmcbrideellis/the-latest-trends-in-data-science)
* [The Titanic using SQL](https://www.kaggle.com/carlmcbrideellis/the-titanic-using-sql)
* [Some pretty t-SNE plots](https://www.kaggle.com/carlmcbrideellis/some-pretty-t-sne-plots)
* [Encuesta kaggle 2021: ¿España es diferente?](https://www.kaggle.com/carlmcbrideellis/encuesta-kaggle-2021-espa-a-es-diferente)
* [How much do people on kaggle earn by country (2021)](https://www.kaggle.com/carlmcbrideellis/how-much-do-people-on-kaggle-earn-by-country-2021)

**Finance related**
* [S&P 500 daily returns: Normal and Cauchy fits](https://www.kaggle.com/carlmcbrideellis/s-p-500-daily-returns-normal-and-cauchy-fits)
* [Bitcoin candlestick chart (2021)](https://www.kaggle.com/carlmcbrideellis/bitcoin-candlestick-chart-2021)
* [Irrational Exuberance? S&P vs Bitcoin](https://www.kaggle.com/carlmcbrideellis/irrational-exuberance-s-p-vs-bitcoin)



#  <center style="background-color:Gainsboro; width:80%;"> 14. fun with the Meta Kaggle dataset </center>
The [Meta Kaggle](https://www.kaggle.com/kaggle/meta-kaggle) dataset consists of data regarding the kaggle site 
* [Kaggle in numbers](https://www.kaggle.com/carlmcbrideellis/kaggle-in-numbers) - updated almost daily
* [Notebooks: Number of views, and days, per vote](https://www.kaggle.com/carlmcbrideellis/notebooks-number-of-views-and-days-per-vote)
* [kaggle discussions: busiest time of the day?](https://www.kaggle.com/carlmcbrideellis/kaggle-discussions-busiest-time-of-the-day) - (mentioned in ["*Notebooks of the week: Hidden Gems*"](https://www.kaggle.com/general/193544))
* [The kaggle working week](https://www.kaggle.com/carlmcbrideellis/the-kaggle-working-week)
* [WordCloud of gold medal winning notebook titles](https://www.kaggle.com/carlmcbrideellis/wordcloud-of-gold-medal-winning-notebook-titles)
* [Shakeup interactive scatterplot maker](https://www.kaggle.com/carlmcbrideellis/shakeup-interactive-scatterplot-maker)
* [Shakeup scatterplots: Boxes, strings and things...](https://www.kaggle.com/carlmcbrideellis/shakeup-scatterplots-boxes-strings-and-things)
* [Kagglers that have reached the Number 1 rank](https://www.kaggle.com/carlmcbrideellis/kagglers-that-have-reached-the-number-1-rank)

![](https://raw.githubusercontent.com/Carl-McBride-Ellis/images_for_kaggle/main/TPS_Nov_2021_small.png)
### <center style="background-color:white; width:100%;">All the best!</center>
