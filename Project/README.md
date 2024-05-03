# Title: Stock Prediction Based on Report Similarity

**Group 6:** [Wang Xin](https://github.com/ShawnWangXin) & [Gong Jiaxin](https://github.com/mecc10?tab=repositories)

**Instructor Professor:** [Jaehyuk Choi](https://github.com/jaehyukchoi)

**RA mentor:** Su Nan

## how to run the main.ipynb
0. I set up the environment and run the code on autodl website. If you don't want to run the code on your laptop, I can start up a GPU server for you.
1. If you want to run locally, we offer the large data in this baidu netdisk link:

```
link: https://pan.baidu.com/s/1XHwETlhbO6fkO6E2wzjgsQ 
password: 3cth 
```
2. download the project on github, unzip mktcap.zip data.zip close.zip in the "./data/" folder
3. then run the main.ipynb with libraries shown in the requirements.txt
4. the calculation process and data flow of these neural network models is commented inside these following .py scripts

### project modules
* **main.ipynb**: main code file

* **GCN.py**: Graph Convolutional Model
    This module builds a graph convolutional model which can be accessed by instantiating the GCN class. It allows users to apply graph-based neural network architectures to various types of graph-structured data.

* **models.py**: Time Series and Cross-Sectional Models
    This file defines two models: a GRU for time series modeling and an MLP (Multilayer Perceptron) for cross-sectional modeling. These models can be utilized by creating instances of GRU and MLP classes, respectively, for sequence data processing and static pattern recognition.

* **utils.py**: Optimization and Plotting Functions
    The Optim class is constructed to facilitate the optimization steps during the training process. Additionally, the module defines a plot_result function for visualizing results and includes several functions like mse (Mean Squared Error) and ic (Information Coefficient) to compute various metrics.

* **dataset.py**: Data Loading and Pre-processing for Time Series Models
    This script handles the loading of factor data, list of trading days, stock closing prices, market capitalizations, and adjacency matrices. It calculates returns and performs data preprocessing, preparing the dataset specifically for time series models.

* **Customloader.py**: Batch Data Iterator
    Constructs an iterator that returns a batch of data (one day at a time), facilitating the training process for models that require sequential input.

* **datasetMLP.py**: Data Loading and Pre-processing for Cross-Sectional Models
    Similar to dataset.py, this file loads factor data, lists of trading days, and stock closing prices. It calculates returns and conducts data preprocessing, tailored for cross-sectional models which analyze data across different entities without regard to the sequential order.

## 1. Motivation

### 1.1 Shared Analyst Coverage

> **Shared Analyst Coverage Factor** (Ali & Hirshleifer, 2020):
> visualize stocks' fundamental connection by using analyst report data.

**(1) Matrix Construction**

<!-- for each quarter: initialize a zero matrix $A_{m×m}$, select reports published in the past 6 months
	for each analyst: find all stocks covered by it
		for each pair of covered stock $i$ and stock $j$:
			$A_{ij} += 1$ -->

<p align='center'><img src="picture\pic_formula.png" width="800px" /></p>

<p align='center'><img src="picture/pic1_sac.png" width="800px" /></p>


**(2) Factor Calculation**

$$
s a c_i=\frac{\sum_{j=1}^N \log \left(A_{i j}+1\right) mom_j}{\sum_{j=1}^N \log \left(A_{i j}+1\right)} {\text{  }} (sac_i
{\text{ is the factor of }}stock_i{\text{, }} mom_j{\text{ is the past momentum of }}stock_j)
$$

* If some $stocks_j$ increased in the past, the $stock_i$ connected to them may increase in the future

* However, we used to replicate this anomaly factor, **but found it was not as good as said in the paper**.
* So, how can we do some incremental research?

### 1.2 Textual Similarity

Textual similarity provides a new perspective.

<p align='center'><img src="picture/MLF.png" width="500px" /></p>

**Our idea:**

* Textual Similarity reminds us that there is a report text column in original data, which was ignored in past paper.
* However, shared analyst coverage only used the number of analysts, which may not capture the indirect connection between firms.
* Instead, we can try constructing matrix by using report text similarity.

<p align='center'><img src="picture/pic2_sac2.png" width="800px" /></p>

## 2. Methodology

### 2.1 Matrix Construction

<!-- for each quarter: initialize a zero matrix $S_{m×m}$, select reports published in the past 6 months
 	vectorize report text to stock vector
    for each pair of stock i and stock j:
		$S_{ij}+= cos(vec_i, vec_j)$ -->

<p align='center'><img src="picture\pic_formula2.png" width="800px" /></p>

### 2.2 Factor Calculation

$$
sac.txt_i=\frac{\sum_{j=1}^N S_{i j} {\text{  }}mom_j}{\sum_{j=1}^N S_{i j}} {\text{  }} (sac.txt_i
{\text{ is the factor of }}stock_i{\text{, }} mom_j{\text{ is the past momentum of }}stock_j)
$$

### 2.3 Stock Prediction Model

(1) Cross Section: MLP, LR, RF, GBDT
```
[batch_size, feature_num] -> [batch_size, 1]
```
(2) Time series: RNN
```
[batch_size, sequence_len, feature_num] -> [batch_size, 1]
```
(3) Graph:  GCN
```
matrix S + [batch_size, sequence_len, feature_num] -> [batch_size, 1]
```

<p align='center'><img src="picture\pic_model.png" width="800px" /></p>


## 3. Experiment

#### 3.1 Fintune Parameter

| Model | Parameter | Value          |
| ----- | --------- | -------------- |
| BERT  | length    | 500            |
|       | pooling   | first-last-avg |
| RF   | n_estimators| 100         |
|       | max_depth| 15    |
| GBDT   | n_estimators| 100         |
|       | max_depth| 15    |
| MLP   | alpha     | 0.001          |
|       | hidden size| (64, 32)      |
| GRU   | num_layers | 2             |
|       | hidden size| 64            |
| GCN   | num_layers | 4             |


#### 3.2 Single Factor Test
* Assumption: The larger the factor value is, the greater the stock return will be in the future
* Group Invetment Return: rank the stocks by the factor value in descending order into 10 groups, then long the 10th group, short the 1st group.

![](picture/pic6_factor.png)

#### 3.3 Model Comparison

* Invetment Performance
![](picture/pic_model_performance.png)

* Confusion Matrix
* We predict on regression basis. Since classifying the top and bottom group of stock is more significant in investment. We will rank and classify the regression result and analyze **confusion matrix**.

![](picture/pic_confusionmatrix.png)

## 4. Conclusion

### (1) Current Work

* [X] We use textual similarity to improve shared analyst coverage factor.
* [X] We use claculated matrix to enhance stock prediction model.

### (2) Academic Discussion

* [ ] Shared analyst coverage factor seemed invalid after 2023. Thus, we can only explore it in research usage. Besides, we can implement analysis to reveal hidden information channel and **momentum spillover effect**.

## 5. Reference

```
Ali, U., & Hirshleifer, D. (2020). Shared analyst coverage: Unifying momentum spillover effects. Journal of Financial Economics, 136(3), 649-675.

Lee, C. M., Sun, S. T., Wang, R., & Zhang, R. (2019). Technological links and predictable returns. Journal of Financial Economics, 132(3), 76-96.MENZLY, L., & OZBAS, O. (2010). 

Market segmentation and cross-predictability of returns. The Journal of Finance, 65(4), 1555-1580.
```
