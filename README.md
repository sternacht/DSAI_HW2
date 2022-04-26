# DSAI_HW2
## 目錄

- [成員](#成員)
- [目標](#目標)
- [規則](#規則)
- [Dataset](#dataset)
- [策略](#策略)
- [LSTM](#lstm)
## 成員
E94076194 黃彤洋
E94076186 許中瑋

## 目標
預測股票價格，利用模型生成所要執行的交易動作，盡可能將交易動作的獲利最大化

## 規則
持有的股票會介於-1 ~ 1之間，分為3種狀態:
- 1 : 持有一張股票
- 0 : 沒有任何股票
- -1 : 賣空一張股票

可以進行的動作也有3種:
- 1 : 買股票，若已經持有的狀態則無法再買
- 0 : 不做任何動作
- -1 : 賣股票，若是已經賣空一張股票則無法再賣

買賣的價格以開盤價做計算，最後一天股票清零時則以收盤價計算

利用[StockProfitCalculator](https://github.com/NCKU-CCS/StockProfitCalculator)來計算最後的獲利
## Dataset
股票的資料為 NASDAQ: IBM，包含`開盤價`、`最高`、`最低`與`收盤價`

[[Stock Reference](https://www.nasdaq.com/market-activity/stocks/ibm)]
## 策略
在接收到當天的股價時，我們需要決定隔天的買賣，而隔天的買賣是否會賺則是依據後天的股價變化
因此若是要每天的操作保持賺錢則我們要依照將明天與後天預測的股價做比較
- 明天的股價 > 後天的股價 : 買股票
- 明天的股價 < 後天的股價 : 賣股票

在預測上使用了前5天的資料，並且同時將`開盤價`與`收盤價`考慮進去。因為在買賣時是依據`開盤價`，若要預測後天的`開盤價`則要依序預測3天

若是我們得到1~5天的資料，`開盤價`為 **O**，收盤價為 **C**，預測流程為:

1. **利用[O<sub>1</sub>, C<sub>1</sub>, O<sub>2</sub>, ........,O<sub>5</sub>, C<sub>5</sub>] 預測出 O<sub>6</sub>**
2. **利用[ C<sub>1</sub>, O<sub>2</sub>, ........,O<sub>5</sub>, C<sub>5</sub>, O<sub>6</sub>] 預測出C<sub>6</sub>**
3. **利用[O<sub>2</sub>, C<sub>2</sub>, O<sub>3</sub>, ........,O<sub>6</sub>, C<sub>6</sub>] 預測出 O<sub>7</sub>**
4. **比較O<sub>6</sub>與O<sub>7</sub>來決定在第六天的的動作**

當已經無法進行買/賣股票時，則不做任何動作

## LSTM
利用LSTM來進行股價的預測

在預測時可以看到預測與實際資料會有一段的落差，因此將一部分的training data保留不用來訓練模型，並且放入訓練完的模型得出預測與實際的offset，再將預測的資料加上此offset

最後經過StockProfitCalculator的測試後選擇了目前的架構

![](https://i.imgur.com/HzdDG4k.jpg)


![](https://i.imgur.com/bIY0SWa.png)

