from ses_theta import sesThetaF 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

val = [346.6565,  354.4733,  355.663 ,  373.6322,  386.2713,
  400.5881, 425.3325,  485.1494,  506.0482,  526.792 ,  
  560.2689,  570.211, 600.12, 607.23, 610.23, 
  654.27, 676.89, 680.24, 690.56, 734.23, 
  755.23, 785.23, 746.217, 765.23, 751.23, 761.25] 

index= pd.date_range(start='01-01-1996', end='02-28-1998', freq='M')
data = pd.Series(val, index)

model = sThetaF(data, s_period = 12)

mean = model['mean']

fitted =  model['fitted']

residuals = model['residuals']

plt.figure(figsize = (12,6))

plt.plot(fitted, marker = '.', color = 'red', label = 'In-sample Fitted')
plt.plot(mean, marker = '*', color = 'blue', label = 'Forecast')
plt.plot(residuals, marker = '', color = 'green', label = 'Residuals')
plt.title('Standard Theta Model')
plt.legend()
plt.show()
