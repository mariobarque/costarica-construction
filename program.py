import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')


df = pd.read_csv("data/construction-data-processed.csv")


#numeric_columns = ['valobr', '']

df['valobr'].plot.kde()



#sns.pairplot(df)

#df.corr()

print('Hi')
