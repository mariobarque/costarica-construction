import pandas as pd
#import seaborn as sns
#import matplotlib.pyplot as plt
#from matplotlib import style
#style.use('ggplot')


df = pd.read_csv("https://raw.githubusercontent.com/mariobarque/costarica-construction/master/data/construction-data-processed.csv")



#sns.pairplot(df)

df.corr()
