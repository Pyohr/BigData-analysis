import pandas as pd
import numpy as np
from apyori import apriori
import matplotlib.pyplot as plt

transactions = [['beer', 'nuts'],['wine', 'cheese'],['beer','chicken','nuts'],['wine','icecream'],['wine','nuts'],
				['wine','cheese'],['wine','pizza']]
results = list(apriori(transactions,min_support = 0.2, min_lift=2))

transaction_num = 7

subset = []
count = 0
for a  in results:
	subset.append(list(a[0]))
	count= count+1

print(subset)
print(count)



