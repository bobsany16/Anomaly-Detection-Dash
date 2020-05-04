import pandas as pd

###Read Latlong file###
df_latlong = pd.read_csv('static/statelatlong.csv')

###Adding Lat Long to exsiting Dataset###
def addLatLong(dataset2, type1, type):
    list1 = []
    list2 = []
    for i in list(dataset2['state']):
        list1.append(list(df_latlong[df_latlong[type1] == i][type]))
    for i in list1:
        list2.append(i[0])
    return list2