from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

def update_clf(est, sample, behave, contam, rand_state):
    return IsolationForest(n_estimators=est, max_samples=sample, behaviour=behave, contamination=contam, random_state=rand_state)


def update_anomaly_scores(fitter, le_1, le_2, _data):
    scores = fitter.decision_function(_data)
    X_res = _data.copy()

    #Predictions & scores for Validation Set#
    X_res['anomaly'] = fitter.predict(_data)
    X_res['scores'] = scores
    #Get State and Grade Back#
    X_res['state'] = list(le_1.inverse_transform(X_res['state']))
    X_res['grade'] = list(le_2.inverse_transform(X_res['grade']))
    return X_res

def generate_table(data, max_rows=5):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in data.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(data.iloc[i][col]) for col in data.columns
            ]) for i in range(min(len(data), max_rows))
        ])
    ])

def combineCD(state, my_data, n):
    for i in range(1, n):
        my_state = state + " CD-" + str(i)
        my_data.loc[my_data['state'] == my_state, 'state'] = state
    return my_data[my_data['state'] == state].shape