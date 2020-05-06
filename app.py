import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.utils import shuffle
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn import preprocessing
from components.my_functions import update_clf, update_anomaly_scores, generate_table, combineCD, is_anomaly
from components.lat_long import addLatLong
from components.state import state_to_abr
from components.figure_functions import get_plot, get_scatter_mapbox, get_choropleth
from statistics import mean

###Reading .csv files###
###And sort according to date###
dataset = pd.read_csv('static/presidential_polls.csv')
df = dataset.drop(['cycle', 'branch', 'type', 'forecastdate', 'matchup', 'enddate', 'pollster', 'samplesize', 'population', 'poll_wt', 'rawpoll_clinton', 'rawpoll_trump', 'rawpoll_johnson',
                   'rawpoll_mcmullin', 'adjpoll_johnson', 'adjpoll_johnson', 'adjpoll_mcmullin', 'multiversions', 'url', 'poll_id', 'question_id', 'createddate', 'timestamp'], axis=1)
df['startdate'] = pd.to_datetime(df.startdate)
df.sort_values(by='startdate')

###Combining Congressional Districts###
combineCD('Maine', df, 3)
combineCD('Nebraska', df, 4)

###Making a different dataset with US state and dropping it from OG dataset###
df_US = df[df['state'] == "U.S."]
indexNames = df[df['state'] == "U.S."].index
df.drop(indexNames, inplace=True)

###Creating final Datasets###
df2 = shuffle(df, random_state=42)

###Change NAN grade to F###
df2 = df2.replace(np.nan, 'F', regex=True)
df6 = df2.copy()  # Without Lat Long for ML purposes.
df6 = df6.drop(['startdate'], axis=1)


df6['state'] = state_to_abr(df6)
df3 = df2
df4 = df3
df_clinton = df2.drop(['adjpoll_trump'], axis=1)
df_clinton['state']=state_to_abr(df_clinton)
df_trump = df3.drop(['adjpoll_clinton'], axis=1)
df_trump['state']=state_to_abr(df_trump)

###Figures###
fig = get_plot(df_clinton, 'scatter', 'adjpoll_clinton')
fig2 = get_plot(df_trump, 'scatter', 'adjpoll_trump')
fig4 = get_plot(df_clinton, 'box', 'adjpoll_clinton')
fig5 = get_plot(df_trump, 'box', 'adjpoll_trump')


###Machine Learning Part###
###########################
###########################
features = df6.columns

#Encoding Labels for datasets#
le1 = preprocessing.LabelEncoder()
le2 = preprocessing.LabelEncoder()

df6['state'] = le2.fit_transform(list(df6['state']))
df6['grade'] = le1.fit_transform(list(df6['grade']))

# Training Set
X_train = df6[features][:4992]

# Validation Set
X_valid = df6[features][4993:len(df6)]


###Actual Dash session###
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config.suppress_callback_exceptions = True


###Main Page with Tabs###
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
app.layout = html.Div(children=[
    html.H1(children='Anomaly Detection', style={
            'font-weight': '900', 'text-align': 'center'}),
    html.Div(children='''
        A Machine Learning research project on possible anomalies detection in the 2016 US Presidential Election datasets.

        Project by Jason Green and Linh Nguyen
    '''),
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Data Visualization', value='tab-1'),
        dcc.Tab(label='Data Analysis/ML', value='tab-2'), ]),
    #Div for Tab-content to display#
    html.Div(id='tabs-content')
], style={'width': '100%', 'height': '100%', 'display': 'flex', 'justify-content': 'center', 'text-align': 'center', 'flex-direction': 'column'})

@app.callback(Output('tabs-content', 'children'),
              [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return tab1_content
    else:
        return tab2_content


###Tab1_content when clicked Tab1###
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
#----------------------------------------------------------------------#
tab1_content = html.Div([
    html.Div(id='graph-display1', children=generate_table(df2),
             style={'overflow-y': 'scroll', 'display': 'flex', 'text-align': 'center', 'justify-content': 'center'}),
    html.H2(children='Data Visualization', style={
            'font-weight': '900', 'text-align': 'center'}),
    html.Div(children=[
        html.H4(children='Choose Candidate', style={'font-weight': '900'}),
        html.Div(children='''
            Please choose one to see how their results are distributed, according to adjusted polling results
            ''', style={'width': '450px'}
                 ),
        html.Div([dcc.Dropdown(
            id='can-dropdown',
            options=[
                {'label': 'Trump', 'value': 'Trump'},
                {'label': 'Clinton', 'value': 'Clinton'},
            ],
            value='Clinton',
            style={'margin-top': '2em', 'align-items': 'center', 'width': '450px', 'cursor': 'default'})]),
    ], style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center'}),
    html.Div(id='graph-display', style={'display': 'flex', 'flex-direction': 'column', 'width': '100vw', 'height': '100vh'})])


@app.callback(
    Output(component_id='graph-display', component_property='children'),
    [Input(component_id='can-dropdown', component_property='value')]
)
def update_graph(value):
    if (value == 'Trump'):
        return dcc.Graph(figure=fig2), dcc.Graph(figure=fig5)
    else:
        return dcc.Graph(figure=fig), dcc.Graph(figure=fig4)


###Tab2_content when clicked Tab1###
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# Multiple Inputs, estimators and samples, contamination [they have to be sliders and drop-down]
tab2_content = html.Div([
    html.Div(id='tab2-wrapper', children=[
        html.Div(id='output', children='''Possible predicted anomalies via Isolation Forest
        '''
                 ),
        html.Div(id='input_section', children=[html.Div(children='''
            Please choose a dataset:
            ''', style={'margin-top': '2em'}),
            dcc.Dropdown(id='data-dropdown', options=[
                {'label': 'Training Set', 'value': 'Train'},
                {'label': 'Validation Set', 'value': 'Valid'}],
            value='Valid'),
            html.Div(children='''
            Please choose n_estimators:
            ''', style={'margin-top': '2em'}),
            dcc.Slider(
            id='estimator-slider',
            min=10,
            max=200,
            step=10,
            value=100,
            marks={
                10: '10',
                25: '25',
                50: '50',
                75: '75',
                100: '100',
                150: '150',
                200: '200'}),
            html.Div(children='''
            Please choose a contamination value:
            ''', style={'margin-top': '2em'}),
            dcc.Slider(
            id='contamination-slider',
            min=0,
            max=0.5,
            step=0.05,
            value=0.1,
            marks={
                0: '0',
                0.05: '0.05',
                0.1: '0.1',
                0.15: '0.15',
                0.2: '0.2',
                0.25: '0.25',
                0.3: '0.3',
                0.35: '0.35',
                0.4: '0.4',
                0.45: '0.45',
                0.5: '0.5'}),
            html.Div(children='''
            Please choose a random_state value:
            ''', style={'margin-top': '2em'}),
            dcc.Input(
            id="random-state-input", type="number",
            min=10, max=150, step=20, value=42, style={'margin-bottom': '2em'})
        ]),
            html.P(children='''Note: The first graph is predicted anomalies by state before offset by predicted normal data points. The second graph is predicted anomalies by state after offset'''),

    ], style={'width': '40vw', 'height': '100%', 'display': 'flex', 'flex-direction': 'column'} ),  # End of input section

    html.Div(id='ml-output-section',
             style={'display': 'flex', 'flex-direction': 'column', 'width': '60vw'})

], style={'display': 'flex', 'flex-direction': 'row', 'height': '100vh', 'width': '100vw'})


@app.callback(
    Output(component_id='ml-output-section', component_property='children'),
    [Input(component_id='data-dropdown', component_property='value'),
     Input(component_id='estimator-slider', component_property='value'),
     Input(component_id='contamination-slider', component_property='value'),
     Input(component_id='random-state-input', component_property='value')
     ])
def update_ml_graph(my_data, est_val, cont_val, rand_val):
    clf = update_clf(est_val, 'auto', 'new', cont_val, rand_val)
    clf.fit(X_train)
    if my_data == 'Valid':
        X_val_res = update_anomaly_scores(clf, le2, le1, X_valid)
        y_pred_acc = X_val_res['anomaly']
        val_scores = X_val_res['scores']
        acc = 'Anomalies Percentage: ' + str(list(y_pred_acc).count(-1)/y_pred_acc.shape[0]*100) + '%'
        acc2 = 'Dataset Accuracy : ' + str(list(y_pred_acc).count(1)/y_pred_acc.shape[0]*100) + '%'
        anomaly_val_scores = [i for i in val_scores if i < 0]
        norm_val_scores = [i for i in val_scores if i>=0]
        score1= " .Mean Anomaly scores: " + str(mean(anomaly_val_scores))
        score2 = " .Mean Anomaly scores: " + str(mean(norm_val_scores))

        overall = X_val_res.groupby('state').sum()
        fig9 = get_choropleth(overall, overall.index, 'anomaly')

        sum_incorrect = is_anomaly(X_val_res, -1)
        fig11 = get_choropleth(sum_incorrect, sum_incorrect.index, 'anomaly')
        return dcc.Graph(figure=fig11),acc, score1, dcc.Graph(figure=fig9), acc2, score2
    else:
        X_train_res = update_anomaly_scores(clf, le2, le1, X_train)
        y_train_acc = X_train_res['anomaly']
        val_scores = X_train_res['scores']
        acc3 = 'Anomalies Percentage: ' + str(list(y_train_acc).count(-1)/y_train_acc.shape[0]*100) + '%'
        acc4 = 'Dataset Accuracy : ' + str(list(y_train_acc).count(1)/y_train_acc.shape[0]*100) + '%'
        anomaly_val_scores = [i for i in val_scores if i < 0]
        norm_val_scores = [i for i in val_scores if i>=0]
        score3= " .Mean Anomaly scores: " + str(mean(anomaly_val_scores))
        score4 = " .Mean Anomaly scores: " + str(mean(norm_val_scores))

        overall = X_train_res.groupby('state').sum()
        fig10 = get_choropleth(overall, overall.index, 'anomaly')

        train_incorrect = is_anomaly(X_train_res, -1)
        fig12 = get_choropleth(train_incorrect, train_incorrect.index, 'anomaly')
        return dcc.Graph(figure=fig12), acc3, score3, dcc.Graph(figure=fig10), acc4, score4



if __name__ == '__main__':
    app.run_server(debug=True)
