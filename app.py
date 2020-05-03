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


###Reading .csv files###
###And sort according to date###
dataset = pd.read_csv('presidential_polls.csv')
df = dataset.drop(['cycle', 'branch', 'type', 'forecastdate', 'matchup', 'enddate', 'pollster', 'samplesize', 'population', 'poll_wt', 'rawpoll_clinton', 'rawpoll_trump', 'rawpoll_johnson',
                   'rawpoll_mcmullin', 'adjpoll_johnson', 'adjpoll_johnson', 'adjpoll_mcmullin', 'multiversions', 'url', 'poll_id', 'question_id', 'createddate', 'timestamp'], axis=1)
df['startdate'] = pd.to_datetime(df.startdate)
df.sort_values(by='startdate')


###Read Latlong file###
df_latlong = pd.read_csv('statelatlong.csv')


###Combining Congressional Districts###
def combineCD(state, n):
    for i in range(1, n):
        my_state = state + " CD-" + str(i)
        df.loc[df['state'] == my_state, 'state'] = state
    return df[df['state'] == state].shape


combineCD('Maine', 3)
combineCD('Nebraska', 4)

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


###Changing states to abrs###
state_abr = {'Alabama': 'AL',
             'Alaska': 'AK',
             'Arizona': 'AZ',
             'Arkansas': 'AR',
             'California': 'CA',
             'Colorado': 'CO',
             'Connecticut': 'CT',
             'Delaware': 'DE',
             'District of Columbia': 'DC',
             'Florida': 'FL',
             'Georgia': 'GA',
             'Hawaii': 'HI',
             'Idaho': 'ID',
             'Illinois': 'IL',
             'Indiana': 'IN',
             'Iowa': 'IA',
             'Kansas': 'KS',
             'Kentucky': 'KY',
             'Louisiana': 'LA',
             'Maine': 'ME',
             'Maryland': 'MD',
             'Massachusetts': 'MA',
             'Michigan': 'MI',
             'Minnesota': 'MN',
             'Mississippi': 'MS',
             'Missouri': 'MO',
             'Montana': 'MT',
             'Nebraska': 'NE',
             'Nevada': 'NV',
             'New Hampshire': 'NH',
             'New Jersey': 'NJ',
             'New Mexico': 'NM',
             'New York': 'NY',
             'North Carolina': 'NC',
             'North Dakota': 'ND',
             'Ohio': 'OH',
             'Oklahoma': 'OK',
             'Oregon': 'OR',
             'Pennsylvania': 'PA',
             'Rhode Island': 'RI',
             'South Carolina': 'SC',
             'South Dakota': 'SD',
             'Tennessee': 'TN',
             'Texas': 'TX',
             'Utah': 'UT',
             'Vermont': 'VT',
             'Virgin Islands': 'VI',
             'Virginia': 'VA',
             'Washington': 'WA',
             'West Virginia': 'WV',
             'Wisconsin': 'WI',
             'Wyoming': 'WY'}

res = []
for i in list(df6['state']):
    res.append(state_abr[i])

df6['state'] = res


###Adding Lat Long to exsiting Dataset###
def addLatLong(dataset2, type1, type):
    list1 = []
    list2 = []
    for i in list(dataset2['state']):
        list1.append(list(df_latlong[df_latlong[type1] == i][type]))
    for i in list1:
        list2.append(i[0])
    return list2


df2['lat'] = addLatLong(df2, 'City', 'Latitude')
df2['long'] = addLatLong(df2, 'City', 'Longitude')

df3 = df2
df4 = df3
df_clinton = df2.drop(['adjpoll_trump'], axis=1)
df_trump = df3.drop(['adjpoll_clinton'], axis=1)

###Figures###
fig = px.scatter(df_clinton, x='startdate', y='adjpoll_clinton',
                 color='state', hover_name='grade', size_max=60)
fig2 = px.scatter(df_trump, x='startdate', y='adjpoll_trump',
                  color='state', hover_name='grade', size_max=60)
fig3 = px.strip(df4, x="startdate", y="grade", orientation="h", color='state')

fig4 = px.box(df_clinton, x="state", y="adjpoll_clinton", notched=True)
fig5 = px.box(df_trump, x="state", y="adjpoll_trump", notched=True)

fig6 = px.scatter_mapbox(df_clinton, lat="lat", lon="long", hover_name="state", hover_data=["state", "adjpoll_clinton"],
                         color='grade', size='adjpoll_clinton', zoom=3, height=400)
fig6.update_layout(mapbox_style="open-street-map")
fig6.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

fig7 = px.scatter_mapbox(df_trump, lat="lat", lon="long", hover_name="state", hover_data=["state", 'adjpoll_trump'],
                         color='grade', size='adjpoll_trump', zoom=3, height=400)
fig7.update_layout(mapbox_style="open-street-map")
fig7.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})


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

#Reusable compoenent for clf#


def update_clf(est, sample, behave, contam, rand_state):
    return IsolationForest(n_estimators=est, max_samples=sample, behaviour=behave, contamination=contam, random_state=rand_state)

#Fitting the Isolation Forest#
#clf = IsolationForest(n_estimators=100, max_samples='auto', behaviour="new", contamination=0.1, random_state=42)
# clf.fit(X_train)
#clf = update_clf(100, 'auto', 'new', 0.1, 42)
# clf.fit(X_train)

#train_scores = clf.decision_function(X_train)
#val_scores = clf.decision_function(X_valid)

#Predictions & scores for Validation Set#
#X_valid['anomaly'] = clf.predict(X_valid)
#X_valid['scores'] = val_scores

#Prediction & scores for Training Set#
# X_train['anomaly']=clf.predict(X_train)
# X_train['scores']=train_scores

#Get State and Grade Back#
##X_valid['state'] = list(le2.inverse_transform(X_valid['state']))
#X_valid['grade'] = list(le1.inverse_transform(X_valid['grade']))
#Add Lat and Long#

#X_valid['lat'] = addLatLong(X_valid, 'State', 'Latitude')
#X_valid['long'] = addLatLong(X_valid, 'State', 'Longitude')

###Mapping Anomalies to States###
# fig8 = px.scatter_mapbox(X_valid, lat="lat", lon="long", hover_name="state", hover_data=["state",'adjpoll_trump', 'adjpoll_clinton'],
    # color='anomaly', zoom=3, height=400, color_continuous_scale='Blackbody', title="Predicted Anomalies for X_valid set")
# fig8.update_layout(mapbox_style="open-street-map")
# fig8.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

#fig9 = px.choropleth(X_valid, locations=list(X_valid['state']), locationmode="USA-states", color='anomaly', scope="usa", color_continuous_scale='Blackbody',hover_name="state", hover_data=["state",'adjpoll_trump', 'adjpoll_clinton'])

###Showing Original Table###


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
    html.Div(children='''
        Overall poll results distribution by grades of both candidates
    '''),

    dcc.Graph(figure=fig3),
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
    html.Div(id='graph-display', style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap': 'wrap'})])


@app.callback(
    Output(component_id='graph-display', component_property='children'),
    [Input(component_id='can-dropdown', component_property='value')]
)
def update_graph(value):
    if (value == 'Trump'):
        return dcc.Graph(figure=fig2), dcc.Graph(figure=fig5), dcc.Graph(figure=fig7)
    else:
        return dcc.Graph(figure=fig), dcc.Graph(figure=fig4), dcc.Graph(figure=fig6)


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
        ])

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
        #train_scores = clf.decision_function(X_train)
        val_scores = clf.decision_function(X_valid)

        #Predictions & scores for Validation Set#
        X_valid['anomaly'] = clf.predict(X_valid)
        X_valid['scores'] = val_scores
        #Get State and Grade Back#
        X_valid['state'] = list(le2.inverse_transform(X_valid['state']))
        X_valid['grade'] = list(le1.inverse_transform(X_valid['grade']))
        fig9 = px.choropleth(X_valid, locations=list(X_valid['state']), locationmode="USA-states", color='anomaly', scope="usa",
                             color_continuous_scale='Blackbody', hover_name="state", hover_data=["state", 'adjpoll_trump', 'adjpoll_clinton'])
        return dcc.Graph(figure=fig9)
    else:
        train_scores = clf.decision_function(X_train)
        #Predictions & scores for Validation Set#
        X_train['anomaly'] = clf.predict(X_train)
        X_train['scores'] = train_scores

        #Prediction & scores for Training Set#
        X_train['anomaly'] = clf.predict(X_train)
        X_train['scores'] = train_scores

        #Get State and Grade Back#
        X_train['state'] = list(le2.inverse_transform(X_train['state']))
        X_train['grade'] = list(le1.inverse_transform(X_train['grade']))
        fig10 = px.choropleth(X_train, locations=list(X_train['state']), locationmode="USA-states", color='anomaly', scope="usa",
                              color_continuous_scale='Blackbody', hover_name="state", hover_data=["state", 'adjpoll_trump', 'adjpoll_clinton'])
        return dcc.Graph(figure=fig10)



if __name__ == '__main__':
    app.run_server(debug=True)
