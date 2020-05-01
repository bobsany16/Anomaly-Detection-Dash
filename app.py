import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from pylab import *
import shapefile as shp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import numpy as np



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

###Adding Lat Long to exsiting Dataset###
def addLatLong(type):
    list1 = []
    list2 = []
    for i in list(df2['state']):
        list1.append(list(df_latlong[df_latlong['City']==i][type]))
    for i in list1:
        list2.append(i[0])
    return list2

df2['lat'] = addLatLong('Latitude')
df2['long'] = addLatLong('Longitude')

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

#fig6 = px.scatter_mapbox(df_clinton, color="state", size="adjpoll_clinton",
                  #color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)
#fig6.show()


fig6 = px.scatter_mapbox(df_clinton, lat="lat", lon="long", hover_name="state", hover_data=["state", "adjpoll_clinton"],
                        color='grade', size='adjpoll_clinton' , zoom=3, height=400)
fig6.update_layout(mapbox_style="open-street-map")
fig6.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig7 = px.scatter_mapbox(df_trump, lat="lat", lon="long", hover_name="state", hover_data=["state",'adjpoll_trump'],
                        color='grade', size='adjpoll_trump', zoom=3, height=400)
fig7.update_layout(mapbox_style="open-street-map")
fig7.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig8 = px.scatter_matrix(df4)
fig8.show()

###Getting the states Long and Lat###
sns.set(style= "whitegrid", palette="pastel", color_codes=True)
sns.mpl.rc("figure", figsize=(10,6))

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

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.config.suppress_callback_exceptions = True


app.layout = html.Div(children=[
    html.H1(children='Anomaly Detection', style={
            'font-weight': '900', 'text-align': 'center'}),
    html.Div(children='''
        A Machine Learning research project on possible anomalies detection in the 2016 US Presidential Election datasets
    '''),

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
            style={'margin-top': '2em','align-items': 'center', 'width': '450px', 'cursor': 'default'})]),
    ], style={'display': 'flex', 'flex-direction':'column', 'justify-content': 'center', 'align-items': 'center'}),


    html.Div(id='graph-display', 

                 style={'display': 'flex', 'flex-direction': 'row', 'flex-wrap':'wrap'}),



], style={'width': '100%', 'height': '100%', 'display': 'flex', 'justify-content': 'center', 'text-align': 'center', 'flex-direction': 'column'})


@app.callback(
    Output(component_id='graph-display', component_property='children'),
    [Input(component_id='can-dropdown', component_property='value')]
)
def update_graph(value):
    if (value == 'Trump'):

        return dcc.Graph(figure=fig2), dcc.Graph(figure=fig5), dcc.Graph(figure=fig7)
    else:
        return dcc.Graph(figure=fig), dcc.Graph(figure=fig4), dcc.Graph(figure=fig6)




if __name__ == '__main__':
    app.run_server(debug=True)
