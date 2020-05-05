import os
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px


def get_plot(my_data, plot_type, my_y_axis):
    if plot_type == 'scatter':
        return px.scatter(my_data, x='startdate', y=my_y_axis,
                hover_name='grade', trendline="lowess", size_max=60)
    else: 
        return px.box(my_data, x="state", y=my_y_axis, notched=True)

def get_scatter_mapbox(my_data, size_var):
    fig = px.scatter_mapbox(my_data, lat="lat", lon="long", hover_name="state", hover_data=["state", size_var],
                         color='grade', size=size_var, zoom=3, height=400)
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    return fig

def get_choropleth(my_data, cat_color, hover_list):
    return px.choropleth(my_data, locations=list(my_data['state']), locationmode="USA-states", color=cat_color, scope="usa",
                             color_continuous_scale='Blackbody', hover_name="state", hover_data=hover_list)