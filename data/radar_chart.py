"""
    For more details of Radar Charts in Python, please refer to https://plotly.com/python/radar-chart/
"""
import plotly.graph_objects as go
import plotly.offline as pyo
import json

def draw_radar_chart(pl_num_dict, color):
    print(pl_num_dict)
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=list(pl_num_dict.values()), 
                theta=list(pl_num_dict.keys()), 
                fill='toself', 
                name='programming language',
                fillcolor=color,
                line_color=color,
            ),
        ],
        layout=go.Layout(
            title=go.layout.Title(text='The instruction number of per programming language'),
            polar={'radialaxis': {'visible': True}},
            showlegend=False
        )
    )
    pyo.plot(fig)


with open('pl_num_dis_190k.json', 'r') as pl_dis:
    pl_dis_data = json.load(pl_dis) 

# draw_radar_chart(pl_dis_data["raw data"], color='#e9c46a')
draw_radar_chart(pl_dis_data["clean data"], color='#a2d2ff')
    