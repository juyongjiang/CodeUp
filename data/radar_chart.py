"""
    For more details of Radar Charts in Python, please refer to https://plotly.com/python/radar-chart/
"""
import plotly.graph_objects as go
import plotly.offline as pyo
import json

def draw_radar_chart(pl_num_dict):
    fig = go.Figure(
        data=[
            go.Scatterpolar(r=list(pl_num_dict.values()), theta=list(pl_num_dict.keys()), fill='toself', name='programming language'),
        ],
        layout=go.Layout(
            title=go.layout.Title(text='The instruction number of per programming language'),
            polar={'radialaxis': {'visible': True}},
            showlegend=False
        )
    )
    pyo.plot(fig)


with open('pl_raw.json', 'r') as pl_raw, open('pl_clean.json', 'r') as pl_clean:
    pl_raw = json.load(pl_raw).items() # key=lambda x:x[1], reverse=True)
    pl_clean = json.load(pl_clean).items()

# print(pl_raw)
# print(pl_clean)

# draw_radar_chart(pl_raw)
draw_radar_chart(pl_clean)
    
