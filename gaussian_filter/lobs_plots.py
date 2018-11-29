import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.stats import norm

import datetime
import time
import zipfile

import plotly.offline as py
import plotly.figure_factory as ff
import plotly.graph_objs as go

py_config = {'displayModeBar': False, 'showLink': False, 'editable': False}



def load_order_book_snapshots(filename):
    with zipfile.ZipFile(filename) as z:
        with z.open("OrderBookSnapshots.csv") as f:
            order_book_snapshots = f.readlines()
    return order_book_snapshots

def transform_order_book_snapshot_to_orders(snapshot):
    s = snapshot.split()

    i1 = s.index(b'BID')
    i2 = s.index(b'ASK')
    i3 = len(s)

    date_time = datetime.datetime.strptime(s[0].decode("utf-8") + ' ' + s[1].decode("utf-8"), '%Y%m%d %H%M%S%f')

    buy_orders = np.array(s[i1+1:i2]).astype(np.float64).reshape(-1, 2)

    sell_orders = np.array(s[i2+1:i3]).astype(np.float64).reshape(-1, 2)

    return date_time, buy_orders, sell_orders



def gdf_representation(buy_orders, sell_orders, gdf):
    buy_gdf_y = gdf[0] * norm.pdf(buy_orders[:, 0], loc=gdf[1], scale=gdf[2])
    sell_gdf_y = gdf[0] * norm.pdf(sell_orders[:, 0], loc=gdf[1], scale=gdf[2])
    return np.clip(buy_orders[:, 1], 0.0, buy_gdf_y).sum() + np.clip(sell_orders[:, 1], 0.0, sell_gdf_y).sum()



def plot_lob(buy_orders, sell_orders, levels=15, showMidPrice=True, title='Limit Order Book', xtitle='Price', ytitle='Volume'):
    b_Buy = go.Bar(
        x = buy_orders[-levels:, 0],
        y = buy_orders[-levels:, 1],
        name = 'buy orders',
        marker = dict(
            color = 'rgb(0, 128, 0)',
            line = dict(
                color = 'rgb(0, 128, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    b_Sell = go.Bar(
        x = sell_orders[:levels, 0],
        y = sell_orders[:levels, 1],
        name = 'sell orders',
        marker = dict(
            color = 'rgb(128, 0, 0)',
            line = dict(
                color = 'rgb(128, 0, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    data = go.Data([b_Buy, b_Sell])

    layout = go.Layout(
        title = title,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=ytitle),
        showlegend = True,
        legend = dict(orientation="h")
    )

    if showMidPrice:
        layout.update(
            shapes = [dict(
                type = 'line',
                xref = 'x',
                x0 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                x1 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',            
                y0 = 0.0,
                y1 = 1.0,
                line = dict(
                    color = 'rgb(80, 80, 80)',
                    width = 4,
                    dash = 'dash'
                )
            )],
            annotations=[dict(
                text = 'Mid-Price',
                textangle = -90,
                xref = 'x',
                x = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',
                y = 1.0,
                showarrow = False,
                xanchor = 'left',
                xshift = 0
            )]
        )

    figure = go.Figure(data=data, layout=layout)

    py.iplot(figure, config=py_config)



def plot_lob_and_gdf(buy_orders, sell_orders, gdf, levels=15, showMidPrice=True, title='Limit Order Book and Gaussian Density Filter', xtitle='Price (normalized)', ytitle='Volume (normalized)'):

    gdf_x = np.linspace(buy_orders[-levels:, 0].min(), sell_orders[:levels, 0].max(), 200)
    gdf_y = gdf[0] * norm.pdf(gdf_x, loc=gdf[1], scale=gdf[2])

    b_Buy = go.Bar(
        x = buy_orders[-levels:, 0],
        y = buy_orders[-levels:, 1],
        name = 'buy orders',
        marker = dict(
            color = 'rgb(0, 128, 0)',
            line = dict(
                color = 'rgb(0, 128, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    b_Sell = go.Bar(
        x = sell_orders[:levels, 0],
        y = sell_orders[:levels, 1],
        name = 'sell orders',
        marker = dict(
            color = 'rgb(128, 0, 0)',
            line = dict(
                color = 'rgb(128, 0, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    s_gdf = go.Scatter(
        x = gdf_x,
        y = gdf_y,
        name = 'GDF',
        showlegend = False,
        fill = 'tozeroy',
        mode = 'lines',
        line = dict(
            color = 'rgb(80, 80, 80)',
            width = 1.5
        ),
        opacity = 0.75
    )
    data = go.Data([b_Buy, b_Sell, s_gdf])

    layout = go.Layout(
        title = title,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=ytitle),
        showlegend = True,
        legend = dict(orientation="h")
    )

    if showMidPrice:
        layout.update(
            shapes = [dict(
                type = 'line',
                xref = 'x',
                x0 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                x1 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',            
                y0 = 0.0,
                y1 = 1.0,
                line = dict(
                    color = 'rgb(80, 80, 80)',
                    width = 4,
                    dash = 'dash'
                )
            )],
            annotations=[dict(
                text = 'Mid-Price',
                textangle = -90,
                xref = 'x',
                x = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',
                y = 1.0,
                showarrow = False,
                xanchor = 'left',
                xshift = 0
            )]
        )

    figure = go.Figure(data=data, layout=layout)

    py.iplot(figure, config=py_config)



def plot_lob_clipped_to_gdf(buy_orders, sell_orders, gdf, levels=15, showMidPrice=True, title='Limit Order Book clipped to Gaussian Density Filter', xtitle='Price (normalized)', ytitle='Volume (normalized)'):
    buy_gdf_y = gdf[0] * norm.pdf(buy_orders[:, 0], loc=gdf[1], scale=gdf[2])
    buy_orders_clipped = buy_orders.copy()
    buy_orders_clipped[:, 1] = np.clip(buy_orders[:, 1], 0.0, buy_gdf_y)

    sell_gdf_y = gdf[0] * norm.pdf(sell_orders[:, 0], loc=gdf[1], scale=gdf[2])
    sell_orders_clipped = sell_orders.copy()
    sell_orders_clipped[:, 1] = np.clip(sell_orders[:, 1], 0.0, sell_gdf_y)

    b_BuyC = go.Bar(
        x = buy_orders[-levels:, 0],
        y = buy_orders_clipped[-levels:, 1],
        name = 'buy orders',
        marker = dict(
            color = 'rgb(0, 128, 0)',
            line = dict(
                color = 'rgb(0, 128, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    b_Buy = go.Bar(
        x = buy_orders[-levels:, 0],
        y = buy_orders[-levels:, 1] - buy_orders_clipped[-levels:, 1],
        name = 'buy orders',
        showlegend = False,
        marker = dict(
            color = 'rgb(128, 128, 128)',
            line = dict(
                color = 'rgb(128, 128, 128)',
                width = 1.5
            )
        ),
        opacity = 0.25
    )
    b_SellC = go.Bar(
        x = sell_orders[:levels, 0],
        y = sell_orders_clipped[:levels, 1],
        name = 'sell orders',
        marker = dict(
            color = 'rgb(128, 0, 0)',
            line = dict(
                color = 'rgb(128, 0, 0)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    b_Sell = go.Bar(
        x = sell_orders[:levels, 0],
        y = sell_orders[:levels, 1] - sell_orders_clipped[:levels, 1],
        name = 'sell orders',
        showlegend = False,
        marker = dict(
            color = 'rgb(128, 128, 128)',
            line = dict(
                color = 'rgb(128, 128, 128)',
                width = 1.5
            )
        ),
        opacity = 0.25
    )
    data = go.Data([b_BuyC, b_Buy, b_SellC, b_Sell])

    layout = go.Layout(
        title = title,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=ytitle),
        barmode = 'stack',
        showlegend = True,
        legend = dict(orientation="h")
    )

    if showMidPrice:
        layout.update(
            shapes = [dict(
                type = 'line',
                xref = 'x',
                x0 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                x1 = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',            
                y0 = 0.0,
                y1 = 1.0,
                line = dict(
                    color = 'rgb(80, 80, 80)',
                    width = 4,
                    dash = 'dash'
                )
            )],
            annotations=[dict(
                text = 'Mid-Price',
                textangle = -90,
                xref = 'x',
                x = (buy_orders[-1, 0] + sell_orders[0, 0]) / 2.0,
                yref = 'paper',
                y = 1.0,
                showarrow = False,
                xanchor = 'left',
                xshift = 0
            )]
        )

    figure = go.Figure(data=data, layout=layout)

    py.iplot(figure, config=py_config)



def plot_gdf_features(gdf_features, title='Limit Order Book in GDF Encoding', xtitle='GDF Feature', ytitle='GDF Value'):
    b_gdf = go.Bar(
        x = np.arange(len(gdf_features)),
        y = gdf_features,
        name = 'GDF Features',
        marker = dict(
            color = 'rgb(0, 0, 128)',
            line = dict(
                color = 'rgb(0, 0, 128)',
                width = 1.5
            )
        ),
        opacity = 0.75
    )
    data = go.Data([b_gdf])

    layout = go.Layout(
        title = title,
        xaxis = dict(title=xtitle),
        yaxis = dict(title=ytitle),
        showlegend = False,
    )

    figure = go.Figure(data=data, layout=layout)

    py.iplot(figure, config=py_config)
