# -*- coding:/ utf-8 -*-
"""
This piece of software is bound by The MIT License (MIT)
Copyright (c) 2020 Prashank Kadam
Code written by : Prashank Kadam
Email ID : kadam.pr@husky.neu.edu
Created on - 03/14/2020
version : 1.0
"""
"""
This tool is a part of the Introduction to Data Management 
and Processing course (IDMP), Spring 2020.
The initial version provides features like data filtering and
analysis, basic visualizations, hypothesis testing and 
modeling. In the v1.0 modeling only consists of Linear and 
Logistic regression further versions will have more complex
modeling algorithms
"""
########################################################################################################################
# Importing the required libraries

import base64  # for image conversion
import io

import flask  # using flask server

import dash  # python based framework used for building the app
import dash_table  # table layout component
import dash_html_components as html  # html layout component
import dash_core_components as dcc  # core layout component
import dash_bootstrap_components as dbc  # bootstrap layout component
from dash.dependencies import Input, Output, State

import pandas as pd  # pandas used for wrangling

import plotly.express as px  # for plotting graph components

from sklearn import preprocessing  # used for normalization of target variables
import statsmodels.api as sm
import statsmodels.formula.api as smf  # for modeling
from scipy.stats import shapiro, normaltest, anderson, \
    pearsonr, spearmanr, kendalltau, chi2_contingency, \
    ttest_ind, ttest_rel, f_oneway  # for hypothesis testing

########################################################################################################################
# package lxml required
# Statmodels installation required for trendline

# Declaring a dataframe for uploaded data
data_name = ""
df_up = pd.DataFrame()

# Reading the breast cancer dataset(default) into a dataframe
df = pd.read_csv('data/bcw.csv')
# inserting an index row in the dataframe which will later help in sorting
df.insert(loc=0, column=' index', value=range(1, len(df) + 1))

# fetching the column names of the dataframe
colnames = df.columns
# fetching the data types of each of the columns
dtype_mapping = dict(df.dtypes)
# Separating the continuous and categorical variables into different lists
numeric_cols = [c for c in colnames if dtype_mapping[c] != 'O']
catagory_cols = [c for c in colnames if dtype_mapping[c] == 'O']

# Setting the default page size
PAGE_SIZE = 10

# Setting default dropdowns for the various test which we will be using
# for data analysis
dimensions = ["x", "y", "color", "facet_col", "facet_row"]
graph_types = ["Scatter", "Bar", "Box", "Heatmap"]
hypothesis_tests = ["Normality", "Correlation", "Parametric"]
normality_tests = ["Shapiro-Wilk", "D’Agostino’s K^2", "Anderson-Darling"]
correlation_tests = ["Pearson", "Spearman", "Kendall", "Chi-Squared"]
parametric_tests = ["Student t-test", "Paired Student t-test", "ANOVA"]
modeling_types = ["Regression", "Classification"]

# Setting dropdown dictionaries which will be used in the drop down layouts
col_options = [dict(label=x, value=x) for x in df.columns]
num_options = [dict(label=x, value=x) for x in list(set(numeric_cols))]
cat_options = [dict(label=x, value=x) for x in list(set(catagory_cols))]
graph_options = [dict(label=x, value=x) for x in graph_types]
hypothesis_options = [dict(label=x, value=x) for x in hypothesis_tests]
normality_options = [dict(label=x, value=x) for x in normality_tests]
correlation_options = [dict(label=x, value=x) for x in correlation_tests]
parametric_options = [dict(label=x, value=x) for x in parametric_tests]
types_options = [dict(label=x, value=x) for x in modeling_types]

# Setting the update variable for the scatter plot
upd_scat_x = 0

# Defining the tab layout and formatting
tabs_styles = {
    'height': '44px'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold'
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#17a2b8',
    'color': 'white',
    'padding': '6px'
}

########################################################################################################################

# This app will be deployed on a flask server
# Declaring the flask server
server = flask.Flask(__name__)

# Defining external page formatting for the app
external_stylesheets = [dbc.themes.BOOTSTRAP]

# Declaring the dash app
app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=external_stylesheets
)

# Due to callbacks updating multiple outputs, certain callback
# exceptions can be generated. Inorder to suppress those exceptions
# during runtime, we use the following code
app.config.suppress_callback_exceptions = True

########################################################################################################################

# Fetching the logo image
IDMP_LOGO = 'idmp_logo.png'
# Setting the encoded URL
ENCODED_URL = 'data:image/png;base64,{}'

# Encoding the IDMP logo
logo_enc = base64.b64encode(open(IDMP_LOGO, 'rb').read())

# Defining the layout for the navigation bar
navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo
            dbc.Row(
                [
                    # Setting the logo in the navbar using bootstrap column component
                    dbc.Col(html.Img(src=ENCODED_URL.format(logo_enc.decode()), height="50px")),
                    # Setting the app header
                    dbc.Col(dbc.NavbarBrand("IDMP - Data Analysis Tool", className="ml-2"),
                            style={'font-weight': 'bold', 'color': '#17a2b8'}),
                ],
                align="center",
                no_gutters=True,
            ),
            # href="/apps/dat",
        ),
        dbc.NavbarToggler(id="navbar-toggler")
    ],
)

# Defining the layout for the nav links
nav = html.Div([
    dbc.Nav(
        [
            # Nav link for data table page
            dbc.NavLink("Data Table",
                        id="id_dat",
                        href="/apps/dat",
                        style={'min-width': '200px', 'color': '#17a2b8'}),
            # Nav link for plot page
            dbc.NavLink("Plot",
                        id="id_plt",
                        href="/apps/plt",
                        style={'min-width': '150px', 'color': '#17a2b8'}),
            # Nav link for hypothesis testing page
            dbc.NavLink("Quantization",
                        id="id_qnt",
                        href="/apps/qnt",
                        style={'min-width': '225px', 'color': '#17a2b8'}),
            # Nav link for modeling page
            dbc.NavLink("Modeling",
                        id="id_mod",
                        href="/apps/mod",
                        style={'min-width': '200px', 'color': '#17a2b8'}),
            # dbc.NavLink("Disabled", disabled=True, href="#", style={'min-width': '200px', 'color': 'skyblue'}),
        ],
        id='id_nav'
    )
], style={'padding-top': '10px', 'padding-left': '50px', 'font-weight': 'bold', 'line-height': '30px',
          'font-size': '20px'})

# Defining the layout of our dash app
app.layout = html.Div([
    navbar,
    nav,
    dcc.Location(id='url', refresh=False),  # url routes
    html.Div(id='page-content')  # routed page content
])


########################################################################################################################


# Callback for page routing
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    """
    This is a callback function that takes care of the routing
    of our multipage dash app
    :param pathname: input path name
    :return: page content of the input path
    """
    if pathname == '/apps/dat':
        return tab_data_content
    elif pathname == '/apps/plt':
        return tab_plot_content
    elif pathname == '/apps/qnt':
        return tab_qnt_content
    elif pathname == '/apps/mod':
        return tab_mod_content
    else:
        return tab_data_content


########################################################################################################################
"""
Here we define the layout for our first page which is the data table page
"""
tab_data_content = dbc.Card(
    dbc.CardBody([
        html.Div([
            # dcc.Upload(
            #     id='upload-data',
            #     children=html.Div([
            #         'Drag and Drop or ',
            #         html.A('Select Files')
            #     ]),
            #     style={
            #         'width': '100%',
            #         'height': '60px',
            #         'lineHeight': '60px',
            #         'borderWidth': '1px',
            #         'borderStyle': 'dashed',
            #         'borderRadius': '5px',
            #         'textAlign': 'center',
            #         'margin': '10px'
            #     },
            #     # Allow multiple files to be uploaded
            #     multiple=True
            # ),

            # Layout for entering the URL of externally added data sets
            html.Div([
                html.P([dbc.Input(id="input-url", placeholder="Enter data URL", type="text")])
            ], style={"width": "89%", "float": "left"}),
            # Load Button on the data table page
            dbc.Button("Load",
                       id="load-button",
                       color="info",
                       className="mr-2",
                       style={"width": "10%",
                              "display": "inline-block"}),
            html.H2(id='data-name', style={"width": "99%", "float": "left"}),
            html.Div(id='no-data', style={'display': 'none'})
        ]),
        # Setting the data table layout
        dash_table.DataTable(
            id='datatable',
            columns=[
                # {"name": [i, "test"], "id": i} for i in df.columns
                {"name": [i, "None"], "id": i} for i in df.columns
            ],
            # df.columns,
            page_current=0,
            page_size=PAGE_SIZE,
            page_action='custom',
            style_table={'overflowX': 'scroll'},

            sort_action='custom',
            sort_mode='single',
            sort_by=[],

            filter_action='custom',
            filter_query=''
        ),
        html.Br(),
        # Layout for row count indicator
        html.P(["Row Count" + ":", dcc.Input(id="datatable-row-count",
                                             type='number',
                                             value=10)],
               style={"display": "inline-block", 'fontSize': 12}),
    ]),
    className="mt-3",
)

# Following the operators which we will be using in the filter
# function
operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    """
    This function will filter the data tabla according to the
    values entered in the column inputs, it used used regular
    expression for filtering out similar looking entries
    :param filter_part: input dataframe conditions
    :return: filtered values
    """
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if v0 == value_part[-1] and v0 in ("'", '"', '`'):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


########################################################################################################################
"""
Now we define the layout of our visualizations (Plot) page
"""
# Layout for the scatter plot tab
scat_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-scat",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout for X label dropdown
                html.P(["X label" + ":", dcc.Dropdown(id="xlab-scat", options=col_options)]),
                # Layout for Y label dropdown
                html.P(["Y label" + ":", dcc.Dropdown(id="ylab-scat", options=col_options)]),
                # Layout for colour variable dropdown
                html.P(["Color" + ":", dcc.Dropdown(id="col-scat", options=col_options)]),
                # Layout for size variable dropdown
                html.P(["Size" + ":", dcc.Dropdown(id="siz-scat", options=col_options)]),
                # Layout for facet row variable dropdown
                html.P(["Facet Row" + ":", dcc.Dropdown(id="fac-scat-row", options=col_options)]),
                # Layout for facet column variable dropdown
                html.P(["Facet Column" + ":", dcc.Dropdown(id="fac-scat-col", options=col_options)]),
                # Layout for trendline dropdown
                html.P(["Trendline" + ":", dcc.Dropdown(id="trnd-scat",
                                                        options=[{'label': 'OLS', 'value': 'ols'},
                                                                 {'label': 'Lowess', 'value': 'lowess'}])])

            ],
            style={"width": "25%", "float": "left"},
        ),
        # Layout for the graph (Scatter plot) generated after selecting one or more of the above variables
        dcc.Graph(id="plot-scatter", figure={}, style={"width": "75%", "display": "inline-block", "height": 700}),
    ]),
    className="mt-3"
)

# Layout for the bar plot tab
bar_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-bar",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout for X label dropdown
                html.P(["X label" + ":", dcc.Dropdown(id="xlab-bar", options=col_options)]),
                # Layout for Y label dropdown
                html.P(["Y label" + ":", dcc.Dropdown(id="ylab-bar", options=col_options)]),
                # Layout for color variable dropdown
                html.P(["Color" + ":", dcc.Dropdown(id="col-bar", options=col_options)]),
                # # Layout for bar type dropdowm
                # html.P(["Type" + ":", dcc.Dropdown(id="typ-bar", options=[{'label': 'None', 'value': 'none'},
                #                                                           {'label': 'Stacked', 'value': 'stack'},
                #                                                           {'label': 'Dodged', 'value': 'dodge'}])]),
                # Layout for facet row variable
                html.P(["Facet Row" + ":", dcc.Dropdown(id="fac-bar-row", options=col_options)]),
                # Layout for facet column variable
                html.P(["Facet Column" + ":", dcc.Dropdown(id="fac-bar-col", options=col_options)]),
            ],
            style={"width": "25%", "float": "left"},
        ),
        # Layout for the graph (bar plot) generated after selecting one or more of the above variables
        dcc.Graph(id="plot-bar", figure={}, style={"width": "75%", "display": "inline-block", "height": 700}),
    ]),
    className="mt-3",
)

# Layout for the box plot tab
box_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-box",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout for X label dropdown
                html.P(["X label" + ":", dcc.Dropdown(id="xlab-box", options=col_options)]),
                # Layout for Y label dropdown
                html.P(["Y label" + ":", dcc.Dropdown(id="ylab-box", options=col_options)]),
                # Layout for color variable dropdown
                html.P(["Color" + ":", dcc.Dropdown(id="col-box", options=col_options)]),
                # Layout for facet row variable dropdown
                html.P(["Facet Row" + ":", dcc.Dropdown(id="fac-box-row", options=col_options)]),
                # Layout for facet column variable dropdown
                html.P(["Facet Column" + ":", dcc.Dropdown(id="fac-box-col", options=col_options)]),
            ],
            style={"width": "25%", "float": "left"},
        ),
        # Layout for the graph (box plot) generated after selecting one or more of the above variables
        dcc.Graph(id="plot-box", figure={}, style={"width": "75%", "display": "inline-block", "height": 700}),
    ]),
    className="mt-3",
)

# Layout for heatmap tab
heat_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-heat",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout for X label dropdown
                html.P(["X label" + ":", dcc.Dropdown(id="xlab-heat", options=col_options)]),
                # Layout for Y label dropdown
                html.P(["Y label" + ":", dcc.Dropdown(id="ylab-heat", options=col_options)]),
                # Layout for facet row variable dropdown
                html.P(["Facet Row" + ":", dcc.Dropdown(id="fac-heat-row", options=col_options)]),
                # Layout for facet column variable dropdown
                html.P(["Facet Column" + ":", dcc.Dropdown(id="fac-heat-col", options=col_options)]),
            ],
            style={"width": "25%", "float": "left"},
        ),
        # Layout for the graph (heatmap plot) generated after selecting one or more of the above variables
        dcc.Graph(id="plot-heat", figure={}, style={"width": "75%", "display": "inline-block", "height": 700}),
    ]),
    className="mt-3",
)

# Putting all the above defined layouts into a tab layout
tab_plot_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Tabs(id="tabs-plot", value='tab-scat', children=[
                dcc.Tab(children=scat_tab,
                        label='Scatter',
                        value='tab-scat',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=bar_tab,
                        label='Bar',
                        value='tab-bar',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=box_tab,
                        label='Box',
                        value='tab-box',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=heat_tab,
                        label='Heat Map',
                        value='tab-heat',
                        style=tab_style,
                        selected_style=tab_selected_style)
            ], style=tabs_styles),
            html.Div(id='no-data-plt', children=colnames, style={'display': 'none'})
        ]

    ),
    className="mt-3",
)

########################################################################################################################
"""
Now we define the layout of our hypothesis testing (Quantization) page
"""
# Layout for normality verification tests
norm_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-norm",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout of the dropdown for choosing the normality test to be done
                html.P(
                    ["Normality Tests" + ":", dcc.Dropdown(id="norm-test",
                                                           options=normality_options)]),
                # Layout of the dropdown for choosing test variable
                html.P(
                    ["Test Variable" + ":", dcc.Dropdown(id="norm-var",
                                                         options=num_options)]),
            ],
            style={"width": "25%", "float": "left", "padding": "20px"},
        ),
        # Layout for the plot which will plot the distribution of the test variable
        dcc.Graph(id="plot-norm", figure={}, style={"width": "75%", "display": "inline-block", "height": 500}),

        # Layout for showing the quantized output of the test results
        html.Table([
            html.Tr(html.Td(id='norm-val1')),
            html.Tr(html.Td(id='norm-val2')),
            html.Tr(html.Td(id='norm-val3')),
            html.Tr(html.Td(id='norm-val4')),
            html.Tr(html.Td(id='norm-val5')),
        ], style={"width": "75%",
                  "float": "right",
                  "display": "inline-block",
                  "padding-left": "75px"})
    ]),
    className="mt-3",
)

# Layout for correlation verification tests
corr_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-corr",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout to select the correlation tests dropdown
                html.P(
                    ["Correlation Tests" + ":", dcc.Dropdown(id="corr-test",
                                                             options=correlation_options)]),
                # Layout to select the test variable dropdown
                html.P(
                    ["Test Variable 1" + ":", dcc.Dropdown(id="corr-var1",
                                                           options=num_options)]),
                # Layout to select the second test variable dropdown
                html.P(
                    ["Test Variable 2" + ":", dcc.Dropdown(id="corr-var2",
                                                           options=num_options)])
            ],
            style={"width": "25%", "float": "left", "padding": "20px"},
        ),

        # Layout for the plot which will plot the distribution of the test variable
        dcc.Graph(id="plot-corr", figure={}, style={"width": "75%", "display": "inline-block", "height": 500}),

        # Layout for showing the quantized output of the test results
        html.Table([
            html.Tr(html.Td(id='corr-val1'))
        ], style={"width": "75%",
                  "float": "right",
                  "display": "inline-block",
                  "padding-left": "75px"})
    ]),
    className="mt-3",
)

# Layout for parametric verification tests
para_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-para",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # Layout to select the parametric test dropdown
                html.P(
                    ["Parametric Tests" + ":", dcc.Dropdown(id="para-test",
                                                            options=parametric_options)]),
                # Layout to select the first test variable dropdown
                html.P(
                    ["Test Variable 1" + ":", dcc.Dropdown(id="para-var1",
                                                           options=num_options)]),
                # Layout to select the second test variable dropdown
                html.P(
                    ["Test Variable 2" + ":", dcc.Dropdown(id="para-var2",
                                                           options=num_options)]),
                # html.P(
                #     ["Test Variables" + ":", dcc.Dropdown(id="para-var3",
                #                                           options=num_options,
                #                                           multi=True)], style={'display': 'none'})
            ],
            style={"width": "25%", "float": "left", "padding": "20px"},
        ),

        # Layout for the plot which will plot the distribution of the test variable
        dcc.Graph(id="plot-para", figure={}, style={"width": "75%", "display": "inline-block", "height": 500}),

        # Layout for showing the quantized output of the test results
        html.Table([
            html.Tr(html.Td(id='para-val1'))
        ], style={"width": "75%",
                  "float": "right",
                  "display": "inline-block",
                  "padding-left": "75px"})
    ]),
    className="mt-3",
)

# Putting all the above defined layouts into a tab layout
tab_qnt_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Tabs(id="tabs-qnt", value='tab-norm', children=[
                dcc.Tab(children=norm_tab,
                        label='Normalize',
                        value='tab-norm',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=corr_tab,
                        label='Correlation',
                        value='tab-corr',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=para_tab,
                        label='Parametric',
                        value='tab-para',
                        style=tab_style,
                        selected_style=tab_selected_style)
            ], style=tabs_styles)
        ]

    ),
    className="mt-3",
)

########################################################################################################################
"""
Now we define the layout of our Modeling page
"""
# Layout for Linear regression modeling
lin_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-lin",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
                # html.P(
                #     ["Type" + ":", dcc.Dropdown(id="lin-type",
                #                                 options=types_options)]),

                # Layout for selecting the predictors multi dropdown
                html.P(
                    ["Predictor" + ":", dcc.Dropdown(id="lin-pre",
                                                     options=col_options,
                                                     multi=True)]),
                # Layout for selecting target dropdown
                html.P(
                    ["Target" + ":", dcc.Dropdown(id="lin-tar",
                                                  options=col_options)]),

                # Run button to run the model
                html.Div([dbc.Button("Run",
                                     id="run-button-lin",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
            ],
            style={"width": "25%", "float": "left", "padding": "40px", "height": 1000},
        ),
        # html.Div([
        #     dash_table.DataTable(
        #         id='summary-tab',
        #         columns=[
        #             # {"name": [i, "test"], "id": i} for i in df.columns
        #             {"name": i, "id": i} for i in []
        #         ])
        #     ], style={"width": "75%", "display": "inline-block", "height": 500})

        # Output header
        html.H4("OLS Regression Results:", style={"width": "75%", "display": "inline-block", "padding": "10px"}),

        # Division to display table 1 of the model summary outputs
        html.Div(id='lin-mod-tab1', style={"width": "70%",
                                           "display": "inline-block",
                                           "height": 300,
                                           "padding": "10px"}),

        # Division to display table 2 of the model summary outputs
        html.Div(id='lin-mod-tab2', style={"width": "70%",
                                           "display": "inline-block",
                                           "height": 150,
                                           "padding": "10px"}),

        # Division to display table 3 of the model summary outputs
        html.Div(id='lin-mod-tab3', style={"width": "70%",
                                           "display": "inline-block",
                                           "height": 200,
                                           "padding": "10px"})

        # html.Table(id='lin-mod-tab', style={"width": "75%",
        #                                     "float": "right",
        #                                     "display": "inline-block",
        #                                     "padding": "200px"})
        # dcc.Graph(id="plot-corr", figure={}, style={"width": "75%", "display": "inline-block", "height": 500}),
        # html.Table([
        #     html.Tr(html.Td(id='corr-val1'))
        # ], style={"width": "75%",
        #           "float": "right",
        #           "display": "inline-block",
        #           "padding-left": "75px"})
    ]),
    className="mt-3",
)

# Layout for Logistic regression modeling
log_tab = dbc.Card(
    dbc.CardBody([
        html.Div(
            [
                # Load button for loading new variables to the dropdown
                html.Div([dbc.Button("Load",
                                     id="load-button-log",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),

                # html.P(
                #     ["Type" + ":", dcc.Dropdown(id="lin-type",
                #                                 options=types_options)]),

                # Layout for selecting the predictors multi dropdown
                html.P(
                    ["Predictor" + ":", dcc.Dropdown(id="log-pre",
                                                     options=col_options,
                                                     multi=True)]),

                # Layout for selecting the target for the dropdown
                html.P(
                    ["Target" + ":", dcc.Dropdown(id="log-tar",
                                                  options=col_options)]),

                # Run button to run the model
                html.Div([dbc.Button("Run",
                                     id="run-button-log",
                                     color="info",
                                     className="mr-2",
                                     style={"width": "100%",
                                            "display": "inline-block"})], style={"padding": "20px"}),
            ],
            style={"width": "25%", "float": "left", "padding": "40px", "height": 1000},
        ),

        # html.Div([
        #     dash_table.DataTable(
        #         id='summary-tab',
        #         columns=[
        #             # {"name": [i, "test"], "id": i} for i in df.columns
        #             {"name": i, "id": i} for i in []
        #         ])
        #     ], style={"width": "75%", "display": "inline-block", "height": 500})

        # Output header
        html.H4("Logistic Regression Results:", style={"width": "75%", "display": "inline-block", "padding": "10px"}),

        # Division to display table 1 of the model summary outputs
        html.Div(id='log-mod-tab1', style={"width": "70%",
                                           "display": "inline-block",
                                           "height": 300,
                                           "padding": "10px"}),

        # Division to display table 2 of the model summary outputs
        html.Div(id='log-mod-tab2', style={"width": "70%",
                                           "display": "inline-block",
                                           "height": 150,
                                           "padding": "10px"}),

        # Division to display table 3 of the model summary outputs
        # html.Div(id='log-mod-tab3', style={"width": "70%",
        #                                    "display": "inline-block",
        #                                    "height": 200,
        #                                    "padding": "10px"})
        # html.Table(id='lin-mod-tab', style={"width": "75%",
        #                                     "float": "right",
        #                                     "display": "inline-block",
        #                                     "padding": "200px"})
        # dcc.Graph(id="plot-corr", figure={}, style={"width": "75%", "display": "inline-block", "height": 500}),
        # html.Table([
        #     html.Tr(html.Td(id='corr-val1'))
        # ], style={"width": "75%",
        #           "float": "right",
        #           "display": "inline-block",
        #           "padding-left": "75px"})
    ]),
    className="mt-3",
)

# Putting all the above defined layouts into a tab layout
tab_mod_content = dbc.Card(
    dbc.CardBody(
        [
            dcc.Tabs(id="tabs-mod", value='tab-lin', children=[
                dcc.Tab(children=lin_tab,
                        label='Linear',
                        value='tab-lin',
                        style=tab_style,
                        selected_style=tab_selected_style),
                dcc.Tab(children=log_tab,
                        label='Logistic',
                        value='tab-log',
                        style=tab_style,
                        selected_style=tab_selected_style),
                # dcc.Tab(children=corr_tab,
                #         label='Bayesian',
                #         value='tab-bay',
                #         style=tab_style,
                #         selected_style=tab_selected_style),
                # dcc.Tab(children=para_tab,
                #         label='Decision Tree',
                #         value='tab-dec',
                #         style=tab_style,
                #         selected_style=tab_selected_style),
                # dcc.Tab(children=para_tab,
                #         label='SVM',
                #         value='tab-svm',
                #         style=tab_style,
                #         selected_style=tab_selected_style),
            ], style=tabs_styles)
        ]

    ),
    className="mt-3",
)

########################################################################################################################
"""
Now we define the callbacks for the all the components defined in the above layouts
Callbacks are used to update the data in various plots in real-time as per the 
constraints chosen by the user from the drop down or other input fields
"""


@app.callback(
    Output("no-data", "children"),
    [Input("input-url", "value"),
     Input("load-button", "n_clicks")])
def output_text(value, n):
    """
    This is a callback function that reads the data from the input URL
    :param value: input URL string
    :param n: button clicks
    :return: dataframe converted to json format
    """
    if value is not None and n is not None:

        # Updating the value of uploaded dataframe globally
        global df_up

        # Reading the data into a dataframe from the given url
        df_up = pd.read_csv(value)
        # Inserting the index column into the dataframe
        df_up.insert(loc=0, column=' index', value=range(1, len(df_up) + 1))
        # data_name = value.split('/')[-1].split('.')[0]

        # Setting new column names based on uploaded dataframe columns
        if not df_up.empty:
            global colnames
            colnames = df_up.columns
            global col_options
            col_options = df_up.columns
        return df_up.to_json(date_format='iso', orient='split')


@app.callback(
    [Output('datatable', 'data'),
     Output('datatable', 'columns')],
    [Input('datatable', "page_current"),
     Input('datatable', "page_size"),
     Input('datatable', 'sort_by'),
     Input('datatable', "filter_query"),
     Input('datatable-row-count', 'value'),
     Input('no-data', 'children')])
def update_table(page_current, page_size, sort_by, filter, row_count_value, data):
    """
    This is the collback function to update the datatable
    with the required filtered, sorted, extended values
    :param page_current: Current page number
    :param page_size: Page size
    :param sort_by: Column selected for sorting
    :param filter: Value entered in the filter
    :param row_count_value: Number of rows
    :param data: dataframe
    :return: processed data aand column values
    """
    # If uploaded dataframe is not empty use that, otherwise
    # use the default dataframe
    if not df_up.empty:
        # df_temp = pd.read_json(data, orient='split')
        df_tab = df_up
    else:
        df_tab = df

    # Setting the page size as row count value
    if row_count_value is not None:
        page_size = row_count_value

    # Applying sort logic
    if len(sort_by):
        dff = df_tab.sort_values(
            sort_by[0]['column_id'],
            ascending=sort_by[0]['direction'] == 'asc',
            inplace=False
        )
    else:
        # No sort is applied
        dff = df_tab

    # Filter logic
    if filter is not None:
        filtering_expressions = filter.split(' && ')

        for filter_part in filtering_expressions:
            col_name, operator, filter_value = split_filter_part(filter_part)

            if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
                # these operators match pandas series operator method names
                dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
            elif operator == 'contains':
                dff = dff.loc[dff[col_name].str.contains(filter_value)]
            elif operator == 'datestartswith':
                # this is a simplification of the front-end filtering logic,
                # only works with complete fields in standard format
                dff = dff.loc[dff[col_name].str.startswith(filter_value)]

    # if selected_cols is not None:
    #     if len(selected_cols) != 0:
    #         return dff[selected_cols].iloc[
    #                page_current * page_size:(page_current + 1) * page_size
    #                ].to_dict('records')
    #     else:
    #         return dff.iloc[
    #                page_current * page_size:(page_current + 1) * page_size
    #                ].to_dict('records')
    # else:

    # Rounding the float values to 2 decimal places
    dff = dff.round(2)

    return [dff.iloc[
            page_current * page_size:(page_current + 1) * page_size
            ].to_dict('records'),
            [{"name": [i, j], "id": i} for i, j in zip(df_tab.columns, [str(x) for x in df_tab.dtypes.to_list()])]]


########################################################################################################################


@app.callback(
    Output("plot-scatter", "figure"),
    [Input("xlab-scat", "value"),
     Input("ylab-scat", "value"),
     Input("col-scat", "value"),
     Input("siz-scat", "value"),
     Input("fac-scat-row", "value"),
     Input("fac-scat-col", "value"),
     Input("trnd-scat", "value")])
def update_scatter(x, y, color, size, facet_row, facet_col, trend):
    """
    This is the callback function to update the scatter plot based on the
    input given by the user
    :param x: X label value from the dropdown
    :param y: Y label value from the dropdown
    :param color: color variable value selected from the dropdown
    :param size: size variable value selected from the dropdown
    :param facet_row: facet-row value selected from the drop-down
    :param facet_col: facet-column value selected from the dropdown
    :param trend: trendline type selected from the dropdown
    :return: scatter plot object of plotly express
    """
    # If the uploaded dataframe is empty we will pass the default dataframe
    # to the scatter plot function else we pass the uploaded dataframe to the
    # the function
    if df_up.empty:
        return px.scatter(
            df,
            x=x,
            y=y,
            color=color,
            size=size,
            facet_row=facet_row,
            facet_col=facet_col,
            trendline=trend,
            height=700)
    elif not df_up.empty:
        return px.scatter(
            df_up,
            x=x,
            y=y,
            color=color,
            size=size,
            facet_row=facet_row,
            facet_col=facet_col,
            trendline=trend,
            height=700)


@app.callback(
    [Output('xlab-scat', 'options'),
     Output('ylab-scat', 'options'),
     Output('col-scat', 'options'),
     Output('siz-scat', 'options'),
     Output('fac-scat-row', 'options'),
     Output('fac-scat-col', 'options'),
     Output('load-button-scat', 'disabled')],
    [Input('load-button-scat', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


@app.callback(
    Output("plot-bar", "figure"),
    [Input("xlab-bar", "value"),
     Input("ylab-bar", "value"),
     Input("col-bar", "value"),
     Input("fac-bar-row", "value"),
     Input("fac-bar-col", "value")])
def update_bar(x, y, color, facet_row, facet_col):
    """
    This is a callback function to update the bar plot based on the values
    choosen by the user from the dropdown
    :param x: X label value from the dropdown
    :param y: Y label value from the dropdown
    :param color: color variable value selected from the dropdown
    :param type: bar type variable value selected from the dropdown
    :param facet_row: facet-row  variable value selected from the dropdown
    :param facet_col: facet-column variable value selected from the dropdown
    :return: bar plot object of plotly express
    """
    # If the uploaded dataframe is empty we will pass the default dataframe
    # to the bar plot function else we pass the uploaded dataframe to the
    # the function
    if df_up.empty:
        return px.bar(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)
    else:
        return px.bar(
            df_up,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)


@app.callback(
    [Output('xlab-bar', 'options'),
     Output('ylab-bar', 'options'),
     Output('col-bar', 'options'),
     Output('fac-bar-row', 'options'),
     Output('fac-bar-col', 'options'),
     Output('load-button-bar', 'disabled')],
    [Input('load-button-bar', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


@app.callback(
    Output("plot-box", "figure"),
    [Input("xlab-box", "value"),
     Input("ylab-box", "value"),
     Input("col-box", "value"),
     Input("fac-box-row", "value"),
     Input("fac-box-col", "value")])
def update_box(x, y, color, facet_row, facet_col):
    """
    This is a callback function to update the box plot based on the values
    choosen by the user from the dropdown
    :param x: X label value from the dropdown
    :param y: Y label value from the dropdown
    :param color: Color variable value selected from the dropdown
    :param facet_row: facet-row variable value selected from the dropdown
    :param facet_col: facet-column varible value selected from the dropdown
    :return: box plot object of plotly express
    """
    # If the uploaded dataframe is empty we will pass the default dataframe
    # to the box plot function else we pass the uploaded dataframe to the
    # the function
    if df_up.empty:
        return px.box(
            df,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)
    else:
        return px.box(
            df_up,
            x=x,
            y=y,
            color=color,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)


@app.callback(
    [Output('xlab-box', 'options'),
     Output('ylab-box', 'options'),
     Output('col-box', 'options'),
     Output('fac-box-row', 'options'),
     Output('fac-box-col', 'options'),
     Output('load-button-box', 'disabled')],
    [Input('load-button-box', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


@app.callback(
    Output("plot-heat", "figure"),
    [Input("xlab-heat", "value"),
     Input("ylab-heat", "value"),
     Input("fac-heat-row", "value"),
     Input("fac-heat-col", "value")])
def update_heatmap(x, y, facet_row, facet_col):
    """
    This is a callback function to update the heatmap plot based on the values
    choosen by the user from the dropdown
    :param x: X label value from the dropdown
    :param y: Y label value from the dropdown
    :param facet_row: facet-row variable value selected from the dropdown
    :param facet_col: facet-column varible value selected from the dropdown
    :return: box plot object of plotly express
    """
    # If the uploaded dataframe is empty we will pass the default dataframe
    # to the heatmap plot function else we pass the uploaded dataframe to the
    # the function
    if df_up.empty:
        return px.density_heatmap(
            df,
            x=x,
            y=y,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)
    else:
        return px.density_heatmap(
            df_up,
            x=x,
            y=y,
            facet_row=facet_row,
            facet_col=facet_col,
            height=700)


@app.callback(
    [Output('xlab-heat', 'options'),
     Output('ylab-heat', 'options'),
     Output('fac-heat-row', 'options'),
     Output('fac-heat-col', 'options'),
     Output('load-button-heat', 'disabled')],
    [Input('load-button-heat', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


########################################################################################################################


@app.callback(
    [Output("plot-norm", "figure"),
     Output("norm-val1", "children"),
     Output("norm-val2", "children"),
     Output("norm-val3", "children"),
     Output("norm-val4", "children"),
     Output("norm-val5", "children")],
    [Input("norm-test", "value"),
     Input("norm-var", "value")])
def update_norm(test, var):
    """
    This a callback function that returns the statistics weather
    the selected variable is normally distributed or not based on
    a hardcoded threshold (p = 0.05)
    :param test: Normality test selected
    :param var: Variable selected
    :return: histogram and test result
    """
    # Selecting uploaded dataframe if it not empty else select
    # default dataframe
    if df_up.empty:
        df_norm = df
    else:
        df_norm = df_up

    # Shapiro-Wilk test
    # The null-hypothesis of this test is that the population is
    # normally distributed. Thus, if the p value is less than the
    # chosen alpha level, then the null hypothesis is rejected and
    # there is evidence that the data tested are not normally
    # distributed. On the other hand, if the p value is greater than
    # the chosen alpha level, then the null hypothesis that the data
    # came from a normally distributed population can not be rejected
    # (e.g., for an alpha level of .05, a data set with a p value of
    # less than .05 rejects the null hypothesis that the data are from
    # a normally distributed population).
    if test == "Shapiro-Wilk" and var is not None:
        stat, p = shapiro(df_norm[var])
        fig = px.histogram(df_norm, x=var, height=500)
        if p > 0.05:
            result1 = 'Probably Gaussian : stat=%.3f, p=%.3f' % (stat, p)
            result2 = ""
            result3 = ""
            result4 = ""
            result5 = ""
            return fig, result1, result2, result3, result4, result5
        else:
            result1 = 'Probably not Gaussian : stat=%.3f, p=%.3f' % (stat, p)
            result2 = ""
            result3 = ""
            result4 = ""
            result5 = ""
            return fig, result1, result2, result3, result4, result5

    # D’Agostino’s K^2
    # The test is based on transformations of the sample kurtosis and skewness,
    # and has power only against the alternatives that the distribution is skewed
    # and/or kurtic.
    elif test == "D’Agostino’s K^2" and var is not None:
        stat, p = normaltest(df_norm[var])
        fig = px.histogram(df_norm, x=var, height=500)
        if p > 0.05:
            result1 = 'Probably Gaussian : stat=%.3f, p=%.3f' % (stat, p)
            result2 = ""
            result3 = ""
            result4 = ""
            result5 = ""
            return fig, result1, result2, result3, result4, result5
        else:
            result1 = 'Probably not Gaussian : stat=%.3f, p=%.3f' % (stat, p)
            result2 = ""
            result3 = ""
            result4 = ""
            result5 = ""
            return fig, result1, result2, result3, result4, result5

    # Anderson-Darling
    # The Anderson–Darling test is a statistical test of whether a given sample of
    # data is drawn from a given probability distribution. In its basic form, the
    # test assumes that there are no parameters to be estimated in the distribution
    # being tested, in which case the test and its set of critical values is
    # distribution-free. However, the test is most often used in contexts where a
    # family of distributions is being tested, in which case the parameters of that
    # family need to be estimated and account must be taken of this in adjusting either
    # the test-statistic or its critical values.
    elif test == "Anderson-Darling" and var is not None:
        res = anderson(df_norm[var])
        fig = px.histogram(df_norm, x=var, height=500)

        if res.statistic < res.critical_values[0]:
            result1 = 'Probably Gaussian at the %.1f%% level,' % res.significance_level[0]
        else:
            result1 = 'Probably not Gaussian at the %.1f%% level,' % res.significance_level[0]
        if res.statistic < res.critical_values[1]:
            result2 = 'Probably Gaussian at the %.1f%% level,' % res.significance_level[1]
        else:
            result2 = 'Probably not Gaussian at the %.1f%% level,' % res.significance_level[1]
        if res.statistic < res.critical_values[2]:
            result3 = 'Probably Gaussian at the %.1f%% level,' % res.significance_level[2]
        else:
            result3 = 'Probably not Gaussian at the %.1f%% level,' % res.significance_level[2]
        if res.statistic < res.critical_values[3]:
            result4 = 'Probably Gaussian at the %.1f%% level,' % res.significance_level[3]
        else:
            result4 = 'Probably not Gaussian at the %.1f%% level,' % res.significance_level[3]
        if res.statistic < res.critical_values[4]:
            result5 = 'Probably Gaussian at the %.1f%% level,' % res.significance_level[4]
        else:
            result5 = 'Probably not Gaussian at the %.1f%% level,' % res.significance_level[4]
    return fig, result1, result2, result3, result4, result5


@app.callback(
    [Output('norm-var', 'options'),
     Output('load-button-norm', 'disabled')],
    [Input('load-button-norm', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options]]


@app.callback(
    [Output("plot-corr", "figure"),
     Output("corr-val1", "children")],
    [Input("corr-test", "value"),
     Input("corr-var1", "value"),
     Input("corr-var2", "value")])
def update_corr(test, var1, var2):
    """
    This a callback function that returns the statistics weather
    the selected variables are correlated or not based on
    a hardcoded threshold (p = 0.05)
    :param test: Correlation test selected
    :param var1: Variable one selected
    :param var2: Variable two selected
    :return: scatter plot and test result
    """
    # Selecting uploaded dataframe if it not empty else select
    # default dataframe
    if df_up.empty:
        df_corr = df
    else:
        df_corr = df_up

    # Pearson test is a measure of the linear correlation between two variables X and Y.
    # According to the Cauchy–Schwarz inequality it has a value between +1 and −1, where
    # 1 is total positive linear correlation, 0 is no linear correlation, and −1 is total
    # negative linear correlation.
    if test == "Pearson" and var1 is not None and var2 is not None:
        stat, p = pearsonr(df_corr[var1], df_corr[var2])
        fig = px.scatter(df_corr, x=var1, y=var2, height=500)
        if p > 0.05:
            result1 = 'Probably independent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably dependent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1

    # Spearman test often denoted by the Greek letter {(rho) or
    # as r_{s}, is a nonparametric measure of rank correlation
    # (statistical dependence between the rankings of two variables). It assesses how
    # well the relationship between two variables can be described using a monotonic function.
    elif test == "Spearman" and var1 is not None and var2 is not None:
        stat, p = spearmanr(df_corr[var1], df_corr[var2])
        fig = px.scatter(df_corr, x=var1, y=var2, height=500)
        if p > 0.05:
            result1 = 'Probably independent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably dependent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1

    # Intuitively, the Kendall correlation between two variables will be high when observations
    # have a similar (or identical for a correlation of 1) rank (i.e. relative position label
    # of the observations within the variable: 1st, 2nd, 3rd, etc.) between the two variables,
    # and low when observations have a dissimilar (or fully different for a correlation of −1)
    # rank between the two variables.
    elif test == "Kendall" and var1 is not None and var2 is not None:
        stat, p = kendalltau(df_corr[var1], df_corr[var2])
        fig = px.scatter(df_corr, x=var1, y=var2, height=500)
        if p > 0.05:
            result1 = 'Probably independent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably dependent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1

    # Chi Squared test  is a statistical test applied to sets of categorical data to evaluate
    # how likely it is that any observed difference between the sets arose by chance. It is the
    # most widely used of many chi-squared tests (e.g., Yates, likelihood ratio, portmanteau
    # test in time series, etc.) – statistical procedures whose results are evaluated by
    # reference to the chi-squared distribution.
    elif test == "Chi-Squared" and var1 is not None and var2 is not None:
        stat, p, dof, expected = chi2_contingency(df_corr[[var1, var2]])
        fig = px.scatter(df_corr, x=var1, y=var2, height=500)
        if p > 0.05:
            result1 = 'Probably independent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably dependent : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1


@app.callback(
    [Output('corr-var1', 'options'),
     Output('corr-var2', 'options'),
     Output('load-button-corr', 'disabled')],
    [Input('load-button-corr', 'n_clicks')])
def update_labs(n):
    """
        This is a callback function to update the dropdown values as
        per columns of the uploaded dataframe on clicking of the load
        button
        :param n: No. of button clicks
        :return: updated dropdown values and a boolean value to disable
        the load button
        """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


@app.callback(
    [Output("plot-para", "figure"),
     Output("para-val1", "children")],
    [Input("para-test", "value"),
     Input("para-var1", "value"),
     Input("para-var2", "value")])
def update_para(test, var1, var2):
    """
    This a callback function that returns the statistics weather
    the selected variables are correlated or not based on
    a hardcoded threshold (p = 0.05)
    :param test: Correlation test selected
    :param var1: Variable one selected
    :param var2: Variable two selected
    :return: scatter plot and test result
    """
    # Selecting uploaded dataframe if it not empty else select
    # default dataframe
    if df_up.empty:
        df_para = df
    else:
        df_para = df_up

    # The t-test is any statistical hypothesis test in which the test statistic
    # follows a Student's t-distribution under the null hypothesis. A t-test is
    # most commonly applied when the test statistic would follow a normal distribution
    # if the value of a scaling term in the test statistic were known. When the scaling
    # term is unknown and is replaced by an estimate based on the data, the test
    # statistics (under certain conditions) follow a Student's t distribution. The
    # t-test can be used, for example, to determine if the means of two sets of data
    # are significantly different from each other
    if test == "Student t-test" and var1 is not None and var2 is not None:
        stat, p = ttest_ind(df_para[var1], df_para[var2])
        df_sub = df_para[[' index', var1, var2]]
        df_melt = pd.melt(df_sub, id_vars=' index')
        fig = px.histogram(df_melt, x='value', color='variable', height=500)
        if p > 0.05:
            result1 = 'Probably the same distribution : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably different distributions : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1

    # This function gives a paired Student t test, confidence intervals for the difference between
    # a pair of means and, optionally, limits of agreement for a pair of samples
    elif test == "Paired Student t-test" and var1 is not None and var2 is not None:
        stat, p = ttest_rel(df_para[var1], df_para[var2])
        df_sub = df_para[[' index', var1, var2]]
        df_melt = pd.melt(df_sub, id_vars=' index')
        fig = px.histogram(df_melt, x='value', color='variable', height=500)
        if p > 0.05:
            result1 = 'Probably the same distribution : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably different distributions : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1

    # The ANOVA is based on the law of total variance, where the observed variance in a
    # particular variable is partitioned into components attributable to different sources
    # of variation. In its simplest form, ANOVA provides a statistical test of whether two
    # or more population means are equal, and therefore generalizes the t-test beyond two means.
    elif test == "ANOVA" and var1 is not None and var2 is not None:
        stat, p = f_oneway(df_para[var1], df_para[var2])
        df_sub = df_para[[' index', var1, var2]]
        df_melt = pd.melt(df_sub, id_vars=' index')
        fig = px.histogram(df_melt, x='value', color='variable', height=500)
        if p > 0.05:
            result1 = 'Probably the same distribution : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1
        else:
            result1 = 'Probably different distributions : stat=%.3f, p=%.3f' % (stat, p)
            return fig, result1


@app.callback(
    [Output('para-var1', 'options'),
     Output('para-var2', 'options'),
     Output('load-button-para', 'disabled')],
    [Input('load-button-para', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


########################################################################################################################


@app.callback(
    [Output('lin-mod-tab1', "children"),
     Output('lin-mod-tab3', "children"),
     Output('lin-mod-tab2', "children")],
    [Input('lin-pre', "value"),
     Input('lin-tar', "value"),
     Input('run-button-lin', "n_clicks")])
def run_lin(predictors, target, run):
    """
    This is a callback function for running a Linear regression based
    on the model parameters chosen by the user
    :param predictors: Predictor variables selected from the dropdown
    :param target: Target variable selected from the dropdown
    :param run: No. of clicks of the run button
    :return: Three tables containing various regression statistics
    """
    # Selecting uploaded dataframe if it not empty else select
    # default dataframe
    if df_up.empty:
        df_lin = df
    else:
        df_lin = df_up

    # If the run button click count is greater than 0 we execute the following
    # code
    if run:

        # The statsmodel package does not accept a categorical variable as target
        # for a logistic regression so we need to encode the target column (if categorical)
        # into numeric values and then run the regression
        # Checking if the data type of the target is object. If yes, we convert it to
        # categorical and encode it to numbers
        if "object" in str(df_lin[[target]].dtypes):
            df_lin[target] = pd.Categorical(df_lin[target])
            df_lin[target] = df_lin[target].cat.codes

        # Now we normalize the target such that all the elements lie between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
        df_lin[[target]] = min_max_scaler.fit_transform(df_lin[[target]])

        # Creating a R type formula out of the prodictors and target
        pred_str = ' + '.join(map(str, predictors))
        formula = ' ~ '.join([target, pred_str])

        # Running the ordinary least squares regression for our formula
        model = smf.ols(formula=formula, data=df_lin).fit()

        # predictions = model.predict(X)

        # Getting the required data into datafarmes
        results_as_html = model.summary().tables[0].as_html()
        results_as_html2 = model.summary().tables[2].as_html()
        results_as_html3 = model.summary().tables[1].as_html()
        df_summ = pd.read_html(results_as_html, header=0)[0]
        df_summ2 = pd.read_html(results_as_html2, header=0)[0]
        df_summ3 = pd.read_html(results_as_html3, header=0)[0]

        return [
            html.Div([
                dash_table.DataTable(
                    id='table1',
                    columns=[{"name": i, "id": i} for i in df_summ.columns],
                    data=df_summ.to_dict("rows"),
                    style_header={
                        'backgroundColor': 'white'
                    },
                    style_header_conditional=[
                        {'if': {'column_id': 'Dep. Variable:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'R-squared (uncentered):'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'R-squared:'},
                         'backgroundColor': '#DCDCDC'}
                    ],
                    style_data_conditional=[
                        {'if': {'column_id': 'Dep. Variable:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'R-squared (uncentered):'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'R-squared:'},
                         'backgroundColor': '#DCDCDC'}
                    ])
            ]),
            html.Div([
                dash_table.DataTable(
                    id='table1',
                    columns=[{"name": i, "id": i} for i in df_summ3.columns],
                    data=df_summ3.to_dict("rows"),
                    style_header={
                        'backgroundColor': '#DCDCDC'})
            ]),
            html.Div([
                dash_table.DataTable(
                    id='table2',
                    columns=[{"name": i, "id": i} for i in df_summ2.columns],
                    data=df_summ2.to_dict("rows"),
                    style_header={
                        'backgroundColor': 'white'},
                    style_header_conditional=[
                        {'if': {'column_id': 'Omnibus:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'Durbin-Watson:'},
                         'backgroundColor': '#DCDCDC'}
                    ],
                    style_data_conditional=[
                        {'if': {'column_id': 'Omnibus:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'Durbin-Watson:'},
                         'backgroundColor': '#DCDCDC'}]
                )
            ])
        ]


@app.callback(
    [Output('lin-pre', 'options'),
     Output('lin-tar', 'options'),
     Output('load-button-lin', 'disabled')],
    [Input('load-button-lin', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


@app.callback(
    [Output('log-mod-tab1', "children"),
     Output('log-mod-tab2', "children")],
    [Input('log-pre', "value"),
     Input('log-tar', "value"),
     Input('run-button-log', "n_clicks")])
def run_log(predictors, target, run):
    """
    This is a callback function for running a Logistic regression based
    on the model parameters chosen by the user
    :param predictors: Predictor variables selected from the dropdown
    :param target: Target variable selected from the dropdown
    :param run: No. of clicks of the run button
    :return: Two tables containing various regression statistics
    """
    # Selecting uploaded dataframe if it not empty else select
    # default dataframe
    if df_up.empty:
        df_log = df
    else:
        df_log = df_up
    if run:
        # X = df[predictors].values.tolist()
        # y = df[target].values.tolist()
        # model = sm.OLS(y, X).fit()

        # The statsmodel package does not accept a categorical variable as target
        # for a logistic regression so we need to encode the target column (if categorical)
        # into numeric values and then run the regression
        # Checking if the data type of the target is object. If yes, we convert it to
        # categorical and encode it to numbers
        if "object" in str(df_log[[target]].dtypes):
            df_log[target] = pd.Categorical(df_log[target])
            df_log[target] = df_log[target].cat.codes

        # Now we normalize the target such that all the elements lie between 0 and 1
        min_max_scaler = preprocessing.MinMaxScaler()
        df_log[[target]] = min_max_scaler.fit_transform(df_log[[target]])

        # Creating a R style formula for model input
        pred_str = ' + '.join(map(str, predictors))
        formula = ' ~ '.join([target, pred_str])

        # Fitting the Logistic regression model for our current data
        model = smf.logit(formula=formula, data=df_log).fit()

        # predictions = model.predict(X)

        # Getting the required data into datafarmes
        results_as_html = model.summary().tables[0].as_html()
        results_as_html2 = model.summary().tables[1].as_html()
        df_summ = pd.read_html(results_as_html, header=0)[0]
        df_summ2 = pd.read_html(results_as_html2, header=0)[0]

        return [
            html.Div([
                dash_table.DataTable(
                    id='table1',
                    columns=[{"name": i, "id": i} for i in df_summ.columns],
                    data=df_summ.to_dict("rows"),
                    style_header={
                        'backgroundColor': 'white'
                    },
                    style_header_conditional=[
                        {'if': {'column_id': 'Dep. Variable:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'No. Observations:'},
                         'backgroundColor': '#DCDCDC'}
                    ],
                    style_data_conditional=[
                        {'if': {'column_id': 'Dep. Variable:'},
                         'backgroundColor': '#DCDCDC'},
                        {'if': {'column_id': 'No. Observations:'},
                         'backgroundColor': '#DCDCDC'}])
            ]),
            html.Div([
                dash_table.DataTable(
                    id='table2',
                    columns=[{"name": i, "id": i} for i in df_summ2.columns],
                    data=df_summ2.to_dict("rows"),
                    style_header={
                        'backgroundColor': '#DCDCDC'
                    },
                    style_header_conditional=[
                        {'if': {'column_id': 'Unnamed: 0'},
                         'backgroundColor': '#DCDCDC'},
                    ],
                )
            ])
        ]


@app.callback(
    [Output('log-pre', 'options'),
     Output('log-tar', 'options'),
     Output('load-button-log', 'disabled')],
    [Input('load-button-log', 'n_clicks')])
def update_labs(n):
    """
    This is a callback function to update the dropdown values as
    per columns of the uploaded dataframe on clicking of the load
    button
    :param n: No. of button clicks
    :return: updated dropdown values and a boolean value to disable
    the load button
    """
    # Once the load button is clicked return the values of the uploaded
    # dataframe else return default dataframe columns. Also the button
    # disable flag is set in the first case
    if n is not None:
        return [[{'label': i, 'value': i} for i in colnames],
                [{'label': i, 'value': i} for i in colnames],
                True]
    else:
        return [[{'label': i, 'value': i} for i in col_options],
                [{'label': i, 'value': i} for i in col_options]]


########################################################################################################################


# Finally we run the app on our flask server
if __name__ == '__main__':
    app.server.run(debug=True)
########################################################################################################################

