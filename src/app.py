import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
import pandas as pd
import webbrowser
from threading import Timer
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Function to open the browser automatically
def open_browser():
    webbrowser.open_new("http://127.0.0.1:8050/")

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# Update county_df with the new Grant Type column
county_df = pd.read_csv(r'FastTrack_Data_10-16-24.csv')

county_df['Landed Month / Year'] = pd.to_datetime(county_df['Landed Month / Year'])

# Extracting min and max year for the year slider
min_year = county_df['Landed Month / Year'].dt.year.min()
max_year = county_df['Landed Month / Year'].dt.year.max()

# Summarize county_df by County, include the Region
def summarize_by_county(df):
    county_summary = df.groupby(['County', 'Region']).agg({
        'New Jobs': 'sum',
        'Capital Investment': 'sum',
        'FIPS': 'first'
    }).reset_index()

    # Format New Jobs and Capital Investment with comma and dollar sign
    county_summary['Total New Jobs'] = county_summary['New Jobs'].apply(lambda x: f"{x:,}")
    county_summary['Total Capital Investment'] = county_summary['Capital Investment'].apply(lambda x: f"${x:,.0f}")
    county_summary.sort_values(by='New Jobs', ascending=False, inplace=True)
    
    return county_summary

# Filter the dataframe based on selected filters
def filter_data(df, regions, project_types, year_range, grant_types):
    filtered_df = df.copy()

    # Filter by Region
    if regions:
        filtered_df = filtered_df[filtered_df['Region'].isin(regions)]

    # Filter by Project Type
    if project_types:
        filtered_df = filtered_df[filtered_df['Project Type'].isin(project_types)]

    # Filter by Year Range
    filtered_df = filtered_df[filtered_df['Landed Month / Year'].dt.year.between(year_range[0], year_range[1])]

    # Filter by Grant Type
    if grant_types:
        filtered_df = filtered_df[filtered_df['Grant Type'].apply(lambda grants: any(grant in grants for grant in grant_types))]

    return filtered_df

# Generate county map
def generate_county_map(df, values_column, title):
    fig = px.choropleth(df, 
                        geojson='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json', 
                        locations='FIPS', 
                        color=values_column,
                        color_continuous_scale='Blues',
                        scope="usa",
                        labels={values_column: values_column},
                        hover_data={'County': True})

    fig.update_geos(center=dict(lon=-85.90, lat=35.5), projection_scale=7.25)
    fig.update_layout(
        autosize=True,
        title_text=title,
        title_x=0.5,
        title_y=0.95,
        title_font=dict(size=24),
        template="plotly_dark",
        height=600,
        margin=dict(l=0, r=0, t=0, b=0),
        coloraxis_colorbar=dict(
            len=0.8,
            x=0.5,
            y=0.05,
            orientation='h',
            title_side="top"
        )
    )
    
    return fig

# Generate scatter plot for New Jobs and Capital Investment
def generate_scatter_plot(df, y_column, title):
    df_yearly = df.groupby(df['Landed Month / Year'].dt.year)[y_column].sum().reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_yearly['Landed Month / Year'],
        y=df_yearly[y_column],
        mode='lines+markers',
        name=title,
        line=dict(shape='linear')
    ))

    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Year",
        yaxis_title=y_column,
        height=400,
        margin=dict(l=10, r=10, t=50, b=30)
    )
    
    return fig

# Layout for the app
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#9e9ead"},
    children=[
        # Title row
        dbc.Row(
            dbc.Col(
                html.H1("Tennessee: Forecasting Employment Growth", className="text-center"),
                width={"size": 6, "offset": 3},
                className="d-flex justify-content-center align-items-center mb-4"
            ),
        ),
        
        # Filters panel row
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dbc.Row(
                            [
                                dbc.Col(dcc.Dropdown(id='region-filter', options=[{'label': f'Region {i}', 'value': f'Region {i}'} for i in range(1, 5)], multi=True, placeholder='Filter Region'), width=3),
                                dbc.Col(dcc.Dropdown(id='project-type-filter', options=[{'label': ptype, 'value': ptype} for ptype in ['Recruitment', 'Expansion', 'Expansion New Location', 'New Startup']], multi=True, placeholder='Filter Project Type'), width=3),
                                dbc.Col(dcc.RangeSlider(id='year-slider', min=min_year, max=max_year, marks={str(year): str(year) for year in range(min_year, max_year + 1)}, value=[min_year, max_year], step=1), width=4),
                                dbc.Col(dcc.Dropdown(id='grant-type-filter', options=[{'label': grant, 'value': grant} for grant in ['FJTAP', 'FIDP', 'ED']], multi=True, placeholder='Filter Grant Type'), width=2)
                            ]
                        )
                    ),
                    className="mb-4"
                ),
                width=12
            )
        ),

        # Map and top counties table row
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='county-map'), width=8, style={"width": "65vw", "margin-left": "0px", "margin-right": "auto"}),
                dbc.Col(
                    dbc.Container(
                        [
                            html.Div([
                                html.H4("Top Counties for New Jobs"),
                                dbc.Table.from_dataframe(pd.DataFrame(), bordered=True, hover=True, responsive=True, id='top-counties-table')
                            ], style={"margin-bottom": "20px"})
                        ]
                    ),
                    width=4
                )
            ]
        ),

        # Scatter plots row and statistics box
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='new-jobs-plot'), width=4, style={"margin-top": "30px"}),
                dbc.Col(dcc.Graph(id='capital-investment-plot'), width=4, style={"margin-top": "30px"}),
                dbc.Col(
                    html.Div(
                        [html.H4("Statistics"), html.Div(id='statistics-box')],
                        style={"border": "1px solid #ccc", "padding": "10px", "background-color": "#f8f9fa", "margin-top": "-22vh"}
                    ),
                    width=4
                )
            ]
        )
    ]
)

# Updated callback function to create compartmentalized statistics box content
@app.callback(
    [Output('county-map', 'figure'),
     Output('top-counties-table', 'children'),
     Output('new-jobs-plot', 'figure'),
     Output('capital-investment-plot', 'figure'),
     Output('statistics-box', 'children')],
    [Input('region-filter', 'value'),
     Input('project-type-filter', 'value'),
     Input('year-slider', 'value'),
     Input('grant-type-filter', 'value')]
)
def update_dashboard(selected_regions, selected_project_types, selected_years, selected_grants):
    filtered_df = filter_data(county_df, selected_regions, selected_project_types, selected_years, selected_grants)
    county_summary_df = summarize_by_county(filtered_df)

    # Generate updated visuals
    county_map_fig = generate_county_map(county_summary_df, 'New Jobs', 'County-wise New Jobs')
    top_counties_table = dbc.Table.from_dataframe(
        county_summary_df[['County', 'Region', 'Total New Jobs', 'Total Capital Investment']].head(5),
        bordered=True,
        hover=True,
        responsive=True
    )
    new_jobs_plot_fig = generate_scatter_plot(filtered_df, 'New Jobs', 'New Jobs Over Time')
    investment_plot_fig = generate_scatter_plot(filtered_df, 'Capital Investment', 'Capital Investment Over Time')

    # Statistics content calculation
    unique_counties = filtered_df['FIPS'].nunique()
    total_projects = len(filtered_df)

    # Grant type counts - ensure all grant types are represented even if they have 0 count
    grant_type_counts = {grant: (filtered_df[grant] > 0).sum() if grant in filtered_df else 0 for grant in ['FJTAP', 'FIDP', 'ED']}
    
    # Ensure regions are always displayed in order 1 through 4
    region_order = [f'Region {i}' for i in range(1, 5)]
    region_counts = {region: filtered_df['Region'].value_counts().get(region, 0) for region in region_order}

    # Ensure all project types are represented even if they have 0 count
    all_project_types = ['Recruitment', 'Expansion', 'Expansion New Location', 'New Startup']
    project_type_counts = {ptype: filtered_df['Project Type'].value_counts().get(ptype, 0) for ptype in all_project_types}

    # Compartmentalized statistics content using Cards
    statistics_content = dbc.Container(
        [
            dbc.Card(
                dbc.CardBody([
                    html.H5(f"Unique Counties Analyzed: {unique_counties}", className="card-title"),
                    html.H5(f"Total Number of Projects: {total_projects}", className="card-title")
                ]),
                className="mb-3"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H6("Projects Breakdown by Grant Type:", className="card-subtitle"),
                    html.Ul([html.Li(f"{grant}: {count}") for grant, count in grant_type_counts.items()])
                ]),
                className="mb-3"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H6("Projects Breakdown by Region:", className="card-subtitle"),
                    html.Ul([html.Li(f"{region}: {region_counts[region]}") for region in region_order])
                ]),
                className="mb-3"
            ),
            dbc.Card(
                dbc.CardBody([
                    html.H6("Projects Breakdown by Project Type:", className="card-subtitle"),
                    html.Ul([html.Li(f"{ptype}: {count}") for ptype, count in project_type_counts.items()])
                ]),
                className="mb-3"
            )
        ],
        style={"padding": "10px"}
    )

    return county_map_fig, top_counties_table, new_jobs_plot_fig, investment_plot_fig, statistics_content

# Run the app
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True)