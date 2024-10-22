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
county_df = pd.read_csv('FastTrack_Data_10-22-24.csv')
regions = pd.read_excel("Forecasting_Employment.xlsx", sheet_name='Regions')

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

# Function to get all unique counties by selected regions and handle missing values and duplicates
def get_unique_counties_by_region(selected_regions):
    applicable_counties = []
    for region in selected_regions:
        # Drop NaN and convert to list, concatenate across regions
        applicable_counties += regions[region].dropna().tolist()
    return list(set(applicable_counties))  # Use set to remove duplicates

# Generate county map with text overlay for custom statistics
def generate_county_map(df, values_column, title, stats):
    fig = px.choropleth(
        df,
        geojson='https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json',
        locations='FIPS',
        color=values_column,
        color_continuous_scale='Blues',
        scope="usa",
        labels={values_column: values_column},
        hover_data={'County': True}
    )

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
        ),
        annotations=[
            dict(
                x=0.0,
                y=0.9,
                xref='paper',
                yref='paper',
                text=f"<b>Counties</b>:<br>{stats['counties']} out of {stats['total_counties']}",
                showarrow=False,
                font=dict(size=20, color="white"),
                align='center'
            ),
            dict(
                x=0.0,
                y=0.8,
                xref='paper',
                yref='paper',
                text=f"<b>Projects</b>:<br>{stats['projects']}",
                showarrow=False,
                font=dict(size=20, color="white"),
                align='center'
            ),
            dict(
                x=0.0,
                y=0.7,
                xref='paper',
                yref='paper',
                text=f"<b>New Jobs</b>:<br>{stats['new_jobs']}",
                showarrow=False,
                font=dict(size=20, color="white"),
                align='center'
            ),
            dict(
                x=0.0,
                y=0.55,
                xref='paper',
                yref='paper',
                text=f"<b>Capital Investment</b>:<br>{stats['capital_investment']}",
                showarrow=False,
                font=dict(size=20, color="white"),
                align='center'
            )
        ]
    )
    
    return fig

# Generate cumulative scatter plot for New Jobs and Capital Investment
def generate_scatter_plot(df, y_column, title):
    df_yearly = df.groupby(df['Landed Month / Year'].dt.year)[y_column].sum().cumsum().reset_index()  # Use cumsum() for cumulative sum

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_yearly['Landed Month / Year'],
        y=df_yearly[y_column].round(2),  # Round to 2 decimal places for better formatting
        mode='lines+markers',
        name=title,
        line=dict(shape='linear')
    ))

    # Update layout to format the y-axis with 2 decimal places
    fig.update_layout(
        template="plotly_dark",
        title=title,
        xaxis_title="Year",
        yaxis_title=y_column,
        height=400,
        margin=dict(l=10, r=10, t=50, b=30),
    )
    
    return fig

# Generate top projects table with formatted values
def generate_top_projects_table(df):
    df['Year'] = df['Landed Month / Year'].dt.year  # Add 'Year' column
    
    # Format 'New Jobs' with a comma separator and 'Capital Investment' with $ and comma separator
    df['New Jobs '] = df['New Jobs'].apply(lambda x: f"{x:,}")
    df['Capital Investment '] = df['Capital Investment'].apply(lambda x: f"${x:,.0f}")
    
    # Sort by 'New Jobs' and return the formatted table
    df = df.sort_values(by='New Jobs', ascending=False)
    return df[['Company', 'County', 'Year', 'New Jobs ', 'Capital Investment ']].head(5)

# Layout for the app
app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#8a8a8a"},
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

        # Map and top projects table row
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id='county-map'), width=8, style={"width": "65vw", "margin-left": "0px", "margin-right": "auto"}),
                dbc.Col(
                    dbc.Container(
                        [
                            html.Div([
                                html.H4("Top 5 Projects for New Jobs"),
                                dbc.Table.from_dataframe(pd.DataFrame(), bordered=True, hover=True, responsive=True, id='top-projects-table')
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

# Updated callback function to create compartmentalized statistics box content and map text overlay
@app.callback(
    [Output('county-map', 'figure'),
     Output('top-projects-table', 'children'),
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

    # Calculate unique counties analyzed and total possible counties based on selected regions
    total_new_jobs = filtered_df['New Jobs'].sum()
    total_capital_investment = filtered_df['Capital Investment'].sum()

    unique_counties_analyzed = filtered_df['County'].nunique()

    # Get the total applicable counties based on the selected regions
    if selected_regions:
        applicable_counties = get_unique_counties_by_region([f'{i}' for i in selected_regions])
        total_applicable_counties = len(applicable_counties)
    else:
        # If no regions are selected, use all counties across all regions
        applicable_counties = get_unique_counties_by_region(regions.columns)
        total_applicable_counties = len(applicable_counties)

    # Stats dictionary for the text overlay
    stats = {
        'counties': unique_counties_analyzed,
        'total_counties': total_applicable_counties,
        'projects': len(filtered_df),
        'new_jobs': f"{total_new_jobs:,}",
        'capital_investment': f"${total_capital_investment:,.0f}"
    }

    # Generate updated visuals
    county_map_fig = generate_county_map(county_summary_df, 'New Jobs', 'County-wise New Jobs', stats)
    top_projects_table = generate_top_projects_table(filtered_df)
    
    # Convert the dataframe into a table format
    top_projects_table_fig = dbc.Table.from_dataframe(top_projects_table, bordered=True, hover=True, responsive=True)

    new_jobs_plot_fig = generate_scatter_plot(filtered_df, 'New Jobs', 'New Jobs Over Time')
    investment_plot_fig = generate_scatter_plot(filtered_df, 'Capital Investment', 'Capital Investment Over Time')

    # Statistics box content (excluding the overlayed statistics)
    grant_type_counts = {grant: (filtered_df[grant] > 0).sum() if grant in filtered_df else 0 for grant in ['FJTAP', 'FIDP', 'ED']}
    region_order = [f'Region {i}' for i in range(1, 5)]
    region_counts = {region: filtered_df['Region'].value_counts().get(region, 0) for region in region_order}
    all_project_types = ['Recruitment', 'Expansion', 'Expansion New Location', 'New Startup']
    project_type_counts = {ptype: filtered_df['Project Type'].value_counts().get(ptype, 0) for ptype in all_project_types}

    statistics_content = dbc.Container(
        [
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

    return county_map_fig, top_projects_table_fig, new_jobs_plot_fig, investment_plot_fig, statistics_content

# Run the app
if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True)