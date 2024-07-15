import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from streamlit_folium import folium_static
from branca.colormap import LinearColormap
from datetime import datetime, timedelta
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create database connection
@st.cache_resource
def get_connection():
    return create_engine(os.getenv('DATABASE_URL'))

engine = get_connection()

# Initialize session state for map view and other variables (webapp)
if 'map_view' not in st.session_state:
    st.session_state.map_view = {
        'center': [47.3769, 8.5417],
        'zoom': 9
    }
if 'show_overpredictions' not in st.session_state:
    st.session_state.show_overpredictions = False
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False
if 'show_recall_values' not in st.session_state:
    st.session_state.show_recall_values = True
if 'recall_range' not in st.session_state:
    st.session_state.recall_range = (0.0, 1.0)
if 'selected_canton' not in st.session_state:
    st.session_state.selected_canton = 'CH'
if 'last_update_time' not in st.session_state:
    st.session_state.last_update_time = datetime.now()
if 'selected_auschnitte' not in st.session_state:
    st.session_state.selected_auschnitte = []

st.title('Field Parcel Segmentation Analysis')

# Sidebar for filters
st.sidebar.header('Filters')
new_recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, st.session_state.recall_range, 0.1)

# Check if the recall range has changed
if new_recall_range != st.session_state.recall_range:
    current_time = datetime.now()
    if current_time - st.session_state.last_update_time > timedelta(seconds=10):
        st.session_state.recall_range = new_recall_range
        st.session_state.last_update_time = current_time
        st.rerun()
    else:
        pass

new_show_overpredictions = st.sidebar.checkbox('Show Overpredictions', st.session_state.show_overpredictions, help="All Overpredictions (Predicted Parcels - Model Predictions), which are bigger than 5000m2)")
new_show_original_data = st.sidebar.checkbox('Show Original Data', False, help="Original Data without removing 5000m2 parcels")
new_show_predictions = st.sidebar.checkbox('Show Predictions', st.session_state.show_predictions, help="Predictions from the model")
new_show_recall_values = st.sidebar.checkbox('Show Recall Values', st.session_state.show_recall_values, help="Recall values of all parcels > 5000m2")

# Load cantons
with engine.connect() as conn:
    cantons = ['CH'] + sorted(pd.read_sql("SELECT DISTINCT canton FROM analysis", conn)['canton'].tolist())

# Canton selection
new_selected_canton = st.selectbox('Select a Canton', cantons, index=cantons.index(st.session_state.selected_canton))

# Update session state
st.session_state.show_overpredictions = new_show_overpredictions
st.session_state.show_predictions = new_show_predictions
st.session_state.show_recall_values = new_show_recall_values
if new_selected_canton != st.session_state.selected_canton:
    st.session_state.selected_canton = new_selected_canton
    st.session_state.selected_auschnitte = []  # Reset selected Auschnitte when canton changes

# Filter data based on the selected canton and auschnitte
with engine.connect() as conn:
    if st.session_state.selected_canton == 'CH':
        # Limit how many should show up when CH is selected LIMIT: Select the amount
        filtered_gdf = gpd.read_postgis("""
        WITH distinct_excerpts AS (
            SELECT DISTINCT excerpt
            FROM analysis
            LIMIT 2
        )
        SELECT t.*
        FROM analysis t
        JOIN distinct_excerpts de ON t.excerpt = de.excerpt
        WHERE t.nutzung IS NOT NULL
        AND t.area IS NOT NULL
        AND t.class_id IS NOT NULL
        AND t.canton IS NOT NULL
        AND t.excerpt IS NOT NULL
        AND t.geom IS NOT NULL;
        """, conn, geom_col='geom')
        excerpts = tuple(filtered_gdf['excerpt'].unique())
        if len(excerpts) == 1:
            excerpts = f"('{excerpts[0]}')"
        else:
            excerpts = str(excerpts)
        filtered_stats_df = pd.read_sql(f"""
        SELECT *
        FROM statistics
        WHERE excerpt IN {excerpts};
        """, conn)
        # Fetching filtered predictions
        filtered_predictions_gdf = gpd.read_postgis(f"""
            SELECT *
            FROM predictions
            WHERE file_name IN (SELECT DISTINCT excerpt FROM analysis WHERE excerpt IN {excerpts});
        """, conn, geom_col='geom')
        
        filtered_original_data = gpd.read_postgis(f"""
        SELECT *
        FROM original_parcels
        WHERE excerpt IN {excerpts};
        """, conn, geom_col='geom')
        overpredicted_gdf = gpd.read_postgis(f"""
        SELECT *
        FROM analysis
        WHERE excerpt IN {excerpts}
        AND nutzung IS NULL
        AND area IS NULL
        AND class_id IS NULL
        AND true_positive IS NULL
        AND false_negative IS NULL
        AND recall IS NULL;
    """, conn, geom_col='geom')
        # Show max 
        # filtered_stats_df = pd.read_sql("SELECT * FROM statistics", conn)
        #filtered_predictions_gdf = gpd.read_postgis("SELECT * FROM predictions", conn, geom_col='geom')
        #filtered_original_data = gpd.read_postgis("SELECT * FROM original_parcels ", conn, geom_col='geom')
    else:
        filtered_gdf = gpd.read_postgis(f"SELECT * FROM analysis WHERE canton = '{st.session_state.selected_canton}'", conn, geom_col='geom')
        filtered_stats_df = pd.read_sql(f"SELECT * FROM statistics WHERE canton = '{st.session_state.selected_canton}'", conn)
        filtered_predictions_gdf = gpd.read_postgis(f"SELECT * FROM predictions WHERE LEFT(predictions.file_name, 2) = '{st.session_state.selected_canton}'", conn, geom_col='geom')
        filtered_original_data = gpd.read_postgis(f"SELECT * FROM original_parcels WHERE canton = '{st.session_state.selected_canton}'", conn, geom_col='geom')

if st.session_state.selected_auschnitte:
    filtered_gdf = filtered_gdf[filtered_gdf['excerpt'].isin(st.session_state.selected_auschnitte)]
    filtered_predictions_gdf = filtered_predictions_gdf[filtered_predictions_gdf['file_name'].isin(st.session_state.selected_auschnitte)]
    filtered_original_data = filtered_original_data[filtered_original_data['excerpt'].isin(st.session_state.selected_auschnitte)]

# Explicitly separate overpredicted parcels
if not  st.session_state.selected_canton == 'CH':
    overpredicted_gdf = filtered_gdf[filtered_gdf['overpredicted'] == True].copy()
else:
    overpredicted_gdf = overpredicted_gdf
normal_gdf = filtered_gdf[filtered_gdf['overpredicted'] != True].copy()

# Apply recall filter to both normal and overpredicted parcels
normal_gdf = normal_gdf[(normal_gdf['recall'] >= st.session_state.recall_range[0]) & (normal_gdf['recall'] <= st.session_state.recall_range[1])]
overpredicted_gdf = overpredicted_gdf[overpredicted_gdf['recall'].isnull() | ((overpredicted_gdf['recall'] >= st.session_state.recall_range[0]) & (overpredicted_gdf['recall'] <= st.session_state.recall_range[1]))]

# Create map function
# Create map function
def create_map():
    m = folium.Map(location=st.session_state.map_view['center'], zoom_start=st.session_state.map_view['zoom'], tiles='CartoDB positron')
    
    # Create a color map
    colormap = LinearColormap(colors=['purple', 'blue', 'green', 'yellow'], vmin=0, vmax=1)
    
    def style_function(feature):
        recall = feature['properties']['recall']
        if pd.isna(recall):
            return {'fillColor': 'red', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        else:
            return {'fillColor': colormap(recall), 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}

    def highlight_function(feature):
        return {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    # Add predictions if checkbox is checked
    if st.session_state.show_predictions:
        folium.GeoJson(
            filtered_predictions_gdf,
            style_function=lambda x: {'fillColor': 'orange', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5},
        ).add_to(m)

    # Add original data if checkbox is checked
    if new_show_original_data:
        folium.GeoJson(
            filtered_original_data,
            style_function=lambda x: {'fillColor': 'gray', 'color': 'black', 'weight': 1, 'fillOpacity': 0.3},
            tooltip=folium.GeoJsonTooltip(
                fields=['nutzung', 'area', 'canton'],
                aliases=['Usage', 'Area', 'Canton'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add normal parcels if show_recall_values is checked
    if st.session_state.show_recall_values:
        folium.GeoJson(
            normal_gdf,
            style_function=style_function,
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['nutzung', 'area', 'canton', 'recall', 'overpredicted', 'excerpt'],
                aliases=['Usage', 'Area', 'Canton', 'Recall', 'Overpredicted', 'Excerpt'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add overpredicted parcels if checkbox is checked
    if st.session_state.show_overpredictions:
        folium.GeoJson(
            overpredicted_gdf,
            style_function=lambda x: {'fillColor': 'red', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7},
            highlight_function=highlight_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['canton', 'overpredicted', 'excerpt'],
                aliases=['Canton', 'Overpredicted', 'excerpt'],
                localize=True,
                sticky=False,
                labels=True,
                style="""
                    background-color: #F0EFEF;
                    border: 2px solid black;
                    border-radius: 3px;
                    box-shadow: 3px;
                """,
                max_width=800,
            ),
        ).add_to(m)

    # Add color legend
    colormap.add_to(m)
    colormap.caption = 'Recall Score'
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    return m

m = create_map()
folium_static(m, width=700, height=500)

# Display overall statistics
st.header('Overall Statistics')

# Filter overall statistics based on selected canton
with engine.connect() as conn:
    if st.session_state.selected_canton == 'CH':
        filtered_overall_stats = pd.read_sql("SELECT * FROM overall_statistics WHERE canton = 'ch'", conn)
    else:
        filtered_overall_stats = pd.read_sql(f"SELECT * FROM overall_statistics WHERE canton = '{st.session_state.selected_canton}'", conn)

# Calculate average recall
with engine.connect() as conn:
    if st.session_state.selected_canton == 'CH':
        avg_recall = pd.read_sql("SELECT AVG(recall) as avg_recall FROM analysis", conn)['avg_recall'].iloc[0]
    else:
        avg_recall = pd.read_sql(f"SELECT AVG(recall) as avg_recall FROM analysis WHERE canton = '{st.session_state.selected_canton}'", conn)['avg_recall'].iloc[0]

for _, row in filtered_overall_stats.iterrows():
    st.subheader(f"Statistics for {row['canton']}")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Area of Parcels (m²)", f"{row['area']:,.2f}")
        st.metric("Overpredicted Area (m²)", f"{row['overpredicted']:,.2f}")
        st.metric("Low Recall Area (m²)", f"{row['low_recall']:,.2f}", 
                  help="The area of parcels with a Recall < 0.7")
        st.metric("Average Recall", f"{avg_recall:.4f}", 
                  help="Average Recall value over all parcels")
    with col2:
        st.metric("Average Overprediction Error", f"{row['average_overprediction_error']:.4f}")
        st.metric("Average Recall Error", f"{row['average_recall_error']:.4f}")
        st.metric("Average Total Error", f"{row['average_total_error']:.4f}")

# Use smaller text for statistics
st.markdown("""
<style>
    .stMetric {
        font-size: 0.8rem;
    }
    .stMetric .st-emotion-cache-16v4zaw {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

# Display statistics table
st.header('Statistics by Parcel')

# Rename 'auschnitt' to 'Parcel' in filtered_stats_df
filtered_stats_df = filtered_stats_df.rename(columns={'excerpt': 'Parcel'})

# Sort the filtered_stats_df by Total Error in descending order
filtered_stats_df = filtered_stats_df.sort_values('total_error', ascending=False)

# Add a 'Select' column to the dataframe
filtered_stats_df['Select'] = filtered_stats_df['Parcel'].isin(st.session_state.selected_auschnitte)

# Reorder columns to put Select and Parcel at the beginning
cols = ['Select', 'Parcel', 'canton'] + [col for col in filtered_stats_df.columns if col not in ['Select', 'Parcel', 'canton']]
filtered_stats_df = filtered_stats_df[cols]

# Create a styled dataframe
styled_df = filtered_stats_df.style.format({
    'area': '{:.2f}',
    'overpredicted': '{:.2f}',
    'low_recall': '{:.2f}',
    'total_error': '{:.2f}',
    'overprediction_error': '{:.2f}',
    'recall_error': '{:.2f}'
})

# Apply background color to highlight selected rows
def highlight_selected(s):
    return ['background-color: #ADD8E6' if s.Select else '' for _ in s]

styled_df = styled_df.apply(highlight_selected, axis=1)

# Unselect All button
if st.session_state.selected_auschnitte:
    if st.button("Unselect All"):
        st.session_state.selected_auschnitte = []
        st.rerun()

# Display the dataframe
# Display the dataframe
edited_df = st.data_editor(
    styled_df,
    hide_index=True,
    column_config={
        "Select": st.column_config.CheckboxColumn(
            "Select",
            help="Select Parcel",
            default=False,
        )
    },
    disabled=["Parcel", "canton", "area", "overpredicted", "low_recall", "total_error", "overprediction_error", "recall_error"],
    key="edited_df"
)

# Update selected_auschnitte based on the checkboxes
st.session_state.selected_auschnitte = edited_df[edited_df['Select'] == True]['Parcel'].tolist()

# Display current filter state
st.write(f"Selected Canton: {st.session_state.selected_canton}")
st.write(f"Selected Parcels: {', '.join(st.session_state.selected_auschnitte) if st.session_state.selected_auschnitte else 'None'}")

# Rerun the app if selections have changed
if edited_df['Select'].tolist() != filtered_stats_df['Select'].tolist():
    st.rerun()