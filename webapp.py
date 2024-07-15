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
from PIL import Image
import base64
import io

# Load environment variables
load_dotenv()

# Create database connection
@st.cache_resource
def get_connection():
    return create_engine(os.getenv('DATABASE_URL'))

engine = get_connection()

# Initialize session state
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
if 'show_satellite' not in st.session_state:
    st.session_state.show_satellite = False

st.title('Field Parcel Segmentation Analysis')

# Sidebar for filters
st.sidebar.header('Filters')

# Recall range slider
new_recall_range = st.sidebar.slider('Recall Range', 0.0, 1.0, 
                                     value=(st.session_state.recall_range[0], st.session_state.recall_range[1]), 
                                     step=0.1)

# Check if the recall range has changed
if new_recall_range != st.session_state.recall_range:
    st.session_state.recall_range = new_recall_range
    st.session_state.last_update_time = datetime.now()

# Check if any parcel is selected
any_parcel_selected = len(st.session_state.selected_auschnitte) > 0

# Display options, disabled until a parcel is selected, except for Show Recall Values
new_show_overpredictions = st.sidebar.checkbox('Show Overpredictions', False, 
                                               help="All Overpredictions (Predicted Parcels - Model Predictions), which are bigger than 5000m2)",
                                               disabled=not any_parcel_selected)
new_show_original_data = st.sidebar.checkbox('Show Original Data', False, 
                                             help="Original Data without removing 5000m2 parcels",
                                             disabled=not any_parcel_selected)
new_show_predictions = st.sidebar.checkbox('Show Predictions', False, 
                                           help="Predictions from the model",
                                           disabled=not any_parcel_selected)
new_show_recall_values = st.sidebar.checkbox('Show Recall Values', True, 
                                             help="Recall values of all parcels > 5000m2")
new_show_satellite = st.sidebar.checkbox('Show Satellite Imagery', False, 
                                         help="Show satellite imagery for selected parcels", 
                                         disabled=not any_parcel_selected)

# Load cantons
with engine.connect() as conn:
    cantons = ['CH'] + sorted(pd.read_sql("SELECT DISTINCT canton FROM analysis", conn)['canton'].tolist())

# Canton selection
new_selected_canton = st.selectbox('Select a Canton', cantons, index=cantons.index(st.session_state.selected_canton))

# Update session state
st.session_state.show_recall_values = new_show_recall_values  # Always update Show Recall Values

if any_parcel_selected:
    st.session_state.show_overpredictions = new_show_overpredictions
    st.session_state.show_predictions = new_show_predictions
    st.session_state.show_satellite = new_show_satellite
else:
    st.session_state.show_overpredictions = False
    st.session_state.show_predictions = False
    st.session_state.show_satellite = False

if new_selected_canton != st.session_state.selected_canton:
    st.session_state.selected_canton = new_selected_canton
    st.session_state.selected_auschnitte = []  # Reset selected Auschnitte when canton changes

# Initialize GeoDataFrames
filtered_gdf = gpd.GeoDataFrame()
filtered_predictions_gdf = gpd.GeoDataFrame()
filtered_original_data = gpd.GeoDataFrame()
overpredicted_gdf = gpd.GeoDataFrame()

# Filter data based on the selected canton and auschnitte
with engine.connect() as conn:
    if st.session_state.selected_canton == 'CH':
        filtered_stats_df = pd.read_sql("""
        SELECT * FROM statistics;
        """, conn)
    else:
        filtered_stats_df = pd.read_sql(f"""
        SELECT * FROM statistics 
        WHERE canton = '{st.session_state.selected_canton}'
        """, conn)

    if st.session_state.selected_auschnitte:
        excerpts = tuple(st.session_state.selected_auschnitte)
        if len(excerpts) == 1:
            excerpts = f"('{excerpts[0]}')"
        else:
            excerpts = str(excerpts)
        
        filtered_gdf = gpd.read_postgis(f"""
        SELECT * FROM analysis 
        WHERE excerpt IN {excerpts}
        """, conn, geom_col='geom')
        
        filtered_predictions_gdf = gpd.read_postgis(f"""
        SELECT * FROM predictions
        WHERE file_name IN {excerpts}
        """, conn, geom_col='geom')
        
        filtered_original_data = gpd.read_postgis(f"""
        SELECT * FROM original_parcels
        WHERE excerpt IN {excerpts}
        """, conn, geom_col='geom')
        
        overpredicted_gdf = gpd.read_postgis(f"""
        SELECT * FROM analysis
        WHERE excerpt IN {excerpts}
        AND (overpredicted = TRUE OR overpredicted IS NULL)
        """, conn, geom_col='geom')

# Apply filters only if the necessary columns exist
if not filtered_gdf.empty:
    if 'overpredicted' in filtered_gdf.columns:
        normal_gdf = filtered_gdf[filtered_gdf['overpredicted'] != True].copy()
    else:
        normal_gdf = filtered_gdf.copy()

    if 'recall' in normal_gdf.columns:
        normal_gdf = normal_gdf[(normal_gdf['recall'] >= st.session_state.recall_range[0]) & 
                                (normal_gdf['recall'] <= st.session_state.recall_range[1])]
else:
    normal_gdf = gpd.GeoDataFrame()

if not overpredicted_gdf.empty and 'recall' in overpredicted_gdf.columns:
    overpredicted_gdf = overpredicted_gdf[overpredicted_gdf['recall'].isnull() | 
                                          ((overpredicted_gdf['recall'] >= st.session_state.recall_range[0]) & 
                                           (overpredicted_gdf['recall'] <= st.session_state.recall_range[1]))]

def get_satellite_image(file_name):
    # Ensure file_name has .png extension
    if not file_name.lower().endswith('.png'):
        file_name += '.png'

    with engine.connect() as conn:
        image_data = pd.read_sql(f"SELECT * FROM image_data WHERE file_name = '{file_name}'", conn)
    
    if image_data.empty:
        st.warning(f"No satellite image found for parcel {file_name}")
        return None, None
    
    image_data = image_data.iloc[0]
    file_path = image_data['full_path']    
    try:
        img = None
        # First, try to open as a file
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                img = Image.open(f).copy()
        else:
            img_data = base64.b64decode(file_path)
            img = Image.open(io.BytesIO(img_data))
        
        if img is None:
            raise FileNotFoundError(f"Unable to locate image for parcel {file_name}")
        
        # Convert to RGB if it's not already
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Convert to base64 for Folium
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        # Use the bounds from the database
        bounds = (image_data['min_lon'], image_data['min_lat'], 
                  image_data['max_lon'], image_data['max_lat'])
        
        return img_str, bounds
    except Exception as e:
        st.error(f"Error processing satellite image for parcel {file_name}: {str(e)}")
        return None, None

# Create map function
def create_map():
    m = folium.Map(location=st.session_state.map_view['center'], zoom_start=st.session_state.map_view['zoom'], tiles='CartoDB positron')
    
    colormap = LinearColormap(colors=['purple', 'blue', 'green', 'yellow'], vmin=0, vmax=1)
    
    def style_function(feature):
        recall = feature['properties']['recall']
        if pd.isna(recall):
            return {'fillColor': 'red', 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}
        else:
            return {'fillColor': colormap(recall), 'color': 'black', 'weight': 1, 'fillOpacity': 0.7}

    def highlight_function(feature):
        return {'fillColor': '#000000', 'color': '#000000', 'fillOpacity': 0.50, 'weight': 0.1}

    if st.session_state.show_predictions and not filtered_predictions_gdf.empty:
        folium.GeoJson(
            filtered_predictions_gdf,
            style_function=lambda x: {'fillColor': 'orange', 'color': 'black', 'weight': 1, 'fillOpacity': 0.5},
        ).add_to(m)

    if new_show_original_data and not filtered_original_data.empty:
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

    if st.session_state.show_recall_values and not normal_gdf.empty:
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

    if st.session_state.show_overpredictions and not overpredicted_gdf.empty:
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

    if st.session_state.show_satellite and st.session_state.selected_auschnitte:
        for parcel in st.session_state.selected_auschnitte:
            img_str, bounds = get_satellite_image(parcel)
            if img_str is not None and bounds is not None:
                folium.raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{img_str}",
                    bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
                    opacity=1.0,
                ).add_to(m)

    colormap.add_to(m)
    colormap.caption = 'Recall Score'
    
    folium.LayerControl().add_to(m)
    
    return m

# Display the map
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