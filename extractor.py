import os
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
from geoalchemy2 import Geometry
from dotenv import load_dotenv

class DataLoader:
    def __init__(self, base_path):
        load_dotenv()  # Load environment variables from .env file
        self.base_path = base_path
        self.engine = create_engine(os.getenv('DATABASE_URL'))

    def load_data(self):
        self.create_tables()
        self.load_analysis()
        self.load_statistics()
        self.load_predictions()
        self.load_original_parcels()
        self.load_overall_statistics()

    def process_gdf(self, gdf, table_name):
        # Ensure the geometry column is named 'geom'
        if 'geometry' in gdf.columns:
            gdf = gdf.rename(columns={'geometry': 'geom'})
        
        # If 'geom' is not in the columns, it might be because the geometry column has a different name
        if 'geom' not in gdf.columns:
            geom_column = gdf.geometry.name
            gdf = gdf.rename(columns={geom_column: 'geom'})
        
        # Set 'geom' as the active geometry column
        gdf = gdf.set_geometry('geom')
        
        # Ensure the CRS is set to EPSG:4326
        gdf = gdf.to_crs(epsg=4326)
        
        # Use the correct schema when writing to PostGIS
        gdf.to_postgis(table_name, self.engine, if_exists='replace', index=False, 
                       dtype={'geom': Geometry('POLYGON', srid=4326)})

    def load_analysis(self):
        path = os.path.join(self.base_path, 'evaluation', 'analysis.gpkg')
        gdf = gpd.read_file(path)
        self.process_gdf(gdf, 'analysis')

    def load_statistics(self):
        path = os.path.join(self.base_path, 'evaluation', 'statistics.csv')
        df = pd.read_csv(path)
        df.to_sql('statistics', self.engine, if_exists='replace', index=False)

    def load_predictions(self):
        path = os.path.join(self.base_path, 'predictions', 'prediction_combined.gpkg')
        gdf = gpd.read_file(path)
        self.process_gdf(gdf, 'predictions')

    def load_original_parcels(self):
        path = os.path.join(self.base_path, 'evaluation', 'all_original_parcels.gpkg')
        gdf = gpd.read_file(path)
        self.process_gdf(gdf, 'original_parcels')

    def load_overall_statistics(self):
        path = os.path.join(self.base_path, 'evaluation', 'overall_statistics.csv')
        df = pd.read_csv(path)
        df.columns = [
            'canton', 'area', 'overpredicted', 'low_recall', 
            'average_total_error', 'average_overprediction_error', 'average_recall_error'
        ]
        df.to_sql('overall_statistics', self.engine, if_exists='replace', index=False)

    def execute_sql_file(self, file_path):
        with open(file_path, 'r') as sql_file:
            sql = sql_file.read()
            with self.engine.connect() as conn:
                # Split the SQL into individual statements
                statements = sql.split(';')
                for statement in statements:
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()

    def drop_tables(self):
        tables = ['analysis', 'statistics', 'predictions', 'original_parcels', 'overall_statistics']
        with self.engine.connect() as conn:
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
            conn.commit()

    def create_tables(self):
        # Enable PostGIS extension
        with self.engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
            conn.commit()

        # Drop existing tables
        self.drop_tables()

        # Execute table creation SQL
        sql_file_path = os.path.join(os.path.dirname(__file__), "tables.sql")
        self.execute_sql_file(sql_file_path)

if __name__ == "__main__":
    base_path = r'C:\Users\dshus\Documents\Satelite\data\experiments\resunet_experiment_augmentation_False'
    loader = DataLoader(base_path)
    loader.create_tables()
    loader.load_data()
    print("Data loading completed.")