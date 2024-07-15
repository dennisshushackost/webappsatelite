-- For analysis.gpkg
CREATE TABLE analysis (
    fid SERIAL PRIMARY KEY,
    nutzung TEXT,
    area FLOAT,
    class_id INTEGER,
    canton TEXT,
    excerpt TEXT,
    true_positive FLOAT,
    false_negative FLOAT,
    recall FLOAT,
    low_recall BOOLEAN,
    overpredicted BOOLEAN,
    geom GEOMETRY(Polygon, 4326)
);

-- For statistics.csv
CREATE TABLE statistics (
    canton TEXT,
    excerpt TEXT,
    area FLOAT,
    overpredicted FLOAT,
    low_recall FLOAT,
    total_error FLOAT,
    overprediction_error FLOAT,
    recall_error FLOAT
);

-- For prediction_combined.gpkg
CREATE TABLE predictions (
    fid SERIAL PRIMARY KEY,
    file_name TEXT,
    area FLOAT,
    geom GEOMETRY(Polygon, 4326)
);

-- For all_original_parcels.gpkg
CREATE TABLE original_parcels (
    fid SERIAL PRIMARY KEY,
    nutzung TEXT,
    area FLOAT,
    class_id INTEGER,
    canton TEXT,
    excerpt TEXT,
    geom GEOMETRY(Polygon, 4326)
);

-- For overall_statistics.csv
CREATE TABLE overall_statistics (
    canton TEXT PRIMARY KEY,
    area FLOAT,
    overpredicted FLOAT,
    low_recall FLOAT,
    average_total_error FLOAT,
    average_overprediction_error FLOAT,
    average_recall_error FLOAT
);