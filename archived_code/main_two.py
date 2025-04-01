import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import hdbscan

# Parse the Google Takeout location-history JSON file
def parse_location_history(file_path, time_threshold=30):
    """ Parse the JSON file and extract 'visit' entries with duration above the threshold. """
    with open(file_path, 'r') as f:
        data = json.load(f)

    records = []
    for entry in data:
        if "visit" in entry:
            visit = entry["visit"]
            if "topCandidate" in visit and "placeLocation" in visit["topCandidate"]:
                loc_str = visit["topCandidate"]["placeLocation"]
                if loc_str.startswith("geo:"):
                    try:
                        lat_str, lon_str = loc_str[4:].split(',')
                        lat = float(lat_str)
                        lon = float(lon_str)
                    except Exception:
                        continue
                else:
                    continue

                try:
                    start_time = datetime.fromisoformat(entry["startTime"].replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(entry["endTime"].replace("Z", "+00:00"))
                except Exception:
                    continue

                duration = (end_time - start_time).total_seconds() / 60.0  # minutes
                if duration >= time_threshold:
                    records.append({
                        'latitude': lat,
                        'longitude': lon,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })
    return pd.DataFrame(records)

# Cluster locations using HDBSCAN with haversine distance
def cluster_locations(df, min_cluster_size=3):
    """ Cluster the locations using HDBSCAN with the haversine metric. """
    coords = df[['latitude', 'longitude']].to_numpy()
    coords_rad = np.radians(coords)

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine')
    cluster_labels = clusterer.fit_predict(coords_rad)
    df['cluster'] = cluster_labels
    return df, clusterer

# Compute cluster summary (including visit count and average duration)
def compute_cluster_summary(df):
    """ Compute summary statistics for each cluster (excluding noise points with label -1). """
    summary = df[df['cluster'] != -1].groupby('cluster').agg(
        visit_count=('cluster', 'size'),
        average_duration=('duration', 'mean'),
        latitude=('latitude', 'mean'),
        longitude=('longitude', 'mean')
    ).reset_index()
    return summary

# Placeholder function to query an external API for place metadata
def query_place_api(lat, lon):
    """ Placeholder function for an API call to fetch place metadata. """
    return f"Place_{lat:.3f}_{lon:.3f}"

# Label each cluster using an external API (or a placeholder function)
def label_clusters(summary_df):
    """ Label the cluster centers using an API query. """
    labels = {}
    for _, row in summary_df.iterrows():
        cluster = row['cluster']
        lat = row['latitude']
        lon = row['longitude']
        labels[cluster] = query_place_api(lat, lon)
    return labels

# Save the cluster summary along with labels to a CSV file
def save_csv(summary_df, labels, output_file='significant_places.csv'):
    """ Save the cluster summary along with labels to a CSV file. """
    summary_df['label'] = summary_df['cluster'].map(labels)
    summary_df.to_csv(output_file, index=False)
    print(f"Saved significant places to {output_file}")

# Generate a static scatter plot map of significant locations using matplotlib
def plot_static_map(summary_df, labels):
    """ Generate a static scatter plot of significant locations using matplotlib. """
    plt.figure(figsize=(10, 8))
    for _, row in summary_df.iterrows():
        cluster = row['cluster']
        lat = row['latitude']
        lon = row['longitude']
        visits = row['visit_count']
        avg_duration = row['average_duration']
        label_text = labels.get(cluster, 'Unknown')
        plt.scatter(lon, lat, s=100, label=f"Cluster {cluster}: {label_text}\nVisits: {visits}, Avg: {avg_duration:.1f} min")
        plt.text(lon, lat, label_text, fontsize=9, ha='right')
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Significant Locations")
    plt.legend()
    plt.savefig("static_map.png")
    plt.show()
    print("Static map saved as static_map.png")

# Generate an interactive map with folium (including heatmap and cluster labels)
def plot_interactive_map(summary_df, labels, df):
    """ Generate an interactive map with folium. """
    if not df.empty:
        map_center = [df['latitude'].mean(), df['longitude'].mean()]
    else:
        map_center = [0, 0]

    m = folium.Map(location=map_center, zoom_start=12)

    # Add markers for cluster centers with detailed popups
    for _, row in summary_df.iterrows():
        cluster = row['cluster']
        lat = row['latitude']
        lon = row['longitude']
        visits = row['visit_count']
        avg_duration = row['average_duration']
        label_text = labels.get(cluster, 'Unknown')
        popup_text = f"Cluster {cluster}: {label_text}<br>Visits: {visits}<br>Avg Duration: {avg_duration:.1f} minutes"
        folium.Marker(
            location=[lat, lon],
            popup=popup_text,
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Add heatmap of all significant points
    heat_data = df[['latitude', 'longitude']].values.tolist()
    HeatMap(heat_data).add_to(m)

    m.save("interactive_map.html")
    print("Interactive map saved as interactive_map.html")

# Display a summary of the detected clusters
def summarize_clusters(df):
    """ Print a summary of detected clusters and visit counts. """
    summary = df.groupby('cluster').size().reset_index(name='visit_count')
    print("Summary of detected clusters:")
    print(summary)
    return summary

# Main script to process data and generate visualizations
if __name__ == "__main__":
    # Parameter tuning (adjust as needed)
    TIME_THRESHOLD = 30  # minutes
    MIN_CLUSTER_SIZE = 3

    # Parse the location history JSON file
    df = parse_location_history("location-history.json", time_threshold=TIME_THRESHOLD)
    if df.empty:
        print("No significant stays found based on the given time threshold.")
    else:
        print(f"Found {len(df)} significant stay points.")

    # Cluster the locations
    df, clusterer = cluster_locations(df, min_cluster_size=MIN_CLUSTER_SIZE)

    # Compute summary statistics for each cluster (ignoring noise points)
    summary_df = compute_cluster_summary(df)

    # Label each cluster
    cluster_labels = label_clusters(summary_df)

    # Save significant locations summary to CSV
    save_csv(summary_df, cluster_labels, output_file='run_one/significant_places.csv')

    # Generate static map visualization
    plot_static_map(summary_df, cluster_labels)

    # Generate interactive map with detailed info
    plot_interactive_map(summary_df, cluster_labels, df)

    # Display summary statistics of clusters
    summarize_clusters(df)
