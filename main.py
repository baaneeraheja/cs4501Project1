import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import hdbscan


# ---- STEP 1: Parse Location History ---- #
def parse_location_history(file_path, time_threshold=30):
    """
    Parses the JSON file and extracts 'visit' entries.
    Filters out stays shorter than the time_threshold (default: 30 minutes).
    """
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
                        lat, lon = float(lat_str), float(lon_str)
                    except ValueError:
                        continue
                else:
                    continue

                try:
                    start_time = datetime.fromisoformat(entry["startTime"].replace("Z", "+00:00"))
                    end_time = datetime.fromisoformat(entry["endTime"].replace("Z", "+00:00"))
                except ValueError:
                    continue

                duration = (end_time - start_time).total_seconds() / 60.0  # Convert seconds to minutes
                if duration >= time_threshold:
                    records.append({
                        'latitude': lat,
                        'longitude': lon,
                        'start_time': start_time,
                        'end_time': end_time,
                        'duration': duration
                    })

    return pd.DataFrame(records)


# ---- STEP 2: Cluster Locations Using HDBSCAN ---- #
def cluster_locations(df, min_cluster_size=3):
    """
    Cluster the significant locations using HDBSCAN with the haversine metric.
    """
    coords = df[['latitude', 'longitude']].to_numpy()
    coords_rad = np.radians(coords)  # Convert lat/lon to radians

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='haversine')
    df['cluster'] = clusterer.fit_predict(coords_rad)

    return df, clusterer


# ---- STEP 3: Compute Cluster Centers ---- #
def compute_cluster_centers(df):
    """
    Compute the mean latitude/longitude for each cluster (ignoring noise).
    """
    cluster_centers = {}
    for cluster in df['cluster'].unique():
        if cluster == -1:  # Ignore noise points
            continue
        cluster_df = df[df['cluster'] == cluster]
        cluster_centers[cluster] = (cluster_df['latitude'].mean(), cluster_df['longitude'].mean())

    return cluster_centers


# ---- STEP 4: Query API for Place Names (Placeholder) ---- #
def query_place_api(lat, lon):
    """
    Placeholder function to simulate querying an external API (Google Places, Foursquare).
    Replace with actual API calls.
    """
    return f"Place_{lat:.3f}_{lon:.3f}"


def label_clusters(cluster_centers):
    """
    Label clusters using the external API.
    """
    return {cluster: query_place_api(lat, lon) for cluster, (lat, lon) in cluster_centers.items()}


# ---- STEP 5: Save Results to CSV ---- #
def save_csv(cluster_centers, labels, output_file='significant_places.csv'):
    """
    Save cluster centers along with labels to a CSV file.
    """
    centers = [{'cluster': cluster, 'latitude': lat, 'longitude': lon, 'label': labels.get(cluster, 'Unknown')}
               for cluster, (lat, lon) in cluster_centers.items()]

    pd.DataFrame(centers).to_csv(output_file, index=False)
    print(f"Saved significant places to {output_file}")


# ---- STEP 6: Generate Static Map ---- #
def plot_static_map(cluster_centers, labels):
    """
    Generate a scatter plot of significant locations using matplotlib.
    """
    plt.figure(figsize=(10, 8))
    for cluster, (lat, lon) in cluster_centers.items():
        plt.scatter(lon, lat, label=f"Cluster {cluster}: {labels.get(cluster, 'Unknown')}", s=100)
        plt.text(lon, lat, labels.get(cluster, ''), fontsize=9, ha='right')

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Significant Locations")
    plt.legend()
    plt.savefig("static_map.png")
    plt.show()
    print("Static map saved as static_map.png")


# ---- STEP 7: Generate Interactive Map ---- #
def plot_interactive_map(cluster_centers, labels, df):
    """
    Generate an interactive map with folium.
    Includes cluster markers and a heatmap.
    """
    map_center = [df['latitude'].mean(), df['longitude'].mean()] if not df.empty else [0, 0]
    m = folium.Map(location=map_center, zoom_start=12)

    # Add cluster center markers
    for cluster, (lat, lon) in cluster_centers.items():
        folium.Marker(
            location=[lat, lon],
            popup=f"Cluster {cluster}: {labels.get(cluster, 'Unknown')}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(m)

    # Add heatmap
    HeatMap(df[['latitude', 'longitude']].values.tolist()).add_to(m)

    m.save("interactive_map.html")
    print("Interactive map saved as interactive_map.html")


# ---- STEP 8: Print Summary Statistics ---- #
def summarize_clusters(df):
    """
    Print the number of visits per detected cluster.
    """
    summary = df.groupby('cluster').size().reset_index(name='visit_count')
    print("Summary of detected clusters:")
    print(summary)
    return summary


# ---- MAIN EXECUTION ---- #
if __name__ == "__main__":
    # Adjustable parameters
    TIME_THRESHOLD = 30  # Minimum duration (minutes) to consider a visit significant
    MIN_CLUSTER_SIZE = 3  # Minimum number of points per cluster

    # Load and process data
    df = parse_location_history("location-history.json", time_threshold=TIME_THRESHOLD)
    if df.empty:
        print("No significant stays found.")
    else:
        print(f"Found {len(df)} significant stay points.")

        # Cluster locations
        df, clusterer = cluster_locations(df, min_cluster_size=MIN_CLUSTER_SIZE)

        # Compute cluster centers
        cluster_centers = compute_cluster_centers(df)

        # Label clusters
        cluster_labels = label_clusters(cluster_centers)

        # Save results
        save_csv(cluster_centers, cluster_labels)

        # Generate visualizations
        plot_static_map(cluster_centers, cluster_labels)
        plot_interactive_map(cluster_centers, cluster_labels, df)

        # Display summary
        summarize_clusters(df)
