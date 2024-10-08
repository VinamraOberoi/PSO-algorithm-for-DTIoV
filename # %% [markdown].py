# %% [markdown]
# <a href="https://colab.research.google.com/github/VinamraOberoi/PSO-algorithm-for-DTIoV/blob/main/PSO_algorithm_v2.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %%
import geopandas as gpd
import random
from shapely.geometry import Point
import folium
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from shapely.geometry import Polygon

# %%
# Load the GeoJSON file for Delhi (replace 'delhi_boundary.geojson' with your actual GeoJSON file path)
geojson_file = 'delhi_boundary.geojson'

# %%
# Load Delhi boundaries using GeoPandas
delhi_gdf = gpd.read_file(geojson_file)

# %%
# Extract the polygon boundary of Delhi
delhi_polygon = delhi_gdf['geometry'].values[0]

# %%
# Function to generate random points within the Delhi polygon
def generate_random_points_within_polygon(polygon, num_points):
    points = []
    minx, miny, maxx, maxy = polygon.bounds  # Bounding box of the polygon
    while len(points) < num_points:
        random_point = Point(random.uniform(minx, maxx), random.uniform(miny, maxy))
        if polygon.contains(random_point):  # Ensure the point is within the polygon
            points.append([random_point.y, random_point.x])  # Latitude, Longitude
    return points

# %%
# Generate the datasets
num_vehicles = 1000
num_rsus = 100
num_mbs = 20

delhi_streets_grid = {
    # Generate 1000 Vehicle Locations within Delhi
    "Vehicle_Locations": generate_random_points_within_polygon(delhi_polygon, num_vehicles),

    # Generate 100 RSU Locations within Delhi
    "RSU_Locations": generate_random_points_within_polygon(delhi_polygon, num_rsus),

    # Generate 20 MBS Locations within Delhi
    "MBS_Locations": generate_random_points_within_polygon(delhi_polygon, num_mbs),

    # Cloud Location fixed in Connaught Place, Delhi
    "Cloud_Location": [28.6139, 77.2090]  # Cloud server at Connaught Place, central Delhi
}

# %%
# Particle Swarm Optimization (PSO) Parameters
num_iterations = 1000  # Number of iterations for each swarm
num_particles = 100  # Swarm size
w = 0.5  # Inertia weight
c1 = 1.5  # Cognitive parameter (particle's own best)
c2 = 1.5  # Social parameter (swarm's global best)

# %%
# Task sizes and latencies (random values for simplicity)
num_tasks = len(delhi_streets_grid['Vehicle_Locations'])
task_sizes = np.random.uniform(0.5, 5, num_tasks)  # Task size in MB
task_latencies = np.random.uniform(10, 100, num_tasks)  # Task latency in ms

# %%
# Calculate distance between two geographical points
def calculate_distance(coord1, coord2):
    return np.sqrt((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2)

# %%
# Functions to compute latency based on allocation and vehicle distance from RSUs, MBS, and Cloud
def compute_latency(position):
    total_latency = 0
    for i, allocation in enumerate(position):
        vehicle_location = delhi_streets_grid["Vehicle_Locations"][i]
        if allocation == 0:  # Local processing
            total_latency += 10  # Local processing latency
        elif allocation == 1:  # RSU processing
            rsu_location = random.choice(delhi_streets_grid["RSU_Locations"])
            distance = calculate_distance(vehicle_location, rsu_location)
            total_latency += 30 + distance  # RSU latency + distance
        elif allocation == 2:  # MBS processing
            mbs_location = random.choice(delhi_streets_grid["MBS_Locations"])
            distance = calculate_distance(vehicle_location, mbs_location)
            total_latency += 50 + distance  # MBS latency + distance
        elif allocation == 3:  # Cloud processing
            cloud_location = delhi_streets_grid["Cloud_Location"]
            distance = calculate_distance(vehicle_location, cloud_location)
            total_latency += 100 + distance  # Cloud latency + distance
    return total_latency

# %%
# Function to compute system throughput based on task allocation
def compute_throughput(position):
    total_throughput = 0
    for i, allocation in enumerate(position):
        task_size = task_sizes[i]
        if allocation == 0:  # Local processing
            total_throughput += task_size / 10
        elif allocation == 1:  # RSU processing
            total_throughput += task_size / 30
        elif allocation == 2:  # MBS processing
            total_throughput += task_size / 50
        elif allocation == 3:  # Cloud processing
            total_throughput += task_size / 100
    return total_throughput

# %%
# Fitness function combining latency and throughput with weights w1 and w2 (w1 + w2 = 1)
def fitness_function(position, w1=0.5, w2=0.5):
    latency = compute_latency(position)
    throughput = compute_throughput(position)
    return w1 * latency + w2 * (1 / throughput)

# %%
# Create Folium map centered in Delhi
m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

# %%
# Add RSU markers to the map
for location in delhi_streets_grid["RSU_Locations"]:
    folium.Marker(location=location, popup="RSU", icon=folium.Icon(color="green")).add_to(m)

# %%
# Add MBS markers to the map
for location in delhi_streets_grid["MBS_Locations"]:
    folium.Marker(location=location, popup="MBS", icon=folium.Icon(color="orange")).add_to(m)

# %%
# Add Cloud marker
folium.Marker(location=delhi_streets_grid["Cloud_Location"], popup="Cloud", icon=folium.Icon(color="red")).add_to(m)

# %%
# Function to run PSO and visualize results (latency, throughput, fitness)
def run_pso_and_report():
    # Load previous best from file if it exists
    try:
        with open('pso_best.npy', 'rb') as f:
            global_best_position = np.load(f)
            global_best_fitness = np.load(f)
    except FileNotFoundError:
        global_best_position = None
        global_best_fitness = np.inf

    particles_position = np.random.randint(0, 4, size=(num_particles, num_tasks))
    particles_velocity = np.random.uniform(-1, 1, (num_particles, num_tasks))

    personal_best_position = np.copy(particles_position)
    personal_best_fitness = np.inf * np.ones(num_particles)

    if global_best_position is None:
        global_best_position = np.copy(particles_position[0])
        global_best_fitness = fitness_function(global_best_position)

    for iteration in range(num_iterations):
        for i in range(num_particles):
            current_fitness = fitness_function(particles_position[i])

            if current_fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = current_fitness
                personal_best_position[i] = np.copy(particles_position[i])

            if current_fitness < global_best_fitness:
                global_best_fitness = current_fitness
                global_best_position = np.copy(particles_position[i])

        for i in range(num_particles):
            r1 = np.random.rand(num_tasks)
            r2 = np.random.rand(num_tasks)
            particles_velocity[i] = (w * particles_velocity[i] +
                                     c1 * r1 * (personal_best_position[i] - particles_position[i]) +
                                     c2 * r2 * (global_best_position - particles_position[i]))

            particles_position[i] = particles_position[i].astype(float) + particles_velocity[i]
            particles_position[i] = np.clip(np.round(particles_position[i]), 0, 3).astype(int)

    final_latency = compute_latency(global_best_position)
    final_throughput = compute_throughput(global_best_position)
    final_fitness = fitness_function(global_best_position, w1=0.7, w2=0.3)

    print(f"Final Latency: {final_latency}")
    print(f"Final Throughput: {final_throughput}")
    print(f"Final Fitness (w1=0.7, w2=0.3): {final_fitness}")

    m.save('delhi_pso_visualization.html')

    # Save best to file
    with open('pso_best.npy', 'wb') as f:
        np.save(f, global_best_position)
        np.save(f, global_best_fitness)

# %%
# Run the PSO simulation and report latency, throughput, and fitness
run_pso_and_report()


