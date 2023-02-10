### AI PROJECT
### TITLE : SOUND LOCALIZATION
### NAME : VENKATARAMAN R - CB.EN.P2AIE22011

import numpy as np
import heapq

# Load the audio signals and their corresponding labels
audio_signals = np.load('audio_signals.npy')
audio_labels = np.load('audio_labels.npy')

# Load the visual signals and their corresponding labels
visual_signals = np.load('visual_signals.npy')
visual_labels = np.load('visual_labels.npy')

# Function to calculate the Euclidean distance between two signals
def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Function to perform the A* algorithm for the CMR task
def a_star(start, end, audio_signals):
    frontier = [(0, start)]
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while frontier:
        current = heapq.heappop(frontier)[1]
        
        if current == end:
            break
            
        for i, audio_signal in enumerate(audio_signals):
            new_cost = cost_so_far[current] + euclidean_distance(audio_signals[current], audio_signal)
            if i not in cost_so_far or new_cost < cost_so_far[i]:
                cost_so_far[i] = new_cost
                priority = new_cost + euclidean_distance(audio_signal, audio_signals[end])
                heapq.heappush(frontier, (priority, i))
                came_from[i] = current
                
    return came_from, cost_so_far

# Train a nearest neighbor model on the audio signals to perform the IEr task
audio_nn = NearestNeighbors(n_neighbors=1)
audio_nn.fit(audio_signals)

# Find the nearest neighbors for each visual signal to perform the AVPE task
_, indices = audio_nn.kneighbors(visual_signals)

# Use A* algorithm to find the path from the nearest neighbor to the target audio signal for the CMR task
paths = []
for index in indices:
    start = index[0]
    end = audio_labels.index(visual_labels[index[0]])
    came_from, cost_so_far = a_star(start, end, audio_signals)
    path = [end]
    while end != start:
        end = came_from[end]
        path.append(end)
    path.reverse()
    paths.append(path)
    
# Plot the paths for the CMR task
for path in paths:
    plt.plot(audio_signals[path].T)
    plt.title('Path from Nearest Neighbor to Target Audio Signal')
    plt.show()
