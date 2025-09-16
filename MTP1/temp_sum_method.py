import numpy as np
import matplotlib.pyplot as plt
from dtaidistance import dtw

# ===== 1. GENERATE SAMPLE DATA =====
n_sequences = 10
sequence_length = 50
t = np.linspace(0, 2 * np.pi, sequence_length)

# --- CHOOSE WHICH SCENARIO TO TEST ---
# To test the edge case, set test_scenario = 'all_identical'
test_scenario = 'single_outlier' 

if test_scenario == 'all_identical':
    sequences = np.array([np.sin(t)] * n_sequences)
    print("Generated a set of all identical sequences.")
else:
    # Initialize as an EMPTY LIST to allow appending
    sequences = [] 
    
    # Generate normal sequences
    for _ in range(n_sequences - 1):
        noise = np.random.normal(0, 0.1, sequence_length)
        sequences.append(np.sin(t) + noise)
        
    # Create and insert the outlier sequence
    outlier_sequence = np.cos(t)
    outlier_index_true = np.random.randint(0, n_sequences)
    sequences.insert(outlier_index_true, outlier_sequence)
    
    # Convert the completed list to a NumPy array AFTER it's been populated
    sequences = np.array(sequences) 
    print(f"Generated {n_sequences} sequences. The true outlier is at index {outlier_index_true}.")

# ===== 2. CHECK FOR "ALL IDENTICAL" EDGE CASE =====
# Calculate the total pairwise distance of the entire initial set.
initial_total_distance = 0
for i in range(n_sequences):
    for j in range(i + 1, n_sequences):
        initial_total_distance += dtw.distance(sequences[i], sequences[j])

# If the total distance is near zero, all sequences are the same.
if np.isclose(initial_total_distance, 0):
    print("\nEDGE CASE DETECTED: All sequences are identical. There is no outlier.")
    
    # Plot all sequences as normal and exit
    plt.figure(figsize=(12, 7))
    for seq in sequences:
        plt.plot(seq, color='blue', alpha=0.5)
    plt.title('Analysis Result: No Outlier Found', fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

else:
    # ===== 3. RUN THE LEAVE-ONE-OUT ALGORITHM =====
    dispersion_scores = []
    print("\nRunning 'Sum of Pairwise Distances' algorithm...")

    for i in range(n_sequences):
        # Create a temporary set of sequences with the i-th one removed
        temp_indices = [j for j in range(n_sequences) if j != i]
        temp_sequences = sequences[temp_indices]
        
        total_pairwise_distance = 0
        num_temp_seqs = len(temp_sequences)
        
        # Loop through every unique pair in the temporary group
        for j in range(num_temp_seqs):
            for k in range(j + 1, num_temp_seqs):
                distance = dtw.distance(temp_sequences[j], temp_sequences[k])
                total_pairwise_distance += distance**2
                
        dispersion_scores.append(total_pairwise_distance)

    # The outlier is the one whose removal caused the minimum dispersion
    outlier_index_found = np.argmin(dispersion_scores)
    print(f"Algorithm finished. The identified outlier is at index {outlier_index_found}.")

    # ===== 4. VISUALIZE THE RESULTS =====
    plt.figure(figsize=(12, 7))
    for i, seq in enumerate(sequences):
        if i == outlier_index_found:
            # Plot the identified outlier in a distinct color
            plt.plot(seq, color='red', linewidth=3, label=f'Identified Outlier (Index {i})')
        else:
            # Plot the normal sequences
            plt.plot(seq, color='blue', alpha=0.5)
            
    # Add a line for one of the normal sequences to create a legend entry
    plt.plot(sequences[0 if outlier_index_found != 0 else 1], color='blue', alpha=0.7, label='Normal Sequences')
    plt.title('Outlier Sequence Identification (Sum of Pairwise Distances)', fontsize=16)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()