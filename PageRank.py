import numpy as np
import time
import sys
import re

def read_input(file_path):    
    adj_list = {}
    vertices = set()
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                # Checking for proper format
                if ':' not in line:
                    raise ValueError(f"Invalid line format: {line}") 
                
                parts = line.split(':')
                if len(parts) != 2:
                    raise ValueError(f"Invalid line format: {line}")
                
                # Validating that the vertex is a number
                vertex_str = parts[0].strip()
                if not re.fullmatch(r'\d+', vertex_str):
                    raise ValueError(f"Invalid vertex (not a number): {vertex_str}")
                
                vertex = int(vertex_str)
                
                # Validating edges (should be numbers only)
                edge_strs = [x.strip() for x in parts[1].split(',') if x.strip()]
                for edge in edge_strs:
                    if not re.fullmatch(r'\d+', edge):
                        raise ValueError(f"Invalid edge (not a number): {edge}")
                
                edges = list(map(int, edge_strs))
                
                # Removing duplicates
                adj_list[vertex] = list(set(edges))  
                
                # Adding to vertices set
                vertices.add(vertex)
                vertices.update(edges)
    
    except FileNotFoundError:
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    if not adj_list:
        raise ValueError("Input file is empty.")
    
    n = max(vertices) + 1
    return adj_list, n


def build_transition_matrix(adj_list, n):
    M = np.zeros((n, n))

    for i in range(n):
        if i in adj_list and adj_list[i]:
            out_edges = adj_list[i]
            for j in out_edges:
                M[j, i] = 1.0 / len(out_edges)
        else:
            M[:, i] = 1.0 / n  
    return M


def incorporate_teleportation(M, d, n):
    return d * M + (1 - d) / n * np.ones((n, n))


def power_iteration(A, n, metrics, threshold=1e-10, max_iter=1000):
    v = np.ones(n) / n
    start_time = time.time()

    for i in range(max_iter):
        v_new = np.dot(A, v) 
        # Normalize
        v_new = v_new / np.sum(v_new)  

        # Using L1 Norm (sum of absolute differences)
        diff = np.linalg.norm(v_new - v, ord=1)
        
        # Stop when the difference is below the threshold
        if diff < threshold:
            if metrics:
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Converged after {i+1} iterations!")
                print(f"\nExecution Time: {elapsed_time:.6f} seconds")
            
            # Return values in scientific notation with 10 decimal places
            return np.array([format(float(x), ".10e") for x in v_new], dtype=str)

        # Update v for next iteration
        v = v_new  

    return np.array([format(float(x), ".10e") for x in v], dtype=str)


def main(input_file, d, metrics=False):
    try:
        start_main_execution_time = time.time()
        adj_list, n = read_input(input_file)
        M = build_transition_matrix(adj_list, n)
        A = incorporate_teleportation(M, d, n)
        
        # If metrics are enabled, it will show the time taken for convergence
        ranks = power_iteration(A, n, metrics)
        
        # Convert ranks back to float for sorting and summation
        ranks_float = np.array(ranks, dtype=float)
        
        # Sorting indices based on the PageRank values
        sorted_index_array = np.argsort(ranks_float)
        top_n = min(10, len(ranks_float))

        top_indices = sorted_index_array[-top_n:][::-1]
        top_values = ranks_float[top_indices]

        final_result = np.round(np.sum(ranks_float), 10)

        # Print all PageRank values in scientific notation
        for rank in ranks_float:
            print(f"{rank:.10e}") 

        # If metrics are enabled, show top 10 values and final result
        if metrics:
            print("\nTop 10 PageRank Values with Indices:")
            for idx, value in zip(top_indices, top_values):
                print(f"Node {idx}: {value:.10e}")
            print(f"INFO:: final_result = {final_result:.10e}")
            end_main_execution_time = time.time()
            elapsed_main_execution_time = end_main_execution_time - start_main_execution_time
            print(f"\nTotal Execution Time: {elapsed_main_execution_time:.6f} seconds")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # Error handling for number of arguments being passed to the program from CLI
    if ((len(sys.argv) != 3) and (len(sys.argv) != 4)):
        print("ERROR: Incorrect number of parameters passed \
              \nExample Standard Usage: python input.py input.txt d > output.txt \
              \nExample Optional Usage(with parameters): python main.py input.txt 0.85 True > output.txt \
              \nEx: python main.py input.txt 0.85 > output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    d = float(sys.argv[2])
    
    # Handling boolean input from CLI properly
    metrics = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False
    
    main(input_file, d, metrics)
