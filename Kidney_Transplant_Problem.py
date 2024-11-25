import json
import pulp
from collections import defaultdict

# Define the path to the JSON file
file_path = r"C:\Users\rguti\OneDrive\Desktop\PhD\1) IE 5318 Intro to Operations Research\Project\GutierrezATX.json"

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Define the swapped compatibility matrix explicitly
compatibility_matrix = {
    "A": {"A": 1, "B": 0, "AB": 0, "O": 1},
    "B": {"A": 0, "B": 1, "AB": 0, "O": 1},
    "AB": {"A": 1, "B": 1, "AB": 1, "O": 1},
    "O": {"A": 0, "B": 0, "AB": 0, "O": 1}
}

# Generate compatibility pairs from the matrix
C = [
    (donor, recipient)
    for recipient, donors in compatibility_matrix.items()
    for donor, compatible in donors.items()
    if compatible == 1
]

# Extract nodes, donors, and recipients from the data
A = list(range(len(data)))  # Assuming each entry in the data list is a node
D = {i: data[i]["Donor"] for i in A}
R = {i: [data[i]["Recipient"]] for i in A}

# Count the number of recipients by blood type
recipient_stats = defaultdict(int)
for i in A:
    recipient_stats[R[i][0]] += 1

print("Recipient Statistics:")
for blood_type, count in recipient_stats.items():
    print(f"Blood type {blood_type}: {count}")


# Print the number of nodes for troubleshooting
print(f"Number of nodes in A: {len(A)}")

# Initialize the problem
problem = pulp.LpProblem("Kidney_Transplant_Optimization", pulp.LpMaximize)

# Define the decision variables
X = {}
for i in A:
    for donor in D[i]:
        for j in A:
            for recipient in R[j]:
                X[(i, donor, j, recipient)] = pulp.LpVariable(f"X_{i}_{donor}_{j}_{recipient}", cat='Binary')

Z = {k: pulp.LpVariable(f"Z_{k}", cat='Binary') for k in A}

# Objective Function: Maximize the number of successful transplants
problem += pulp.lpSum([X[(i, donor, j, recipient)] for i in A for donor in D[i] for j in A for recipient in R[j]])

# Constraints
# Only one donor per node can donate
for i in A:
    problem += pulp.lpSum([X[(i, donor, j, recipient)] for donor in D[i] for j in A for recipient in R[j]]) <= 1

# Each donor can donate to at most one recipient
for i in A:
    for donor in D[i]:
        problem += pulp.lpSum([X[(i, donor, j, recipient)] for j in A for recipient in R[j]]) <= 1

# Each recipient can receive at most one donation
for j in A:
    for recipient in R[j]:
        problem += pulp.lpSum([X[(i, donor, j, recipient)] for i in A for donor in D[i]]) <= 1

# If the recipient is incompatible with the donor, then the donation cannot happen
for i in A:
    for donor in D[i]:
        for j in A:
            for recipient in R[j]:
                if (donor, recipient) not in C:
                    problem += X[(i, donor, j, recipient)] == 0

# Max number of chain lengths is 3 (number of nodes must be less than 3)
problem += pulp.lpSum([Z[k] for k in A]) <= 3

# Compatibility Constraint
for (i, donor, j, recipient) in X:
    if (donor, recipient) not in C:
        problem += X[(i, donor, j, recipient)] == 0

# Solve the problem
problem.solve()

# Initialize dictionaries to store chains of length 2 and 3
chains_length_2 = defaultdict(list)
chains_length_3 = defaultdict(list)

# Output the results
for variable in problem.variables():
    if variable.varValue > 0:
        print(f"{variable.name} = {variable.varValue}")

print(f"Objective Value: {pulp.value(problem.objective)}")