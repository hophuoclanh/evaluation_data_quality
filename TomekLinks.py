import numpy as np

def tomek_links(X,y):
    tomek_links = []
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            if y[i] != y[j]:  # Check if they have different labels
                distance_ij = np.linalg.norm(X[i] - X[j])  # Euclidean distance between x[i] and x[j]
                is_tomek_link = True
                for k in range(len(X)):
                    if y[i] != y[k] and np.linalg.norm(X[i] - X[k]) < distance_ij:
                        is_tomek_link = False
                        break
                    if y[j] != y[k] and np.linalg.norm(X[j] - X[k]) < distance_ij:
                        is_tomek_link = False
                        break
                if is_tomek_link:
                    tomek_links.append((i, j))
    return tomek_links
