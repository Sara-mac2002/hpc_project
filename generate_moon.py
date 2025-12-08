from sklearn.datasets import make_moons
import numpy as np
import os

os.makedirs("data", exist_ok=True)
X, y = make_moons(100, noise=0.20, random_state=3)
np.savetxt("data/data_X.txt", X)
np.savetxt("data/data_y.txt", y, fmt="%d")

print("Data generated and saved to the 'data/' directory.")
