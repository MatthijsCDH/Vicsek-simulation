from Vicsek import Viscek_Model, Animation, Obstacle, Circle, Rect
from List import *
import numpy as np
import matplotlib.pyplot as plt
etas = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
results = []
for eta in etas:
    obstacles = []
    model = Viscek_Model(100, System_size=(5,5), dt=0.1, 
                         length_time=1, R=0.25, Distribution=distributions["Uniform"],
                         eta = eta, v0=0.3, obstacles=obstacles)
    time = 100*10
    for _ in range(time):
        model.simulation_step()

    window = 100
    phi_ss = np.mean(model.phi_hist[-window:])
    chi_ss = np.var(model.phi_hist[-window:]) * model.N_particles
    results.append([phi_ss, chi_ss])
    print("Finished", eta)
results = np.array(results)
plt.plot(etas, results[:, 0])
plt.show()

plt.plot(etas, results[:, 1])
plt.show()