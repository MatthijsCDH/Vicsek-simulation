<<<<<<< HEAD
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.animation import FuncAnimation
from random import choice
from List import DistributionType, distributions
import argparse

"""
Vicsek model simulation with obstacles.

This module implements a 2D Vicsek model for self-propelled particles,
including the interaction between hard circular and rectangular obstacles.

It supports real-time animation and measurement of collective observables:
order parameter, correlation time, correlation length, and susceptibility.
"""

class Viscek_Model:
    """
    Vicsek model for self-propelled particles in 2D

    Parameters:

    N_particles : int
        Number of particles.
    System_size : tuple
        Width and height of the simulation box.
    dt : float
        Time step.
    R : float
        Particle's interaction radius
    Distribution : DistributionType
        Distribution for angular noise
    eta : float
        Noise strength
    v0 : float
        Initial particle speed for all particles
    obstacles : list of optional obstacles from Obstacle
        List of optional obstacles (circle or rectangle)
    """
    
    def __init__(self, N_particles, System_size, dt, R, Distribution: DistributionType  ,eta, v0, obstacles = None):
        self.N_particles = N_particles
        self.System_size = System_size
        self.dt = dt
        self.R = R
        self.eta = eta
        self.v0 = v0
        self.Distribution = Distribution
        self.obstacles = obstacles if obstacles is not None else []
        self.t = 0

        self.t_hist   = []
        self.phi_hist = []
        self.Ct_hist  = []
        self.Cr_hist  = []
        self.chi_hist = []
        self.x = np.zeros(N_particles)
        self.y = np.zeros(N_particles)

        # Initialize particle positions avoiding obstacles
        for i in range(self.N_particles):
            while True:
                x_attempt = np.random.rand() * System_size[0]
                y_attempt = np.random.rand() * System_size[1]
                if not any(obs.inside(x_attempt,y_attempt) for obs in self.obstacles):
                    self.x[i] = x_attempt
                    self.y[i] = y_attempt
                    break
        self.theta_initial = np.random.rand(N_particles)*2*np.pi
        self.theta = self.theta_initial.copy()
        self.theta_ref = self.theta.copy()

    def simulation_step(self):
        updated_theta = np.zeros(self.N_particles)
        correlation_length_i = np.zeros(self.N_particles)
        for i in range(self.N_particles):
            mean_theta, neighbors =self.compute_mean_theta(i)
            noise = self.compute_noise()
            correlation_length_i[i] = self.compute_local_correlation(i, neighbors)

            theta_prop = mean_theta + noise
            px, py = self.new_position(i, theta_prop)

            # Collision check with obstacles
            collided, theta_final = self.handle_collisions(i, px, py, theta_prop)
            if not collided:
                self.x[i] = px % self.System_size[0]
                self.y[i] = py % self.System_size[1]
                theta_final = theta_prop

            updated_theta[i] = theta_final


        self.theta = updated_theta
        self.compute_observables(correlation_length_i)

    def compute_mean_theta(self, i):
        dx = self.x - self.x[i]
        dy = self.y - self.y[i]
        dx -= self.System_size[0] * np.rint(dx / self.System_size[0])
        dy -= self.System_size[1] * np.rint(dy / self.System_size[1])
        r = np.sqrt(dx**2 + dy**2)
        neighbors = (r < self.R) & (r > 0)
        sin_sum = np.sum(np.sin(self.theta[neighbors]))
        cos_sum = np.sum(np.cos(self.theta[neighbors]))
        if np.sum(neighbors) > 0:
            mean_theta = np.arctan2(sin_sum, cos_sum)
        else:
            mean_theta = self.theta[i]
        return mean_theta, neighbors
    
    def compute_noise(self):
        return self.eta * (self.Distribution.sample() - 0.5) * np.pi
    
    def compute_local_correlation(self, i , neighbors):
        if np.sum(neighbors) > 0:
            return np.mean(np.cos(self.theta[i] - self.theta[neighbors]))
        return 0
    
    def new_position(self, i, theta):
        px = self.x[i] + self.v0 * np.cos(theta) * self.dt
        py = self.y[i] + self.v0 * np.sin(theta) * self.dt
        return px, py
    
    def handle_collisions(self, i, px, py, theta_prop):
        for obs in self.obstacles:
            if obs.inside(px, py):
                theta_new = self.resolve_collision(i, px, py, theta_prop, obs)
                return True, theta_new
        return False, theta_prop

    def resolve_collision(self, i, px, py, theta_prop, obs):
        x0, y0 = self.x[i], self.y[i]
        vx, vy = px - x0, py - y0
        t_low, t_high = 0.0, 1.0

        for _ in range(8):
            t_mid = 0.5 * (t_low + t_high)
            xm, ym = x0 + vx * t_mid, y0 + vy * t_mid
            if obs.inside(xm, ym):
                t_high = t_mid
            else:
                t_low = t_mid

        xc, yc = x0 + vx * t_low, y0 + vy * t_low

        # Normal
        nx, ny = obs.normal(xc, yc)
        n_norm = np.hypot(nx, ny) + 1e-6
        nx, ny = nx / n_norm, ny / n_norm

        # Reflect velocity
        v_in = np.array([vx, vy])
        v_out = v_in - 2 * np.dot(v_in, [nx, ny]) * np.array([nx, ny])
        theta_reflected = np.arctan2(v_out[1], v_out[0])

        # Move particle slightly away
        tx, ty = -ny, nx
        eps = self.v0 * self.dt + 1e-3

        for alpha in (0.0, 0.5, -0.5, 1.0, -1.0):
            x_try = xc + eps * (nx + alpha * tx)
            y_try = yc + eps * (ny + alpha * ty)
            if not obs.inside(x_try, y_try):
                self.x[i] = x_try % self.System_size[0]
                self.y[i] = y_try % self.System_size[1]
                return theta_reflected

        self.x[i] = (xc + nx * eps) % self.System_size[0]
        self.y[i] = (yc + ny * eps) % self.System_size[1]
        return theta_reflected

    def compute_observables(self, correlation_length_i):
        vx_sum = np.sum(self.v0 * np.cos(self.theta))
        vy_sum = np.sum(self.v0 * np.sin(self.theta))
        phi = np.sqrt(vx_sum**2 + vy_sum**2) / (self.N_particles * self.v0)
        # Calculate correction time
        correlation_time = np.mean(np.cos(self.theta - self.theta_ref))
        # Calculate correction length
        correlation_length = np.mean(correlation_length_i)
        # Calculate susceptibility
        if len(self.phi_hist) > 1:
            chi_t = (np.var(self.phi_hist))
        else:
            chi_t = 0

        self.t += self.dt

        self.t_hist.append(self.t)
        self.phi_hist.append(phi)
        self.Ct_hist.append(correlation_time)
        self.Cr_hist.append(correlation_length)
        self.chi_hist.append(chi_t)
        self.theta_ref = self.theta.copy()


class Animation:
    """
    Animation for the Vicsek model.

    Parameters:

    model : ViscekModel
        The Viscek model for animation
    """
    def __init__(self, model):
        self.model = model
        self.quiver_toggle = "on"
        self.step = 0
        self.anim = None

    def animate(self, interval=30, steps=100, quiver_toggle= "on"):
        self.quiver_toggle = quiver_toggle
        self.max_steps = steps*10
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(
            nrows=4,
            ncols=2,
            width_ratios=[3, 1], 
            hspace=0.35,
            wspace=0.25
        )
        ax_sim = fig.add_subplot(gs[:, 0])
        ax_ct  = fig.add_subplot(gs[0, 1])
        ax_cr  = fig.add_subplot(gs[1, 1])
        ax_chi = fig.add_subplot(gs[2, 1])
        ax_phi = fig.add_subplot(gs[3, 1])

        ax_sim.set_xlim(0, self.model.System_size[0])
        ax_sim.set_ylim(0, self.model.System_size[1])
        ax_sim.set_aspect('equal')

        for obs in self.model.obstacles:
            obs.plot(ax_sim)

        self.scatter = ax_sim.scatter(self.model.x, self.model.y, s=10, c='blue')
        if self.quiver_toggle == "on":
            self.quiver = ax_sim.quiver(self.model.x, self.model.y,
                                    np.cos(self.model.theta), np.sin(self.model.theta),
                                    color='red', scale=20)
        ax_sim.set_title('Vicsek Model Simulation')
        ax_phi.set_xlim(0, steps)
        ax_phi.set_ylim(-0, 1)
        ax_phi.set_xlabel('Time Steps')
        ax_phi.set_ylabel('Order Parameter φ')

        ax_ct.set_xlim(0, steps)
        ax_ct.set_ylim(-1, 1)
        ax_ct.set_ylabel('Correlation Time Ct')

        ax_cr.set_xlim(0, steps)
        ax_cr.set_ylim(-1, 1)
        ax_cr.set_ylabel('Correlation Length Cr')

        ax_chi.set_xlim(0, steps)
        ax_chi.set_ylim(0, 0.1)
        ax_chi.set_ylabel('Susceptibility χ')

        self.line_phi, = ax_phi.plot([], [], lw=2)
        self.line_Ct,  = ax_ct.plot([], [], lw=2)
        self.line_Cr,  = ax_cr.plot([], [], lw=2)
        self.line_chi, = ax_chi.plot([], [], lw=2)

        self.anim = FuncAnimation(fig, self.update, frames=steps, interval=interval, blit=True)
        return fig, self.anim

    def update(self, frame):
        self.model.simulation_step()
        self.scatter.set_offsets(np.c_[self.model.x, self.model.y])
        t = self.model.t_hist
        self.step += 1
        if self.step >= self.max_steps - 1:
            self.anim.event_source.stop()

        self.line_phi.set_data(t, self.model.phi_hist)
        self.line_Ct.set_data(t, self.model.Ct_hist)
        self.line_Cr.set_data(t, self.model.Cr_hist)
        self.line_chi.set_data(t, self.model.chi_hist)

        if self.quiver_toggle == "on":
            self.quiver.set_UVC(np.cos(self.model.theta), np.sin(self.model.theta))
            self.quiver.set_offsets(np.c_[self.model.x, self.model.y])
            return (
                self.scatter,
                self.quiver,
                self.line_phi,
                self.line_Ct,
                self.line_Cr,
                self.line_chi,
            )
        return (
                self.scatter,
                self.line_phi,
                self.line_Ct,
                self.line_Cr,
                self.line_chi,
            )
    
class Obstacle:
    """
    Base class for obstacles in a Vicsek simulation.
    All obstacles must implement these three methods.
    """
     
    def inside(self, x,y):
        """Return True if point (x, y) is inside the obstacle."""
        raise NotImplementedError
    def normal(self, x,y):
        """Return the normal vector at point (x,y) on the obstacle."""
        raise NotImplementedError
    def plot(self, ax):
        """Drawing of the obstacle."""
        raise NotImplementedError

class Circle(Obstacle):
    """
    A circular obstacle in a Vicsek simulation.

    Parameters:

    xpos : float
        x-coordinate of the circle center
    ypos : float
        y-coordinate of the circle center
    radius : float
        Radius of the circle
    """
    def __init__(self, xpos, ypos, radius):
        self.xpos = xpos
        self.ypos = ypos
        self.radius = radius

    def inside(self, x, y):
        """Return True if point (x, y) is inside the circle."""
        return np.sqrt((self.xpos-x)**2+(self.ypos-y)**2) < (self.radius+0.05)
    
    def normal(self, x, y):
        """Return the normal vector at point (x,y) on the circle."""
        return np.array([x - self.xpos, y - self.ypos])
  
    def plot(self, ax):
        """Drawing of the circle"""
        circle = plt.Circle((self.xpos, self.ypos), self.radius, linewidth=2, edgecolor='turquoise', facecolor='turquoise')
        ax.add_patch(circle)


class Rect(Obstacle):
    """
    Rectangular obstacle in a Vicsek simulation.

    Parameters:

    x_min, x_max : float
        x-coordinates of the edges of the rectangle
    y_min, y_max : float
        y-coordinates of the edges of the rectangle
    """
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def inside(self, x, y):
        """Return True if point (x, y) is inside the rectangle."""
        return (self.x_min-0.05) <= x <= (self.x_max+0.05) and (self.y_min-0.05) <= y <= (self.y_max-0.05)
    
    def normal(self, x, y):
        """Return the normal vector at point (x,y) on the rectangle."""
        d = {
            'left':   abs(x - self.x_min),
            'right':  abs(x - self.x_max),
            'bottom': abs(y - self.y_min),
            'top':    abs(y - self.y_max)
        }
        face = min(d, key=d.get)
        if face == 'left':   
            return np.array([ -1.0, 0.0])
        if face == 'right':  
            return np.array([1.0, 0.0])
        if face == 'bottom': 
            return np.array([ 0.0, -1.0])
        return np.array([ 0.0,1.0])
    
    def plot(self, ax):
        """Drawing of the rectangle."""
        rect = Rectangle((self.x_min, self.y_min),
                        self.x_max - self.x_min,
                        self.y_max - self.y_min,
                        linewidth=2, edgecolor='turquoise', facecolor='turquoise')
        ax.add_patch(rect)

__all__ = ['Viscek_Model', 'Animation', "Obstacle", 'Circle', 'Rect']

def parse_args():
    """
    Parse arguments to initiate a Viscek simulation.

    Returns:
    args : argparse.Namespace
        Namespace containing simulation parameters:
        - N : int
            Number of particles (default 500)
        - distribution : str
            Noise distribution type (default "Uniform")
        - eta : float
            Noise amplitude (default 0.6)
        - steps : int
            Number of animation steps (default 100)
        - quiver_toggle : str
            Whether to display velocity vectors in animation ("on"/"off")

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--distribution", type=str, default= "Uniform")
    parser.add_argument("--eta", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--quiver_toggle", type=str, default="on")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    return args

def main():
    """
    Main function to run the Vicsek model simulation and animation.

    This function:
    1. Parses command-line arguments
    2. Initializes a Vicsek model with optional obstacles
    3. Creates an animations for the given Vicsek model instance
    """
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    obstacles = []
    model = Viscek_Model(N_particles=args.N, System_size=(5, 5), dt=0.1, R=0.25, Distribution=distributions[args.distribution],
                            eta=args.eta, v0=0.3, obstacles=obstacles)
    animation_instance = Animation(model)
    fix, ax = animation_instance.animate(steps = args.steps, quiver_toggle = args.quiver_toggle)
    plt.show()

if __name__ == '__main__':
    matplotlib.use("TkAgg")
    main()

def steven():
    print("This YEETis aasdasdasdasd placeholder function for testing purposes.")
=======
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
from matplotlib.animation import FuncAnimation
from random import choice
from List import DistributionType, distributions
import argparse

"""
Vicsek model simulation with obstacles.

This module implements a 2D Vicsek model for self-propelled particles,
including the interaction between hard circular and rectangular obstacles.

It supports real-time animation and measurement of collective observables:
order parameter, correlation time, correlation length, and susceptibility.
"""

class Viscek_Model:
    """
    Vicsek model for self-propelled particles in 2D

    Parameters:

    N_particles : int
        Number of particles.
    System_size : tuple
        Width and height of the simulation box.
    dt : float
        Time step.
    R : float
        Particle's interaction radius
    Distribution : DistributionType
        Distribution for angular noise
    eta : float
        Noise strength
    v0 : float
        Initial particle speed for all particles
    obstacles : list of optional obstacles from Obstacle
        List of optional obstacles (circle or rectangle)
    """
    
    def __init__(self, N_particles, System_size, dt, R, Distribution: DistributionType  ,eta, v0, obstacles = None):
        self.N_particles = N_particles
        self.System_size = System_size
        self.dt = dt
        self.R = R
        self.eta = eta
        self.v0 = v0
        self.Distribution = Distribution
        self.obstacles = obstacles if obstacles is not None else []
        self.t = 0

        self.t_hist   = []
        self.phi_hist = []
        self.Ct_hist  = []
        self.Cr_hist  = []
        self.chi_hist = []
        self.x = np.zeros(N_particles)
        self.y = np.zeros(N_particles)

        # Initialize particle positions avoiding obstacles
        for i in range(self.N_particles):
            while True:
                x_attempt = np.random.rand() * System_size[0]
                y_attempt = np.random.rand() * System_size[1]
                if not any(obs.inside(x_attempt,y_attempt) for obs in self.obstacles):
                    self.x[i] = x_attempt
                    self.y[i] = y_attempt
                    break
        self.theta_initial = np.random.rand(N_particles)*2*np.pi
        self.theta = self.theta_initial.copy()
        self.theta_ref = self.theta.copy()

    def simulation_step(self):
        updated_theta = np.zeros(self.N_particles)
        correlation_length_i = np.zeros(self.N_particles)
        for i in range(self.N_particles):
            mean_theta, neighbors =self.compute_mean_theta(i)
            noise = self.compute_noise()
            correlation_length_i[i] = self.compute_local_correlation(i, neighbors)

            theta_prop = mean_theta + noise
            px, py = self.new_position(i, theta_prop)

            # Collision check with obstacles
            collided, theta_final = self.handle_collisions(i, px, py, theta_prop)
            if not collided:
                self.x[i] = px % self.System_size[0]
                self.y[i] = py % self.System_size[1]
                theta_final = theta_prop

            updated_theta[i] = theta_final


        self.theta = updated_theta
        self.compute_observables(correlation_length_i)

    def compute_mean_theta(self, i):
        dx = self.x - self.x[i]
        dy = self.y - self.y[i]
        dx -= self.System_size[0] * np.rint(dx / self.System_size[0])
        dy -= self.System_size[1] * np.rint(dy / self.System_size[1])
        r = np.sqrt(dx**2 + dy**2)
        neighbors = (r < self.R) & (r > 0)
        sin_sum = np.sum(np.sin(self.theta[neighbors]))
        cos_sum = np.sum(np.cos(self.theta[neighbors]))
        if np.sum(neighbors) > 0:
            mean_theta = np.arctan2(sin_sum, cos_sum)
        else:
            mean_theta = self.theta[i]
        return mean_theta, neighbors
    
    def compute_noise(self):
        return self.eta * (self.Distribution.sample() - 0.5) * np.pi
    
    def compute_local_correlation(self, i , neighbors):
        if np.sum(neighbors) > 0:
            return np.mean(np.cos(self.theta[i] - self.theta[neighbors]))
        return 0
    
    def new_position(self, i, theta):
        px = self.x[i] + self.v0 * np.cos(theta) * self.dt
        py = self.y[i] + self.v0 * np.sin(theta) * self.dt
        return px, py
    
    def handle_collisions(self, i, px, py, theta_prop):
        for obs in self.obstacles:
            if obs.inside(px, py):
                theta_new = self.resolve_collision(i, px, py, theta_prop, obs)
                return True, theta_new
        return False, theta_prop

    def resolve_collision(self, i, px, py, theta_prop, obs):
        x0, y0 = self.x[i], self.y[i]
        vx, vy = px - x0, py - y0
        t_low, t_high = 0.0, 1.0

        for _ in range(8):
            t_mid = 0.5 * (t_low + t_high)
            xm, ym = x0 + vx * t_mid, y0 + vy * t_mid
            if obs.inside(xm, ym):
                t_high = t_mid
            else:
                t_low = t_mid

        xc, yc = x0 + vx * t_low, y0 + vy * t_low

        # Normal
        nx, ny = obs.normal(xc, yc)
        n_norm = np.hypot(nx, ny) + 1e-6
        nx, ny = nx / n_norm, ny / n_norm

        # Reflect velocity
        v_in = np.array([vx, vy])
        v_out = v_in - 2 * np.dot(v_in, [nx, ny]) * np.array([nx, ny])
        theta_reflected = np.arctan2(v_out[1], v_out[0])

        # Move particle slightly away
        tx, ty = -ny, nx
        eps = self.v0 * self.dt + 1e-3

        for alpha in (0.0, 0.5, -0.5, 1.0, -1.0):
            x_try = xc + eps * (nx + alpha * tx)
            y_try = yc + eps * (ny + alpha * ty)
            if not obs.inside(x_try, y_try):
                self.x[i] = x_try % self.System_size[0]
                self.y[i] = y_try % self.System_size[1]
                return theta_reflected

        self.x[i] = (xc + nx * eps) % self.System_size[0]
        self.y[i] = (yc + ny * eps) % self.System_size[1]
        return theta_reflected

    def compute_observables(self, correlation_length_i):
        vx_sum = np.sum(self.v0 * np.cos(self.theta))
        vy_sum = np.sum(self.v0 * np.sin(self.theta))
        phi = np.sqrt(vx_sum**2 + vy_sum**2) / (self.N_particles * self.v0)
        # Calculate correction time
        correlation_time = np.mean(np.cos(self.theta - self.theta_ref))
        # Calculate correction length
        correlation_length = np.mean(correlation_length_i)
        # Calculate susceptibility
        if len(self.phi_hist) > 1:
            chi_t = (np.var(self.phi_hist))
        else:
            chi_t = 0

        self.t += self.dt

        self.t_hist.append(self.t)
        self.phi_hist.append(phi)
        self.Ct_hist.append(correlation_time)
        self.Cr_hist.append(correlation_length)
        self.chi_hist.append(chi_t)
        self.theta_ref = self.theta.copy()


class Animation:
    """
    Animation for the Vicsek model.

    Parameters:

    model : ViscekModel
        The Viscek model for animation
    """
    def __init__(self, model):
        self.model = model
        self.quiver_toggle = "on"
        self.step = 0
        self.anim = None

    def animate(self, interval=30, steps=100, quiver_toggle= "on"):
        self.quiver_toggle = quiver_toggle
        self.max_steps = steps*10
        fig = plt.figure(figsize=(14, 8))
        gs = fig.add_gridspec(
            nrows=4,
            ncols=2,
            width_ratios=[3, 1], 
            hspace=0.35,
            wspace=0.25
        )
        ax_sim = fig.add_subplot(gs[:, 0])
        ax_ct  = fig.add_subplot(gs[0, 1])
        ax_cr  = fig.add_subplot(gs[1, 1])
        ax_chi = fig.add_subplot(gs[2, 1])
        ax_phi = fig.add_subplot(gs[3, 1])

        ax_sim.set_xlim(0, self.model.System_size[0])
        ax_sim.set_ylim(0, self.model.System_size[1])
        ax_sim.set_aspect('equal')

        for obs in self.model.obstacles:
            obs.plot(ax_sim)

        self.scatter = ax_sim.scatter(self.model.x, self.model.y, s=10, c='blue')
        if self.quiver_toggle == "on":
            self.quiver = ax_sim.quiver(self.model.x, self.model.y,
                                    np.cos(self.model.theta), np.sin(self.model.theta),
                                    color='red', scale=20)
        ax_sim.set_title('Vicsek Model Simulation')
        ax_phi.set_xlim(0, steps)
        ax_phi.set_ylim(-0, 1)
        ax_phi.set_xlabel('Time Steps')
        ax_phi.set_ylabel('Order Parameter φ')

        ax_ct.set_xlim(0, steps)
        ax_ct.set_ylim(-1, 1)
        ax_ct.set_ylabel('Correlation Time Ct')

        ax_cr.set_xlim(0, steps)
        ax_cr.set_ylim(-1, 1)
        ax_cr.set_ylabel('Correlation Length Cr')

        ax_chi.set_xlim(0, steps)
        ax_chi.set_ylim(0, 0.1)
        ax_chi.set_ylabel('Susceptibility χ')

        self.line_phi, = ax_phi.plot([], [], lw=2)
        self.line_Ct,  = ax_ct.plot([], [], lw=2)
        self.line_Cr,  = ax_cr.plot([], [], lw=2)
        self.line_chi, = ax_chi.plot([], [], lw=2)

        self.anim = FuncAnimation(fig, self.update, frames=steps, interval=interval, blit=True)
        return fig, self.anim

    def update(self, frame):
        self.model.simulation_step()
        self.scatter.set_offsets(np.c_[self.model.x, self.model.y])
        t = self.model.t_hist
        self.step += 1
        if self.step >= self.max_steps - 1:
            self.anim.event_source.stop()

        self.line_phi.set_data(t, self.model.phi_hist)
        self.line_Ct.set_data(t, self.model.Ct_hist)
        self.line_Cr.set_data(t, self.model.Cr_hist)
        self.line_chi.set_data(t, self.model.chi_hist)

        if self.quiver_toggle == "on":
            self.quiver.set_UVC(np.cos(self.model.theta), np.sin(self.model.theta))
            self.quiver.set_offsets(np.c_[self.model.x, self.model.y])
            return (
                self.scatter,
                self.quiver,
                self.line_phi,
                self.line_Ct,
                self.line_Cr,
                self.line_chi,
            )
        return (
                self.scatter,
                self.line_phi,
                self.line_Ct,
                self.line_Cr,
                self.line_chi,
            )
    
class Obstacle:
    """
    Base class for obstacles in a Vicsek simulation.
    All obstacles must implement these three methods.
    """
     
    def inside(self, x,y):
        """Return True if point (x, y) is inside the obstacle."""
        raise NotImplementedError
    def normal(self, x,y):
        """Return the normal vector at point (x,y) on the obstacle."""
        raise NotImplementedError
    def plot(self, ax):
        """Drawing of the obstacle."""
        raise NotImplementedError

class Circle(Obstacle):
    """
    A circular obstacle in a Vicsek simulation.

    Parameters:

    xpos : float
        x-coordinate of the circle center
    ypos : float
        y-coordinate of the circle center
    radius : float
        Radius of the circle
    """
    def __init__(self, xpos, ypos, radius):
        self.xpos = xpos
        self.ypos = ypos
        self.radius = radius

    def inside(self, x, y):
        """Return True if point (x, y) is inside the circle."""
        return np.sqrt((self.xpos-x)**2+(self.ypos-y)**2) < (self.radius+0.05)
    
    def normal(self, x, y):
        """Return the normal vector at point (x,y) on the circle."""
        return np.array([x - self.xpos, y - self.ypos])
  
    def plot(self, ax):
        """Drawing of the circle"""
        circle = plt.Circle((self.xpos, self.ypos), self.radius, linewidth=2, edgecolor='turquoise', facecolor='turquoise')
        ax.add_patch(circle)


class Rect(Obstacle):
    """
    Rectangular obstacle in a Vicsek simulation.

    Parameters:

    x_min, x_max : float
        x-coordinates of the edges of the rectangle
    y_min, y_max : float
        y-coordinates of the edges of the rectangle
    """
    def __init__(self, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def inside(self, x, y):
        """Return True if point (x, y) is inside the rectangle."""
        return (self.x_min-0.05) <= x <= (self.x_max+0.05) and (self.y_min-0.05) <= y <= (self.y_max-0.05)
    
    def normal(self, x, y):
        """Return the normal vector at point (x,y) on the rectangle."""
        d = {
            'left':   abs(x - self.x_min),
            'right':  abs(x - self.x_max),
            'bottom': abs(y - self.y_min),
            'top':    abs(y - self.y_max)
        }
        face = min(d, key=d.get)
        if face == 'left':   
            return np.array([ -1.0, 0.0])
        if face == 'right':  
            return np.array([1.0, 0.0])
        if face == 'bottom': 
            return np.array([ 0.0, -1.0])
        return np.array([ 0.0,1.0])
    
    def plot(self, ax):
        """Drawing of the rectangle."""
        rect = Rectangle((self.x_min, self.y_min),
                        self.x_max - self.x_min,
                        self.y_max - self.y_min,
                        linewidth=2, edgecolor='turquoise', facecolor='turquoise')
        ax.add_patch(rect)

__all__ = ['Viscek_Model', 'Animation', "Obstacle", 'Circle', 'Rect']

def parse_args():
    """
    Parse arguments to initiate a Viscek simulation.

    Returns:
    args : argparse.Namespace
        Namespace containing simulation parameters:
        - N : int
            Number of particles (default 500)
        - distribution : str
            Noise distribution type (default "Uniform")
        - eta : float
            Noise amplitude (default 0.6)
        - steps : int
            Number of animation steps (default 100)
        - quiver_toggle : str
            Whether to display velocity vectors in animation ("on"/"off")

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=500)
    parser.add_argument("--distribution", type=str, default= "Uniform")
    parser.add_argument("--eta", type=float, default=0.6)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--quiver_toggle", type=str, default="on")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()
    return args

def main():
    """
    Main function to run the Vicsek model simulation and animation.

    This function:
    1. Parses command-line arguments
    2. Initializes a Vicsek model with optional obstacles
    3. Creates an animations for the given Vicsek model instance
    """
    args = parse_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    obstacles = []
    model = Viscek_Model(N_particles=args.N, System_size=(5, 5), dt=0.1, R=0.25, Distribution=distributions[args.distribution],
                            eta=args.eta, v0=0.3, obstacles=obstacles)
    animation_instance = Animation(model)
    fix, ax = animation_instance.animate(steps = args.steps, quiver_toggle = args.quiver_toggle)
    plt.show()

if __name__ == '__main__':
    matplotlib.use("TkAgg")
    main()

def steven():
    print("This is a placeholder function for testing purposes.")  
>>>>>>> cef3c2bf8a1cacbc4c280b0b7506e737e157b20a
