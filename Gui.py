import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from List import *
from Visczek import *
import sys
import ast

class SimulationGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Viscek Model Simulation")
        self.master.geometry("1500x600")

        self.controls_frame = tk.Frame(self.master)
        self.controls_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nw")\
        
        self.number_of_particles_label = tk.Label(self.controls_frame, text="Number of Particles:", width = 15)
        self.number_of_particles_label.grid(row=0, column=0, sticky="w")
        self.number_of_particles_entry = tk.Entry(self.controls_frame)
        self.number_of_particles_entry.grid(row=0, column=1, sticky="w")
        self.number_of_particles_entry.insert(0, "100")

        self.system_size_label = tk.Label(self.controls_frame, text="System Size (Lx, Ly):")
        self.system_size_label.grid(row=1, column=0, sticky="w")

        self.system_size_entry_frame = tk.Frame(self.controls_frame)
        self.system_size_entry_frame.grid(row=1, column=1, sticky="w")
                
        self.system_size_xentry = tk.Entry(self.system_size_entry_frame, width=5)
        self.system_size_xentry.pack(side="left", padx=(0, 5))
        self.system_size_xentry.insert(0, "10")

        self.system_size_yentry = tk.Entry(self.system_size_entry_frame, width=5)
        self.system_size_yentry.pack(side="left")
        self.system_size_yentry.insert(0, "10")

        self.radius_label = tk.Label(self.controls_frame, text="Radius field of view:")
        self.radius_label.grid(row=2, column=0, sticky="w")
        self.radius_entry = tk.Entry(self.controls_frame)
        self.radius_entry.grid(row=2, column=1, sticky="w")
        self.radius_entry.insert(0, "1")

        self.distribution_plot = tk.Frame(master)
        self.distribution_plot.grid(row=0, column=1, sticky="w")

        self.noise_label = tk.Label(self.controls_frame, text="Noise distribution:")
        self.noise_label.grid(row=3, column=0, sticky="w")
        self.noise_var = tk.StringVar(master)
        self.noise_dropdown = ttk.Combobox(self.controls_frame, textvariable=self.noise_var, state="readonly")
        self.noise_dropdown['values'] = list(distributions.keys())
        self.noise_dropdown.grid(row=3, column=1, sticky="w")

        self.noise_parameters = tk.Frame(self.controls_frame)
        self.noise_parameters.grid(row=4, column=0, padx=5, pady=5, sticky="w", columnspan=2)
        self.noise_dropdown.bind("<<ComboboxSelected>>", self.update_distribution_parameters)
        self.noise_dropdown.set("Uniform")
        self.update_distribution_parameters()
        
        # Obstacles Input
        self.obstacles_label = tk.Label(self.controls_frame, text="Obstacles:")
        self.obstacles_label.grid(row=6, column=0, sticky="w")

        self.circle_input_parameters = tk.Label(self.controls_frame, text="(pos_x, pos_y, radius)")
        self.circle_input_parameters.grid(row=7, column=1, sticky="w")

        self.circle_label = tk.Label(self.controls_frame, text="Circle(s):")
        self.circle_label.grid(row=8, column=0, sticky="w")

        self.circle_entry = tk.Entry(self.controls_frame)
        self.circle_entry.grid(row=8, column=1, sticky="w")

        self.rect_input_parameters = tk.Label(self.controls_frame, text="(min_x, max_x, min_y, max_y)")
        self.rect_input_parameters.grid(row=9, column=1, sticky="w")

        self.rect_label = tk.Label(self.controls_frame, text="Rectangle(s):")
        self.rect_label.grid(row=10, column=0, sticky="w")

        self.rect_entry = tk.Entry(self.controls_frame)
        self.rect_entry.grid(row=10, column=1, sticky="w")

        # Start Button
        self.start_button = tk.Button(self.controls_frame, text="Start Simulation", command=self.start_simulation)
        self.start_button.grid(row=12, column=0, columnspan=2, pady=10)

        # Canvas for animation
        self.canvas_frame = tk.Frame(master, bd=1, relief=tk.SUNKEN)
        self.canvas_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nsew")
        
        # Configure grid weights to make canvas expandable
        master.grid_columnconfigure(1, weight=1)
        master.grid_rowconfigure(0, weight=1)
        
        # Set up the window close handler
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Initialize animation-related attributes
        self.anim = None
        self.fig = None
        self.canvas = None
        self.update_job = None 

        self.show_distribution_plot(distributions[self.noise_var.get()])

        self.master.update_idletasks()
        self.master.deiconify()
        self.master.lift()
        self.master.focus_force()

    def parse_circle_entry(self,text):
        if text.strip() == "":
            return []
        try:
            tuples = ast.literal_eval(f"[{text}]")
            if not all(isinstance(t, tuple) and len(t) == 3 for t in tuples):
                return None
            if not all(all(isinstance(n, (int, float)) for n in t) for t in tuples):
                return None
            return [Circle(*t) for t in tuples]
        except (ValueError, SyntaxError):
            return None
        
    def parse_rect_entry(self,text):
        if text.strip() == "":
            return []
        try:
            tuples = ast.literal_eval(f"[{text}]")
            if not all(isinstance(t, tuple) and len(t) == 4 for t in tuples):
                return None
            if not all(all(isinstance(n, (int, float)) for n in t) for t in tuples):
                return None
            return [Rect(*t) for t in tuples]
        except (ValueError, SyntaxError):
            return None

    def rect_input(self):
        return

    def obstacles_intput(self):
        return

    def schedule_distribution_update(self, event=None):
        if self.update_job is not None:
            self.master.after_cancel(self.update_job)
        self.update_job = self.master.after(1000, self.update_distribution_plot)

    def update_distribution_plot(self):
        # Get current distribution and update parameters
        distribution_name = self.noise_var.get()
        DistributionClass = distributions.get(distribution_name)

        if DistributionClass is None:
            return

        # Update params if possible
        params = {}
        for param_name, entry_widget in self.input_entries.items():
            try:
                params[param_name] = float(entry_widget.get())
            except ValueError:
                params[param_name] = DistributionClass.defaults.get(param_name, 0.0)

        DistributionClass.set_params(**params)
        self.show_distribution_plot(DistributionClass)


    def show_distribution_plot(self, distribution):
        # Clear previous plot if it exists
        if hasattr(self, 'distribution_fig') and self.distribution_fig is not None:
            plt.close(self.distribution_fig)

        # Create a new figure for the distribution plot
        self.distribution_fig, ax = plt.subplots(figsize=(4, 3))

        # Generate random samples
        samples = np.array([distribution.sample() for _ in range(100000)])

        # Plot histogram
        ax.hist(samples, bins=50, density=True, alpha=0.75, label=distribution.name)
        ax.set_title(f"Distribution: {distribution.name}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Probability Density")
        ax.legend()

        # Create a canvas for the distribution plot
        if hasattr(self, 'distribution_canvas'):
            self.distribution_canvas.get_tk_widget().destroy()

        self.distribution_canvas = FigureCanvasTkAgg(self.distribution_fig, master=self.distribution_plot)
        self.distribution_canvas.draw()
        self.distribution_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def system_size_read(self):

        # Read the system size from the entry fields
        x_dim = float(self.system_size_xentry.get())
        y_dim = float(self.system_size_yentry.get())
        system_size = (x_dim, y_dim)
        return system_size

    def on_closing(self):

        # Stop the animation if it exists
        if hasattr(self, 'anim') and self.anim is not None:
            self.anim.event_source.stop()
            self.anim._fig.clf()

        # Close any matplotlib figures
        if hasattr(self, 'fig') and self.fig is not None:
            plt.close(self.fig)
    
        # Destroy the tkinter window
        self.master.destroy()
        sys.exit()
    
    def Distribution_parameters(self):
        for widget in self.noise_parameters.winfo_children():
            widget.destroy()
        
        distribution = self.noise_var.get()
        DistributionClass = distributions.get(distribution)

        if not DistributionClass:
            return
        
        parameters = getattr(DistributionClass, 'param_names', [])
        self.input_entries = {}
        for i, param in enumerate(parameters):
            label = tk.Label(self.noise_parameters, text=param + ":")
            label.grid(row=i, column=0, sticky="w")
            entry = tk.Entry(self.noise_parameters)
            entry.grid(row=i, column=1, sticky="w")
            self.input_entries[param] = entry
            default_value = DistributionClass.defaults.get(param, 0.0)
            entry.insert(0, str(default_value))
            entry.bind("<KeyRelease>", self.schedule_distribution_update)


    def update_distribution_parameters(self, event = None):
        self.Distribution_parameters()
        self.update_distribution_plot()


    def start_simulation(self):
        if hasattr(self, 'anim') and self.anim is not None:
            try:
                self.anim.event_source.stop()
            except AttributeError:
                pass
        if self.fig:
            plt.close(self.fig)
            self.fig = None
        
        if hasattr(self, 'canvas') and self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        # Clear previous canvas
        for widget in self.canvas_frame.winfo_children():
            widget.destroy()

        self.number_of_particles = int(self.number_of_particles_entry.get())
        self.system_size = self.system_size_read()
        self.radius = float(self.radius_entry.get())
        self.noise_name = self.noise_var.get()
        self.dt = 0.1
        self.length_time = 1000
        self.eta = 0.1
        self.v0 = 1.0
        self.Distribution = distributions[self.noise_name]
        
        circle_inputs= self.circle_entry.get()
        circles = self.parse_circle_entry(circle_inputs)

        rect_inputs= self.rect_entry.get()
        rects = self.parse_rect_entry(rect_inputs)

        self.obstacles = rects + circles
        







        params = {}
        for param_name, entry_widget in self.input_entries.items():
            params[param_name] = float(entry_widget.get())
        self.Distribution.set_params(**params)

        simulation = Viscek_Model(
            N_particles=self.number_of_particles,
            System_size=self.system_size,
            dt=self.dt,
            length_time=self.length_time,
            R=self.radius,
            Distribution=self.Distribution,
            eta=self.eta,
            v0=self.v0,
            obstacles = self.obstacles
        )

        animator = Animation(simulation)
        self.animator = animator
        fig, anim = animator.animate(interval=30, steps=100)


        self.canvas = FigureCanvasTkAgg(fig, master=self.canvas_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        self.fig = fig
        self.anim = anim






if __name__ == "__main__":
    root = tk.Tk()
    gui = SimulationGUI(root)
    root.mainloop()