import numpy
import scipy.integrate
import umbridge


earth_radius = 6371000 # m
earth_position = [0, 0] # m
earth_mass = 5.972e24 * 0.1 # kg

moon_radius = 1737000 # m
moon_position = [384400000 * 0.1, 0] # m
moon_mass = 7.34767309e22 # kg

gravitational_constant = 6.67408e-11 # m^3 kg^-1 s^-2

# Function taking a 2D launch vector from earth and returning a trajectory of a spacecraft past the Moon, computed via an ODE.
def trajectory(launch_velocity, launch_angle):
    # Define the ODE
    def ode(t, y):
        # y[0] is x, y[1] is y, y[2] is vx, y[3] is vy
        d_earth = numpy.linalg.norm(y[0:2] - earth_position)
        d_moon = numpy.linalg.norm(y[0:2] - moon_position)

        return [y[2],
                y[3],
                gravitational_constant * (earth_mass * -y[0]/d_earth**3 + moon_mass * (moon_position[0] - y[0])/d_moon**3),
                gravitational_constant * (earth_mass * -y[1]/d_earth**3 + moon_mass * (moon_position[1] - y[1])/d_moon**3)]

    # Define events
    def event_pass_moon(t, y):
        return y[0] - moon_position[0]

    def event_impact_moon(t, y):
        return numpy.linalg.norm(numpy.asarray(y[0:2]) - numpy.asarray(moon_position)) - moon_radius

    def event_impact_earth(t, y):
        return numpy.linalg.norm(numpy.asarray(y[0:2]) - numpy.asarray(earth_position)) - (earth_radius - 10)

    def event_t_5e4(t, y):
        return t - 5e4

    event_impact_moon.terminal = True
    event_impact_earth.terminal = True


    # Define the initial conditions
    launch_vector = [launch_velocity * numpy.sin(launch_angle), launch_velocity * numpy.cos(launch_angle)]
    y0 = [0, earth_radius, launch_vector[0], launch_vector[1]]

    # Define time span
    t0 = 0
    tf = 1e5

    # Integrate ODE
    sol = scipy.integrate.solve_ivp(ode, [t0, tf], y0, dense_output=True, first_step = 5e2, max_step = 5e2, rtol=1e99, atol=1e99,
                                    events = [event_pass_moon, event_impact_moon, event_impact_earth, event_t_5e4])

    return sol

class TestModel(umbridge.Model):
    def __init__(self):
        super().__init__("forward") # Give a name to the model

    def get_input_sizes(self, config):
        return [2]

    def get_output_sizes(self, config):
        return [201, 201, 1, 1, 2]

    def __call__(self, parameters, config):
        sol = trajectory(parameters[0][0] * 1e3, parameters[0][1])
        trajectory_x = sol.y[0] * 1e-3
        trajectory_y = sol.y[1] * 1e-3
        padded_trajectory_x = numpy.pad(trajectory_x, (0, 201 - len(trajectory_x)), 'constant', constant_values=(trajectory_x[-1]))
        padded_trajectory_y = numpy.pad(trajectory_y, (0, 201 - len(trajectory_y)), 'constant', constant_values=(trajectory_y[-1]))
        # return trajectory x and y

        return [padded_trajectory_x.tolist(),
                padded_trajectory_y.tolist(),
                [1 if len(sol.t_events[1]) > 0 else 0], # moon impact?
                [1 if len(sol.t_events[2]) > 0 else 0], # earth impact?
                [0,0] if len(sol.t_events[3]) < 1 else (sol.y_events[3][0][0:2] * 1e-3).tolist() # Position at specific point in time (unless crashed earlier)
                ]

    def supports_evaluate(self):
        return True

umbridge.serve_models([TestModel()], 4242)