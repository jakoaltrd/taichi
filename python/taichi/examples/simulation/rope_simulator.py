import taichi as ti

# Initialize Taichi for CPU
ti.init(arch=ti.cpu)

# Global simulation parameters
n_particles = 15
n_segments = n_particles - 1
particle_mass = 1.0
stiffness = 10000.0
damping = 0.5  # Adjusted from implicit_mass_spring for potentially more explicit damping
rest_length = 0.1  # Assuming each segment is 0.1 units long
gravity = ti.Vector([0, -9.81])  # 2D gravity
dt = 0.0005  # Smaller time step for explicit methods

positions = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
velocities = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
forces = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
inv_mass = ti.field(dtype=ti.f32, shape=n_particles)

@ti.kernel
def initialize_particles():
    # Initial position of the first particle
    start_pos = ti.Vector([0.1, 0.8]) 
    for i in range(n_particles):
        positions[i] = start_pos + ti.Vector([i * rest_length, 0])
        velocities[i] = ti.Vector([0, 0])
        
        if i == 0: # Fix the first particle
            inv_mass[i] = 0.0
        else:
            inv_mass[i] = 1.0 / particle_mass

@ti.kernel
def compute_forces():
    # Clear forces
    for i in range(n_particles):
        forces[i] = ti.Vector([0, 0])

    # Gravity
    for i in range(n_particles):
        if inv_mass[i] > 0: # Only apply gravity to non-fixed particles
            forces[i] += gravity * particle_mass

    # Spring forces and Damping forces
    for i in range(n_segments):
        p1_idx = i
        p2_idx = i + 1
        
        pos_p1 = positions[p1_idx]
        pos_p2 = positions[p2_idx]
        
        dist_vec = pos_p2 - pos_p1
        current_length = dist_vec.norm()
        
        if current_length > 1e-6: # Avoid division by zero if particles coincide
            direction_vec = dist_vec / current_length
            
            # Spring force (Hooke's Law)
            spring_force_magnitude = stiffness * (current_length - rest_length)
            spring_force_vec = spring_force_magnitude * direction_vec
            
            forces[p1_idx] += spring_force_vec
            forces[p2_idx] -= spring_force_vec
            
            # Damping force
            vel_p1 = velocities[p1_idx]
            vel_p2 = velocities[p2_idx]
            relative_velocity = vel_p2 - vel_p1
            
            # Project relative velocity onto the spring direction
            projected_relative_velocity = relative_velocity.dot(direction_vec)
            damping_force_magnitude = damping * projected_relative_velocity
            damping_force_vec = damping_force_magnitude * direction_vec
            
            forces[p1_idx] += damping_force_vec
            forces[p2_idx] -= damping_force_vec

@ti.kernel
def integrate():
    for i in range(n_particles):
        if inv_mass[i] > 0: # Only integrate non-fixed particles
            velocities[i] += (forces[i] * inv_mass[i]) * dt
            positions[i] += velocities[i] * dt

def main():
    initialize_particles()

    window = ti.ui.Window("Rope Simulator", (800, 800), vsync=True)
    canvas = window.get_canvas()
    
    pause = False

    while window.running:
        # Handle events
        for e in window.get_events(ti.ui.PRESS):
            if e.key == ti.ui.ESCAPE:
                window.running = False
            elif e.key == ti.ui.SPACE:
                pause = not pause
        
        if not pause:
            for _ in range(10): # Substeps for stability with explicit integration
                compute_forces()
                integrate()

        # Render
        # Prepare data for drawing lines (segments)
        # We need to convert segment data to a format GUI can understand easily.
        # Create a temporary field or numpy array for line endpoints if many segments.
        # For a small number of segments, direct calls are fine.
        
        # Draw segments
        for i in range(n_segments):
            p1 = positions[i]
            p2 = positions[i+1]
            canvas.line(p1, p2, width=3, color=(0.8, 0.8, 0.8))

        # Draw particles
        canvas.circles(positions, radius=5, color=(0.2, 0.5, 1.0))
        
        window.show()

if __name__ == "__main__":
    main()
