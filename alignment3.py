import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2


def surface_function(x, y, params, bed_size):
    """
    Evaluate scaled surface function: f(x/bed_size, y/bed_size) + a6
    f(u,v) = a1*u^2 + a2*v^2 + a3*u*v + a4*u + a5*v + a6
    params = [a1, a2, a3, a4, a5, a6]
    """
    a1, a2, a3, a4, a5, a6 = params

    # Scale coordinates
    u = x / bed_size
    v = y / bed_size

    return a1 * u**2 + a2 * v**2 + a3 * u * v + a4 * u + a5 * v + a6

def objective_function(variables, x_grid, y_grid, ref_surface, sim_surface, bed_size):
    """
    Objective function to minimize: ||f(2*(x-x0)/L, 2*(y-y0)/L) - f(2*(x-x0-x1)/L, 2*(y-y0-y1)/L)||^2
    variables = [a1, a2, a3, a4, a5, a6, x0, y0, x1, y1]
    """
    # Extract parameters
    surface_params = variables[:6]  # [a1, a2, a3, a4, a5, a6]
    x0, y0 = variables[6], variables[7]  # bed offset parameters
    x1, y1 = variables[8], variables[9]  # toolhead offset parameters

    # Evaluate fitted surfaces
    fitted_ref = surface_function(x_grid - x0, y_grid - y0, surface_params, bed_size)
    fitted_sim = surface_function(x_grid - x0 - x1, y_grid - y0 - y1, surface_params, bed_size)

    # Compute residuals
    ref_residual = (ref_surface - fitted_ref).flatten()
    sim_residual = (sim_surface - fitted_sim).flatten()

    # Combined objective: sum of squared residuals
    total_residual = np.concatenate([ref_residual, sim_residual])
    return np.sum(total_residual**2)

def fit_surfaces_unified(x_grid, y_grid, ref_surface, sim_surface, bed_size, initial_guess=None):
    """
    Unified optimization approach to find surface parameters and offsets
    Returns: [a1, a2, a3, a4, a5, a6, x0, y0, x1, y1], optimization_result
    """
    if initial_guess is None:
        # Initialize with reasonable guesses
        # Surface parameters: start with small values, scaled appropriately
        # The bed_distortion/2 * (4/bed_size^2) scaling factor for quadratic terms
        quad_scale = 0.01  # Conservative initial guess
        initial_guess = [quad_scale, -quad_scale, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # Suppress optimization warnings for cleaner output during Monte Carlo
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Perform optimization
        result = minimize(
            objective_function,
            initial_guess,
            args=(x_grid, y_grid, ref_surface, sim_surface, bed_size),
            method='BFGS',
            options={'disp': False, 'maxiter': 1000}
        )

    return result.x, result

def generate_surface_data(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y, machine_noise_std_dev, probe_std_dev, rng):
    """Generate reference and simulated surface data with noise"""

    # Close enough with small offset that it can be added to each surface to show method is not impacted by divots in the bed, as long as the overall surface can be fitted to a quadratic.
    machine_noise = machine_noise_std_dev * rng.standard_normal(gridx.shape)

    ref_surface = (bed_distortion / 2) * ((2 * (gridx - bed_offset_x) / bed_size)**2 - (2 * (gridy - bed_offset_y) / bed_size)**2) + machine_noise + probe_std_dev * rng.standard_normal(gridx.shape)

    sim_surface = (bed_distortion / 2) * ((2 * (gridx - bed_offset_x - sim_offset_x) / bed_size)**2 - (2 * (gridy - bed_offset_y - sim_offset_y) / bed_size)**2) + machine_noise + probe_std_dev * rng.standard_normal(gridx.shape)

    return ref_surface, sim_surface

def run_single_optimization(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y, machine_noise_std_dev, probe_std_dev, rng):
    """Run a single optimization attempt and return results"""
    try:
        ref_surface, sim_surface = generate_surface_data(
            gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y, machine_noise_std_dev, probe_std_dev, rng
        )

        optimal_params, opt_result = fit_surfaces_unified(gridx, gridy, ref_surface, sim_surface, bed_size)

        x_offset = optimal_params[8]
        y_offset = optimal_params[9]
        x_error = abs(x_offset - sim_offset_x)
        y_error = abs(y_offset - sim_offset_y)

        return {
            'x_offset': x_offset,
            'y_offset': y_offset,
            'x_error': x_error,
            'y_error': y_error,
            'success': opt_result.success,
            'objective': opt_result.fun,
            'surface_params': optimal_params[:6]
        }
    except Exception:
        return {
            'x_offset': np.nan,
            'y_offset': np.nan,
            'x_error': np.nan,
            'y_error': np.nan,
            'success': False,
            'objective': np.inf,
            'surface_params': [np.nan] * 6
        }

def run_default_workflow(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y, machine_noise_std_dev, probe_std_dev):
    """
    Run the default single-example workflow with visualization
    """
    print("Running single example for visualization...")
    rng = np.random.default_rng()
    ref_surface, sim_surface = generate_surface_data(
        gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y, machine_noise_std_dev, probe_std_dev, rng
    )

    optimal_params, opt_result = fit_surfaces_unified(gridx, gridy, ref_surface, sim_surface, bed_size)

    # Extract results
    surface_params_opt = optimal_params[:6]
    bed_offset_x_result = optimal_params[6]
    bed_offset_y_result = optimal_params[7]
    x_offset_unified = optimal_params[8]
    y_offset_unified = optimal_params[9]

    # Generate fitted surfaces for visualization
    ref_fitted_unified = surface_function(gridx - bed_offset_x_result, gridy - bed_offset_y_result, surface_params_opt, bed_size)
    sim_fitted_unified = surface_function(gridx - bed_offset_x_result - x_offset_unified, gridy - bed_offset_y_result - y_offset_unified, surface_params_opt, bed_size)

    print("Single example results:")
    print(f"  X offset: {x_offset_unified:.6f} (error: {abs(x_offset_unified - sim_offset_x):.6f})")
    print(f"  Y offset: {y_offset_unified:.6f} (error: {abs(y_offset_unified - sim_offset_y):.6f})")
    print(f"  Optimization success: {opt_result.success}")
    print(f"  Final objective: {opt_result.fun:.8f}")
    print()

    # Visualization of single example
    fig = plt.figure(figsize=(16, 10))

    # Row 1: Original surfaces
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    surf1 = ax1.plot_surface(gridx, gridy, ref_surface, cmap='viridis', alpha=0.8)
    ax1.set_title('Reference Surface (Raw)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    surf2 = ax2.plot_surface(gridx, gridy, sim_surface, cmap='viridis', alpha=0.8)
    ax2.set_title('Shifted Surface (Raw)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Row 1: Fitted surfaces
    ax3 = fig.add_subplot(2, 3, 3, projection='3d')
    surf3 = ax3.plot_surface(gridx, gridy, ref_fitted_unified, cmap='plasma', alpha=0.8)
    ax3.set_title('Reference Surface (Fitted)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Row 2: Fitted sim surface and residuals
    ax4 = fig.add_subplot(2, 3, 4, projection='3d')
    surf4 = ax4.plot_surface(gridx, gridy, sim_fitted_unified, cmap='plasma', alpha=0.8)
    ax4.set_title('Shifted Surface (Fitted)')
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    ref_residual = ref_surface - ref_fitted_unified
    surf5 = ax5.plot_surface(gridx, gridy, ref_residual, cmap='coolwarm', alpha=0.8)
    ax5.set_title('Reference Residuals')
    ax5.set_xlabel('X')
    ax5.set_ylabel('Y')
    ax5.set_zlabel('Z')

    ax6 = fig.add_subplot(2, 3, 6, projection='3d')
    sim_residual = sim_surface - sim_fitted_unified
    surf6 = ax6.plot_surface(gridx, gridy, sim_residual, cmap='coolwarm', alpha=0.8)
    ax6.set_title('Shifted Residuals')
    ax6.set_xlabel('X')
    ax6.set_ylabel('Y')
    ax6.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

def monte_carlo_bivariate_analysis(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y,
                                   machine_noise_std_dev, probe_std_dev, n_attempts=500):
    """
    Run Monte Carlo analysis and plot bivariate distribution with confidence ellipses
    """
    print(f"Running {n_attempts} attempts for bivariate analysis...")

    rng = np.random.default_rng()

    attempts = []
    for _ in range(n_attempts):
        result = run_single_optimization(
            gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y,
            machine_noise_std_dev, probe_std_dev, rng
        )
        attempts.append(result)

    # Extract successful results
    x_offsets = [r['x_offset'] for r in attempts if r['success'] and not np.isnan(r['x_offset'])]
    y_offsets = [r['y_offset'] for r in attempts if r['success'] and not np.isnan(r['y_offset'])]

    if len(x_offsets) < 10:
        print("Warning: Too few successful optimizations for reliable statistics")
        return

    # Convert to numpy arrays
    x_offsets = np.array(x_offsets)
    y_offsets = np.array(y_offsets)

    # Calculate statistics
    mean_x = np.mean(x_offsets)
    mean_y = np.mean(y_offsets)

    # Calculate covariance matrix
    data = np.column_stack([x_offsets, y_offsets])
    cov_matrix = np.cov(data.T)

    print(f"Success rate: {len(x_offsets)/n_attempts:.1%}")
    print(f"Mean X offset: {mean_x:.6f} (true: {sim_offset_x:.6f}, bias: {mean_x - sim_offset_x:.6f})")
    print(f"Mean Y offset: {mean_y:.6f} (true: {sim_offset_y:.6f}, bias: {mean_y - sim_offset_y:.6f})")
    print(f"Std X: {np.sqrt(cov_matrix[0,0]):.6f}")
    print(f"Std Y: {np.sqrt(cov_matrix[1,1]):.6f}")
    print(f"Correlation: {cov_matrix[0,1] / np.sqrt(cov_matrix[0,0] * cov_matrix[1,1]):.3f}")

    # Plot bivariate distribution with confidence ellipses
    plot_bivariate_distribution(x_offsets, y_offsets, sim_offset_x, sim_offset_y,
                               mean_x, mean_y, cov_matrix, probe_std_dev)

def plot_bivariate_distribution(x_offsets, y_offsets, true_x, true_y, mean_x, mean_y, cov_matrix, noise_level):
    """
    Plot bivariate distribution with 1, 2, and 3 sigma confidence ellipses
    """
    fig, ax1 = plt.subplots(1, 1, figsize=(15, 6))

    # Left plot: Scatter plot with confidence ellipses
    ax1.scatter(x_offsets, y_offsets, alpha=0.6, s=20, label='Monte Carlo results')
    ax1.plot(true_x, true_y, 'r*', markersize=15, label=f'True offset ({true_x}, {true_y})')
    ax1.plot(mean_x, mean_y, 'go', markersize=8, label=f'Sample mean ({mean_x:.3f}, {mean_y:.3f})')

    # Calculate and plot confidence ellipses
    colors = ['blue', 'orange', 'red']
    sigmas = [1, 2, 3]

    major_axis_std_dev = None
    minor_axis_std_dev = None

    for sigma, color in zip(sigmas, colors):
        # Chi-square critical value for 2 DOF and given confidence level
        chi2_val = chi2.ppf(chi2.cdf(sigma**2, df=1), df=2)  # Convert 1D sigma to 2D chi-square

        # Eigenvalues and eigenvectors for ellipse orientation
        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)

        # Calculate ellipse parameters
        angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
        width = 2 * np.sqrt(chi2_val * eigenvals[0])
        height = 2 * np.sqrt(chi2_val * eigenvals[1])

        if sigma == 1:
            major_axis_std_dev = max(width, height) / 2
            minor_axis_std_dev = min(width, height) / 2

        # Create ellipse
        ellipse = plt.matplotlib.patches.Ellipse(
            (mean_x, mean_y), width, height, angle=angle,
            fill=False, color=color, linewidth=2, linestyle='--',
            label=f'{sigma}σ ellipse'
        )
        ax1.add_patch(ellipse)

    ax1.text(0.02, 0.98, f'1σ major axis: {major_axis_std_dev:.3f}\n1σ minor axis: {minor_axis_std_dev:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('X Offset')
    ax1.set_ylabel('Y Offset')
    ax1.set_title(f'Bivariate Distribution of Offset Estimates\n(Noise level: {noise_level})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')

    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    num_points_on_axis = 14
    bed_size = 180
    bed_distortion = 0.2
    bed_offset_x = 40.0
    bed_offset_y = 10.0
    sim_offset_x = 0.8
    sim_offset_y = 0.8
    probe_std_dev = 0.001
    machine_noise_std_dev = 0.0075

    # Generate coordinate grids
    axis_points = np.linspace(-bed_size/2, bed_size/2, num=num_points_on_axis)
    gridx, gridy = np.meshgrid(axis_points, axis_points)

    print("=" * 55)
    print("Surface Offset Detection using Unified Optimization")
    print("=" * 55)
    print(f"Bed size: {bed_size}")
    print(f"Bed distortion: {bed_distortion}")
    print(f"True offsets: X={sim_offset_x}, Y={sim_offset_y}")
    print(f"Grid points: {num_points_on_axis}x{num_points_on_axis}")
    print()

    # Run default workflow
    run_default_workflow(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y,
                        machine_noise_std_dev, probe_std_dev)

    # Bivariate Monte Carlo analysis
    print("\n" + "="*55)
    print("MONTE CARLO BIVARIATE DISTRIBUTION ANALYSIS")
    print("="*55)

    monte_carlo_bivariate_analysis(gridx, gridy, bed_size, bed_distortion, bed_offset_x, bed_offset_y, sim_offset_x, sim_offset_y,
                                  machine_noise_std_dev, probe_std_dev, n_attempts=1000)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
