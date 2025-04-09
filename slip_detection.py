# TOOL_CODE
import numpy as np
import time
import cv2 # Using OpenCV for potentially faster interpolation
# from scipy.interpolate import griddata # Alternative interpolation
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# TOOL_CODE
import numpy as np
import time
import cv2 # Using OpenCV for potentially faster interpolation
# from scipy.interpolate import griddata # Alternative interpolation
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors

# --- Configuration ---
ROWS_LOW = 8
COLS_LOW = 4
UPSCALE_FACTOR = 10 # Example: Creates an 80x40 grid
ROWS_HI = ROWS_LOW * UPSCALE_FACTOR
COLS_HI = COLS_LOW * UPSCALE_FACTOR

# --- VelostatTracker Class ---
class VelostatTracker:
    def __init__(self, rows_low=ROWS_LOW, cols_low=COLS_LOW, upscale_factor=UPSCALE_FACTOR, verbose=False):
        self.rows_low, self.cols_low = rows_low, cols_low
        self.grid_shape_low = (rows_low, cols_low)
        self.upscale_factor = upscale_factor
        self.rows_hi = rows_low * upscale_factor
        self.cols_hi = cols_low * upscale_factor
        self.grid_shape_hi = (self.rows_hi, self.cols_hi)
        self.verbose = verbose

        # --- Calibration (CRITICAL - MUST BE DETERMINED EXPERIMENTALLY) ---
        self.VOLTAGE_MIN = 0.5
        self.VOLTAGE_MAX = 1.5
        self.PRESSURE_MAX = 5.0 # Arbitrary max pressure units
        self.PRESSURE_THRESHOLD = 0.1 # Minimum pressure to be considered active

        # --- State Variables ---
        self.P_low_current = np.zeros(self.grid_shape_low)
        self.P_low_previous = np.zeros(self.grid_shape_low)
        self.delta_P_low = np.zeros(self.grid_shape_low)
        self.P_hires_current = np.zeros(self.grid_shape_hi)
        self.delta_P_hires_mag = np.zeros(self.grid_shape_hi) # Store magnitude

        # Use low-res CoP for robustness
        self.CoP_low_current = np.array([(rows_low - 1) / 2.0, (cols_low - 1) / 2.0])
        self.CoP_low_previous = np.array([(rows_low - 1) / 2.0, (cols_low - 1) / 2.0])
        self.delta_CoP_low_vector = np.array([0.0, 0.0])
        self.active_mask_low = np.zeros(self.grid_shape_low, dtype=bool)
        self.active_mask_hires = np.zeros(self.grid_shape_hi, dtype=bool) # Mask for hires grid

        self.time_previous = time.monotonic()
        self.first_reading = True
        self.previous_entropy = 0.0

        # --- Coordinates (Low Res for CoP, High Res for Viz/Entropy) ---
        self.y_coords_low, self.x_coords_low = np.meshgrid(
            np.arange(self.rows_low), np.arange(self.cols_low), indexing='ij'
        )
        # High-res coords might be useful for region definition if needed
        # self.y_coords_hi, self.x_coords_hi = np.meshgrid(
        #     np.arange(self.rows_hi), np.arange(self.cols_hi), indexing='ij'
        # )


        # --- Thresholds (NEEDS EXPERIMENTAL TUNING) ---
        # Based on low-res CoP shift (units = taxels)
        self.DELTA_COP_THRESHOLD_INCIPIENT = 0.30 # Tune!
        self.DELTA_COP_THRESHOLD_GROSS = 0.8 # Tune!
        # Based on high-res delta_P entropy/distribution
        self.ENTROPY_THRESHOLD_INCIPIENT = 2.5 # Tune! (Entropy scale depends on bins/data)
        self.PERIPHERAL_VAR_RATIO_THRESHOLD = 1.5 # Tune! (Ratio of peripheral variance to central variance)
        self.CENTRAL_REGION_RADIUS_HIRES = 3.0 * self.upscale_factor # Radius on high-res grid

        if self.verbose:
            print("VelostatTracker Initialized.")

    # --- Internal Methods ---
    def _voltage_to_pressure(self, V_map):
        # --- MUST BE REPLACED WITH ACTUAL CALIBRATION ---
        # Simple linear scaling placeholder:
        V_clipped = np.clip(V_map, self.VOLTAGE_MIN, self.VOLTAGE_MAX)
        P_map = self.PRESSURE_MAX * (V_clipped - self.VOLTAGE_MIN) / (self.VOLTAGE_MAX - self.VOLTAGE_MIN + 1e-6)
        P_map = np.maximum(P_map, 0)
        return P_map

    def _calculate_cop_low(self, P_map_low, active_mask_low):
        total_pressure = np.sum(P_map_low[active_mask_low])
        if total_pressure < 1e-6:
            # Keep previous CoP if no contact or very low pressure
            return self.CoP_low_previous

        cop_y = np.sum(P_map_low[active_mask_low] * self.y_coords_low[active_mask_low]) / total_pressure
        cop_x = np.sum(P_map_low[active_mask_low] * self.x_coords_low[active_mask_low]) / total_pressure
        return np.array([cop_y, cop_x]) # [row, col]

    def _upscale(self, P_map_low):
         # Use cv2.resize for potentially faster interpolation than griddata
         # INTER_CUBIC provides smoother results than INTER_LINEAR
         P_map_hi = cv2.resize(P_map_low, (self.cols_hi, self.rows_hi), interpolation=cv2.INTER_CUBIC)
         # Ensure non-negative pressures after interpolation
         P_map_hi = np.maximum(P_map_hi, 0)
         return P_map_hi

    def _calculate_entropy(self, data_vector):
        # Calculates entropy of the histogram of a data vector (positive values only)
        positive_data = data_vector[data_vector > 1e-6] # Use only positive values
        if positive_data.size < 10: return 0.0 # Need enough points for meaningful entropy

        # Use automatic binning, consider density=True for scale invariance
        counts, bin_edges = np.histogram(positive_data, bins='auto', density=True)
        bin_widths = np.diff(bin_edges)
        # Filter out zero counts to avoid log(0)
        valid_indices = counts > 0
        if not np.any(valid_indices): return 0.0

        probabilities = counts[valid_indices] * bin_widths[valid_indices]
        # Small epsilon added for stability, though density=True helps
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
        return entropy

    def _classify_regions_hires(self, active_mask_hires, cop_low):
        # Classify high-res pixels based on distance from low-res CoP mapped to high-res grid
        if not np.any(active_mask_hires):
            return np.zeros_like(active_mask_hires), np.zeros_like(active_mask_hires)

        cop_hi_y = cop_low[0] * self.upscale_factor
        cop_hi_x = cop_low[1] * self.upscale_factor

        # Create high-res coordinate grid efficiently ONLY IF NEEDED here
        y_coords_hi, x_coords_hi = np.meshgrid(
            np.arange(self.rows_hi), np.arange(self.cols_hi), indexing='ij'
        )

        distances_sq = (y_coords_hi - cop_hi_y)**2 + (x_coords_hi - cop_hi_x)**2
        # Use radius squared for efficiency
        central_mask = active_mask_hires & (distances_sq <= self.CENTRAL_REGION_RADIUS_HIRES**2)
        peripheral_mask = active_mask_hires & (distances_sq > self.CENTRAL_REGION_RADIUS_HIRES**2)
        return central_mask, peripheral_mask


    # --- Main Processing Function ---
    def process_reading(self, V_current_raw):
        current_time = time.monotonic()
        dt = current_time - self.time_previous
        self.time_previous = current_time

        if V_current_raw.shape != self.grid_shape_low:
            raise ValueError(f"Input shape mismatch. Expected {self.grid_shape_low}, got {V_current_raw.shape}")

        # --- State Updates ---
        self.P_low_previous = np.copy(self.P_low_current)
        self.CoP_low_previous = np.copy(self.CoP_low_current)

        # --- Calculations ---
        self.P_low_current = self._voltage_to_pressure(V_current_raw)
        self.active_mask_low = self.P_low_current > self.PRESSURE_THRESHOLD
        num_active_low = np.sum(self.active_mask_low)

        # --- Handle NO_CONTACT ---
        if num_active_low < 3: # Require a few active taxels
            if self.verbose: print("DEBUG: No/Low contact.")
            self.first_reading = True
            self.previous_entropy = 0.0
            # Reset fields
            self.delta_P_low = np.zeros(self.grid_shape_low)
            self.P_hires_current = np.zeros(self.grid_shape_hi)
            self.delta_P_hires_mag = np.zeros(self.grid_shape_hi)
            self.delta_CoP_low_vector = np.array([0.0, 0.0])
            features = {
                'timestamp': current_time,
                'dt': dt,
                'slip_state': 'NO_CONTACT',
                'CoP_low': self.CoP_low_current.tolist(),
                'delta_CoP_low_vector': [0.0, 0.0],
                'delta_CoP_magnitude': 0.0,
                'num_active_low': 0,
                'hires_delta_P_entropy': 0.0,
                'grasp_type': 'NONE'
            }
            return 'NO_CONTACT', features, self.P_hires_current, self.delta_P_hires_mag

        # --- Upscale Current Pressure ---
        self.P_hires_current = self._upscale(self.P_low_current)
        # Also create a high-res active mask (can be useful)
        self.active_mask_hires = self.P_hires_current > self.PRESSURE_THRESHOLD

        # --- Calculate Deltas and CoP ---
        delta_CoP_magnitude = 0.0
        current_entropy = 0.0
        slip_state = 'NO_SLIP' # Default
        grasp_type = 'SOFT'  # Default grasp type

        if self.first_reading:
            self.delta_P_low = np.zeros(self.grid_shape_low)
            # Upscale zero delta gives zero delta hires
            self.delta_P_hires_mag = np.zeros(self.grid_shape_hi)
            self.delta_CoP_low_vector = np.array([0.0, 0.0])
            # Calculate initial CoP
            self.CoP_low_current = self._calculate_cop_low(self.P_low_current, self.active_mask_low)
            self.first_reading = False
            slip_state = 'INITIAL_CONTACT'
            grasp_type = 'NONE'

        else:
            # Low-res delta pressure
            self.delta_P_low = self.P_low_current - self.P_low_previous
            P_hires_previous = self._upscale(self.P_low_previous)
            delta_P_hires = self.P_hires_current - P_hires_previous
            self.delta_P_hires_mag = np.abs(delta_P_hires)

            # Low-res CoP calculation
            self.CoP_low_current = self._calculate_cop_low(self.P_low_current, self.active_mask_low)
            self.delta_CoP_low_vector = self.CoP_low_current - self.CoP_low_previous
            delta_CoP_magnitude = np.linalg.norm(self.delta_CoP_low_vector)

            # Determine slip state and grasp type based on motion
            if delta_CoP_magnitude > 0.7:  # High motion - possible slip
                slip_state = 'SLIP'
                grasp_type = 'POWER'
            elif delta_CoP_magnitude > 0.3:  # Moderate motion
                slip_state = 'MOTION'
                grasp_type = 'POWER'
            else:
                slip_state = 'STABLE'
                grasp_type = 'SOFT'

        # --- Prepare Features Dictionary ---
        features = {
            'timestamp': current_time,
            'dt': dt,
            'slip_state': slip_state,
            'CoP_low': self.CoP_low_current.tolist(),
            'delta_CoP_low_vector': self.delta_CoP_low_vector.tolist(),
            'delta_CoP_magnitude': delta_CoP_magnitude,
            'num_active_low': num_active_low,
            'hires_delta_P_entropy': current_entropy,
            'grasp_type': grasp_type
        }

        # Return state, features, and the high-res maps for visualization
        return slip_state, features, self.P_hires_current, self.delta_P_hires_mag

class FEMSetup:
    """Handles loading/managing FEM mesh and stiffness matrix."""
    def __init__(self, mesh_file, stiffness_matrix_file):
        # First load mesh to get num_nodes
        self.nodes, self.elements = self._load_mesh(mesh_file)
        self.num_nodes = self.nodes.shape[0]
        # Store node coordinates for interpolation mapping
        self.node_coords = self.nodes[:, :3] # Assuming first 3 columns are x, y, z

        # Then load stiffness matrix which depends on num_nodes
        self.stiffness_matrix_K = self._load_stiffness_matrix(stiffness_matrix_file)

        print(f"FEM Setup: Loaded {self.num_nodes} nodes, K matrix shape {self.stiffness_matrix_K.shape}")
        if self.stiffness_matrix_K.shape[0] != 3 * self.num_nodes:
             print("WARNING: Stiffness matrix dimension mismatch with number of nodes!")


    def _load_mesh(self, filename):
        # Placeholder: Implement loading from your mesh file format (e.g., .inp, .vtk)
        print(f"Placeholder: Loading mesh from {filename}")
        # Example: nodes = array([[x1,y1,z1], [x2,y2,z2], ...])
        #          elements = array([[n1,n2,n3,n4,n5,n6,n7,n8], ...]) node indices
        # For now, return dummy data matching expected shapes for coding flow
        num_dummy_nodes = 50 # Example
        nodes = np.random.rand(num_dummy_nodes, 3)
        elements = np.random.randint(0, num_dummy_nodes, size=(num_dummy_nodes // 2, 8))
        return nodes, elements

    def _load_stiffness_matrix(self, filename):
        # Placeholder: Implement loading K from file (e.g., .npy, .mtx)
        print(f"Placeholder: Loading stiffness matrix K from {filename}")
        # K should be (3*num_nodes, 3*num_nodes)
        # For now, return dummy data matching expected shapes
        num_dofs = 3 * self.num_nodes
        # Create a sparse diagonal matrix for demonstration purposes
        # A real K matrix is dense or sparse depending on FEM formulation
        from scipy.sparse import diags
        K = diags([1.0] * num_dofs, 0, shape=(num_dofs, num_dofs), format='csr') # Use sparse format
        return K


class DisplacementEstimator:
    """Estimates 3D nodal displacement vector U from sensor readings."""
    def __init__(self, fem_setup, grid_shape_hi):
        self.fem_setup = fem_setup
        self.rows_hi, self.cols_hi = grid_shape_hi
        self.grid_shape_hi = grid_shape_hi

        # --- Parameters for Approximation (CRITICAL - NEED TUNING/MODELING) ---
        self.pressure_to_dz_factor = 0.05 # dz = factor * P_hires (Tune this!)
        self.deltaP_to_dxy_factor = 0.02 # |dxy| = factor * |delta_P_hires| (Tune this!)

        # Create high-res grid coordinates for interpolation source points
        # Assuming sensor surface lies primarily in XY plane for simplicity here
        # Adjust ranges based on your actual sensor dimensions
        sensor_width, sensor_height = 1.0, 2.0 # Example dimensions
        self.y_coords_hi, self.x_coords_hi = np.meshgrid(
            np.linspace(0, sensor_height, self.rows_hi),
            np.linspace(0, sensor_width, self.cols_hi),
            indexing='ij'
        )
        # Flatten source coordinates for interpolation
        self.source_points_xy = np.vstack([self.x_coords_hi.ravel(),
                                           self.y_coords_hi.ravel()]).T


    def estimate_nodal_displacement_U(self, P_hires, delta_P_hires_mag, delta_CoP_low_vector):
        """
        Estimates the 3N x 1 nodal displacement vector U using approximations.

        Returns:
            U_nodes_est (np.array): The (3 * num_nodes) x 1 estimated displacement vector.
        """
        # --- 1. Estimate dz_hires (Normal Displacement) ---
        # Approximation: dz is proportional to local pressure
        dz_hires = self.pressure_to_dz_factor * P_hires
        dz_hires_flat = dz_hires.ravel()

        # --- 2. Estimate dxy_hires (Tangential Displacement) ---
        # Approximation: Magnitude proportional to delta_P_hires_mag, direction from delta_CoP
        dxy_mag_hires = self.deltaP_to_dxy_factor * delta_P_hires_mag

        # Use delta_CoP vector [dRow, dCol] to estimate direction
        # Normalize the direction vector (handle zero vector case)
        norm = np.linalg.norm(delta_CoP_low_vector)
        if norm > 1e-6:
            # Be careful with coordinate systems: delta_CoP is [dRow, dCol]
            # Assume Row maps to Y, Col maps to X for displacement
            direction_vector = np.array([delta_CoP_low_vector[1], delta_CoP_low_vector[0]]) / norm # [dx_dir, dy_dir]
        else:
            direction_vector = np.array([0.0, 0.0])

        # Apply direction to magnitude
        dx_hires = dxy_mag_hires * direction_vector[0]
        dy_hires = dxy_mag_hires * direction_vector[1]
        dx_hires_flat = dx_hires.ravel()
        dy_hires_flat = dy_hires.ravel()

        # Combine estimated displacements at high-res grid points
        U_hires_est_flat = np.vstack([dx_hires_flat, dy_hires_flat, dz_hires_flat]).T # Shape: (rows*cols, 3)

        # --- 3. Interpolate U_hires_est onto FEM Node Locations ---
        # We need to map the high-res grid points (source) to the FEM node locations (target)
        # Assuming FEM nodes are defined in the same coordinate system as the sensor surface
        target_points_xy = self.fem_setup.node_coords[:, :2] # Use only X, Y coords of nodes for interpolation

        from scipy.interpolate import griddata

        # Interpolate dx, dy, dz separately
        # Use linear interpolation, 'nearest' for speed, or 'cubic' for smoothness
        # Provide fill_value=0 for nodes outside the convex hull of sensor points
        interp_method = 'linear'
        fill_val = 0.0

        print(f"DEBUG: Interpolating {U_hires_est_flat.shape[0]} hires points onto {target_points_xy.shape[0]} nodes.")
        if self.source_points_xy.shape[0] != U_hires_est_flat.shape[0]:
             print("ERROR: Mismatch between source points and hires displacement data length!")
             return np.zeros(3 * self.fem_setup.num_nodes) # Return zero vector on error


        try:
            U_nodes_dx = griddata(self.source_points_xy, dx_hires_flat, target_points_xy, method=interp_method, fill_value=fill_val)
            U_nodes_dy = griddata(self.source_points_xy, dy_hires_flat, target_points_xy, method=interp_method, fill_value=fill_val)
            U_nodes_dz = griddata(self.source_points_xy, dz_hires_flat, target_points_xy, method=interp_method, fill_value=fill_val)
        except Exception as e:
            print(f"ERROR during griddata interpolation: {e}")
            # Potential issues: Degenerate input points (e.g., all in a line), insufficient points
            return np.zeros(3 * self.fem_setup.num_nodes)


        # --- 4. Assemble the final U vector ---
        # Interleave dx, dy, dz for each node: [dx1, dy1, dz1, dx2, dy2, dz2, ...]
        U_nodes_est = np.ravel(np.vstack([U_nodes_dx, U_nodes_dy, U_nodes_dz]).T)

        if U_nodes_est.shape[0] != 3 * self.fem_setup.num_nodes:
             print(f"ERROR: Final U vector shape mismatch! Expected {3 * self.fem_setup.num_nodes}, Got {U_nodes_est.shape[0]}")
             return np.zeros(3 * self.fem_setup.num_nodes)


        return U_nodes_est


class ForceCalculator:
    """Calculates force F = KU."""
    def __init__(self, fem_setup):
        self.K = fem_setup.stiffness_matrix_K # Get the matrix from setup

    def calculate_force_F(self, U_nodes_est):
        """
        Calculates the 3N x 1 nodal force vector F.

        Args:
            U_nodes_est (np.array): The (3 * num_nodes) x 1 estimated displacement vector.

        Returns:
            F_nodes_est (np.array): The (3 * num_nodes) x 1 estimated force vector.
            F_total (np.array): The resultant [Fx, Fy, Fz] force vector (sum of nodal forces).
        """
        if U_nodes_est.shape[0] != self.K.shape[1]:
             print(f"ERROR: Dimension mismatch between K ({self.K.shape}) and U ({U_nodes_est.shape})")
             return np.zeros_like(U_nodes_est), np.zeros(3)

        # Calculate nodal forces
        F_nodes_est = self.K @ U_nodes_est

        # Calculate total resultant force by summing nodal forces
        # Reshape F to (num_nodes, 3) and sum along the nodes axis (axis=0)
        F_total = np.sum(F_nodes_est.reshape(-1, 3), axis=0)

        return F_nodes_est, F_total


# --- Simulation/Live Data Loop (MODIFIED) ---
def run_tracker_with_force_estimation():
    # --- Initialization ---
    tracker = VelostatTracker(verbose=False) # Keep slip detection
    viz = RealTimeVisualizer(ROWS_HI, COLS_HI, tracker.CoP_low_current * tracker.upscale_factor) # Keep visualization

    # NEW: FEM Setup (Replace with actual file paths)
    try:
        fem_setup = FEMSetup(mesh_file="path/to/your/mesh.inp",
                             stiffness_matrix_file="path/to/your/stiffness_matrix.npy")
    except FileNotFoundError:
        print("ERROR: Mesh or Stiffness matrix file not found. Exiting.")
        return

    # NEW: Estimators/Calculators
    displacement_estimator = DisplacementEstimator(fem_setup, tracker.grid_shape_hi)
    force_calculator = ForceCalculator(fem_setup)

    print("\n--- Starting Tracking with Force Estimation ---")
    # --- Extended simulation with more frames and complex motions ---
    num_frames = 200  # Extended to 200 frames to show all scenarios
    sim_noise = 0.05 * (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN)
    
    # Initial position
    center_row_start, center_col_start = 3.5, 1.5
    center_row, center_col = center_row_start, center_col_start
    
    # Motion pattern controls
    first_contact_frame = 10    # Frame when initial contact happens
    stable_period_frames = 30   # Period of stable contact
    slip_start_frame = 60       # Frame when slip starts
    return_frame = 100          # Frame when object returns to start position
    second_slip_frame = 140     # Second slip event
    release_frame = 180         # When grasp is released
    
    # Motion parameters
    slip_speed_row, slip_speed_col = 0.08, 0.05
    pressure_amplitude = 1.0   # For pressure modulation

    all_features = []
    all_forces = []

    for frame in range(num_frames):
        # --- Generate Simulated Voltage Data with Complex Motion Pattern ---
        V_sim = np.zeros(tracker.grid_shape_low) + tracker.VOLTAGE_MIN
        
        # Define complex motion pattern to demonstrate all states
        if frame < first_contact_frame:
            # No contact state
            pass
            
        elif frame < first_contact_frame + stable_period_frames:
            # Initial contact and stable grasp
            contact_strength = min(1.0, (frame - first_contact_frame) / 10.0)  # Gradual contact
            yy, xx = tracker.y_coords_low, tracker.x_coords_low
            dist_sq = (yy - center_row)**2 + (xx - center_col)**2
            pressure_factor = contact_strength * np.exp(-dist_sq / (1.3**2))
            V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor
            
        elif frame < slip_start_frame:
            # Full stable contact
            yy, xx = tracker.y_coords_low, tracker.x_coords_low
            dist_sq = (yy - center_row)**2 + (xx - center_col)**2
            pressure_factor = np.exp(-dist_sq / (1.3**2))
            # Add slight pressure variations to simulate micro-movements
            pressure_mod = 1.0 + 0.1 * np.sin(frame * 0.2)
            V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor * pressure_mod
            
        elif frame < return_frame:
            # Slip motion
            center_row += slip_speed_row * (1 + 0.2 * np.sin(frame * 0.1))  # Add variation to speed
            center_col += slip_speed_col
            yy, xx = tracker.y_coords_low, tracker.x_coords_low
            dist_sq = (yy - center_row)**2 + (xx - center_col)**2
            pressure_factor = np.exp(-dist_sq / (1.3**2))
            V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor
            
        elif frame < second_slip_frame:
            # Return to stable position
            center_row = center_row_start + 0.5  # Slightly offset from original
            center_col = center_col_start + 0.3
            yy, xx = tracker.y_coords_low, tracker.x_coords_low
            dist_sq = (yy - center_row)**2 + (xx - center_col)**2
            pressure_factor = np.exp(-dist_sq / (1.3**2))
            V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor
            
        elif frame < release_frame:
            # Second slip in different direction
            center_row -= slip_speed_row * 0.5
            center_col += slip_speed_col * 1.2
            yy, xx = tracker.y_coords_low, tracker.x_coords_low
            dist_sq = (yy - center_row)**2 + (xx - center_col)**2
            pressure_factor = np.exp(-dist_sq / (1.3**2))
            V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor
            
        else:
            # Gradual release
            release_progress = (frame - release_frame) / (num_frames - release_frame)
            contact_strength = max(0, 1.0 - release_progress * 2)  # Faster fadeout
            if contact_strength > 0:
                yy, xx = tracker.y_coords_low, tracker.x_coords_low
                dist_sq = (yy - center_row)**2 + (xx - center_col)**2
                pressure_factor = contact_strength * np.exp(-dist_sq / (1.3**2))
                V_sim = tracker.VOLTAGE_MIN + (tracker.VOLTAGE_MAX - tracker.VOLTAGE_MIN) * pressure_factor
        
        # Add noise to all frames
        if frame > first_contact_frame:
            V_sim += np.random.normal(0, sim_noise, tracker.grid_shape_low)
            V_sim = np.clip(V_sim, tracker.VOLTAGE_MIN - 0.1, tracker.VOLTAGE_MAX + 0.1)

        # --- Process Reading (Slip Detection) ---
        slip_state, features, P_hires, delta_P_hires_mag = tracker.process_reading(V_sim)
        all_features.append(features)

        # --- Estimate Displacement (NEW) ---
        delta_cop_vec = np.array(features.get('delta_CoP_low_vector', [0.0, 0.0]))
        U_nodes_estimated = displacement_estimator.estimate_nodal_displacement_U(
            P_hires, delta_P_hires_mag, delta_cop_vec
        )

        # --- Estimate Force (NEW) ---
        _, F_total_estimated = force_calculator.calculate_force_F(U_nodes_estimated)
        all_forces.append(F_total_estimated)

        # --- Update Visualization ---
        cop_hires_coords = features.get('CoP_low', [0,0]) * np.array([tracker.upscale_factor, tracker.upscale_factor])
        if not viz.update(P_hires, delta_P_hires_mag, cop_hires_coords, slip_state, features, F_total_estimated):
             print("Plot closed.")
             break

        print(f"Frame {frame}: St={slip_state}, F_est=[{F_total_estimated[0]:.2f}, {F_total_estimated[1]:.2f}, {F_total_estimated[2]:.2f}]")
        
        # Add a small delay for better visualization
        if frame > first_contact_frame:
            time.sleep(0.03)

    print("\n--- Tracking Complete ---")
    viz.close()

    # --- Plot results (Optional: Add force plot) ---
    times = [f['timestamp'] - all_features[0]['timestamp'] for f in all_features]
    forces_array = np.array(all_forces)

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    # Slip features (reuse previous plotting code for axs[0], axs[1])
    d_cop = [f['delta_CoP_magnitude'] for f in all_features]
    entropy = [f['hires_delta_P_entropy'] for f in all_features]
    axs[0].plot(times, d_cop, 'b-o', label='Delta CoP Mag (Low Res)')
    axs[0].axhline(tracker.DELTA_COP_THRESHOLD_INCIPIENT, color='orange', ls='--', label='Incipient Thr (CoP)')
    axs[0].axhline(tracker.DELTA_COP_THRESHOLD_GROSS, color='red', ls='--', label='Gross Thr (CoP)')
    axs[0].set_ylabel('Delta CoP')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(times, entropy, 'g-s', label='Entropy (Hires |Delta P|)')
    axs[1].axhline(tracker.ENTROPY_THRESHOLD_INCIPIENT, color='orange', ls='--', label='Incipient Thr (Entropy)')
    axs[1].set_ylabel('Entropy')
    axs[1].legend()
    axs[1].grid(True)

    # Force plot
    axs[2].plot(times, forces_array[:, 0], 'r-', label='Est. Fx')
    axs[2].plot(times, forces_array[:, 1], 'g-', label='Est. Fy')
    axs[2].plot(times, forces_array[:, 2], 'b-', label='Est. Fz')
    axs[2].set_ylabel('Estimated Force')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)


    plt.suptitle('Sensor Features & Estimated Force Over Time')
    plt.tight_layout()
    plt.show()


# --- RealTimeVisualizer Class (IMPROVED) ---
class RealTimeVisualizer:
    def __init__(self, rows_hi, cols_hi, initial_cop_hi):
        self.rows_hi, self.cols_hi = rows_hi, cols_hi
        # Create figure with two subplots
        self.fig, self.axs = plt.subplots(1, 2, figsize=(16, 6))

        # --- Pressure Map & CoP Plot ---
        ax1 = self.axs[0]
        self.im_p = ax1.imshow(np.zeros((rows_hi, cols_hi)), 
                              cmap='viridis', vmin=0, vmax=1.0, 
                              interpolation='nearest', origin='upper', 
                              aspect='equal')
        ax1.set_title(f"Pressure Map & CoP (8x4)")
        ax1.set_xlabel(f"Columns (4)")
        ax1.set_ylabel(f"Rows (8)")
        ax1.set_xticks(np.arange(0, cols_hi, 10))
        ax1.set_yticks(np.arange(0, rows_hi, 10))
        self.fig.colorbar(self.im_p, ax=ax1, label='Pressure', shrink=0.75)
        
        # CoP markers and trail
        self.cop_trail_len = 30
        self.cop_deque = deque(maxlen=self.cop_trail_len)
        self.cop_plot, = ax1.plot([], [], 'r.-', alpha=0.7, markersize=4, 
                                 linewidth=1.5, label='CoP Trail')
        self.cop_marker, = ax1.plot([], [], 'ro', markersize=8, 
                                   alpha=0.8, label='Current CoP')
        ax1.legend(loc='upper right', fontsize='small')
        ax1.invert_yaxis()

        # --- Motion Markers Plot ---
        ax2 = self.axs[1]
        ax2.set_facecolor('#F0F0F0')
        ax2.set_title("Motion Vectors (8x4 Grid)")
        ax2.set_xlabel("Columns (4)")
        ax2.set_ylabel("Rows (8)")
        ax2.set_xticks(np.arange(0, cols_hi, 10))
        ax2.set_yticks(np.arange(0, rows_hi, 10))
        ax2.set_xlim(-0.5, cols_hi - 0.5)
        ax2.set_ylim(-0.5, rows_hi - 0.5)
        ax2.set_aspect('equal')
        
        # Use 8x4 grid for motion markers (matching the low-res taxels)
        y_centers = np.linspace(5, rows_hi-5, 8)  # Centers of the 8 rows
        x_centers = np.linspace(5, cols_hi-5, 4)  # Centers of the 4 columns
        grid_y, grid_x = np.meshgrid(y_centers, x_centers, indexing='ij')
        
        self.grid_points_y = grid_y.flatten()
        self.grid_points_x = grid_x.flatten()
        
        # Motion markers (blue dots at each taxel center)
        self.motion_markers = ax2.scatter(
            self.grid_points_x, self.grid_points_y,
            c='blue', s=60, alpha=0.6, zorder=1
        )
        
        # Initialize motion arrows
        self.motion_arrows = ax2.quiver(
            self.grid_points_x, self.grid_points_y,
            np.zeros_like(self.grid_points_x),
            np.zeros_like(self.grid_points_y),
            scale=15, width=0.005, color='red', 
            alpha=0.0, zorder=2
        )
        
        # Central motion indicator with larger arrow
        center_y = rows_hi // 2
        center_x = cols_hi // 2
        self.main_motion_arrow = ax2.quiver(
            [center_x], [center_y],
            [0], [0],
            scale=10, width=0.008, color='darkred',
            alpha=1.0, zorder=4
        )
        
        # Add grid lines to show the 8x4 grid
        for i in range(9):
            y = i * (rows_hi / 8)
            ax2.axhline(y, color='gray', linestyle=':', alpha=0.3)
        for i in range(5):
            x = i * (cols_hi / 4)
            ax2.axvline(x, color='gray', linestyle=':', alpha=0.3)
        
        # Motion state indicator text
        self.motion_state_text = ax2.text(
            cols_hi * 0.05, rows_hi * 0.05, "NO CONTACT",
            fontsize=14, color='gray', weight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round', 
                     edgecolor='black', pad=0.5),
            zorder=5
        )
        
        # Direction indicator text
        self.direction_text = ax2.text(
            cols_hi * 0.05, rows_hi * 0.15, "",
            fontsize=12, color='black',
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'),
            zorder=5
        )
        
        # Store previous state for transition effects
        self.prev_state = "NO_CONTACT"
        self.prev_motion_magnitude = 0.0
        self.motion_avg = 0.0  # Rolling average of motion magnitude
        self.contact_mask = np.zeros((rows_hi, cols_hi), dtype=bool)
        self.frame_count = 0
        
        # Image saving for various states
        self.saved_states = {
            'NO_CONTACT': False,
            'INITIAL_CONTACT': False,
            'STABLE': False,
            'MOTION': False,
            'SLIP': False
        }
        
        # Video recording
        self.record_video = True
        if self.record_video:
            self.video_writer = None
            self.temp_frame_dir = 'temp_frames'
            if not os.path.exists(self.temp_frame_dir):
                os.makedirs(self.temp_frame_dir)
        
        ax2.invert_yaxis()

        # Main title
        self.title = self.fig.suptitle("Tactile Sensor Analysis - State: NO CONTACT | Grasp: NONE | dCoP: 0.000")
        plt.tight_layout(rect=[0, 0.03, 1, 0.93])
        self.fig.canvas.draw()
        plt.show(block=False)
        self.running = True
        self.fig.canvas.mpl_connect('close_event', self.handle_close)

    def update(self, P_hires, delta_P_hires_mag, cop_hires, slip_state, features, F_total_est=None):
        if not self.running: 
            return False
            
        self.frame_count += 1

        # Update Pressure Map
        p_max = np.percentile(P_hires[P_hires > 0], 99) if np.any(P_hires > 0) else 1.0
        self.im_p.set_data(P_hires)
        self.im_p.set_clim(vmin=0, vmax=max(p_max, 0.1))
        
        # Create contact mask for regions with significant pressure
        self.contact_mask = P_hires > 0.2  # Threshold for "contact"

        # Update CoP and trail
        if np.all(np.isfinite(cop_hires)):
            self.cop_deque.append(cop_hires[::-1])
            trail = np.array(self.cop_deque) if len(self.cop_deque) > 0 else np.array([[0, 0]])
            if len(trail) > 0:
                self.cop_plot.set_data(trail[:, 0], trail[:, 1])
                self.cop_marker.set_data([cop_hires[1]], [cop_hires[0]])

            # Get motion vector from delta_CoP or force estimation
            delta_cop_vec = np.array(features.get('delta_CoP_low_vector', [0, 0]))
            motion_magnitude = np.linalg.norm(delta_cop_vec)
            
            # Update rolling average of motion magnitude for better stability
            alpha = 0.3  # Weight for new value (0-1)
            self.motion_avg = (1 - alpha) * self.motion_avg + alpha * motion_magnitude
            
            # Override slip_state if needed based on actual motion
            corrected_slip_state = slip_state
            if slip_state == 'STABLE' and self.motion_avg > 0.15:
                # Override to MOTION if there's actually significant movement
                corrected_slip_state = 'MOTION'
            elif slip_state == 'MOTION' and self.motion_avg > 0.6:
                # Override to SLIP if there's very large movement
                corrected_slip_state = 'SLIP'
                
            # Use force vector as secondary motion indicator if available
            force_vec = np.array([0, 0]) if F_total_est is None else np.array([F_total_est[0], F_total_est[1]])
            force_magnitude = np.linalg.norm(force_vec)
            
            # Direction text update
            if motion_magnitude > 0.1:
                direction = ""
                if abs(delta_cop_vec[0]) > abs(delta_cop_vec[1]) * 1.5:
                    direction = "Moving Vertically"
                elif abs(delta_cop_vec[1]) > abs(delta_cop_vec[0]) * 1.5:
                    direction = "Moving Horizontally"
                else:
                    direction = "Moving Diagonally"
                
                if delta_cop_vec[0] < 0:
                    direction += " Up" if "Vertically" in direction else " Right" if "Horizontally" in direction else " Up-Right"
                elif delta_cop_vec[0] > 0:
                    direction += " Down" if "Vertically" in direction else " Left" if "Horizontally" in direction else " Down-Left"
                
                self.direction_text.set_text(direction)
                self.direction_text.set_visible(True)
            else:
                self.direction_text.set_visible(False)
            
            # State transition handling with visual indicators
            state_changed = (corrected_slip_state != self.prev_state)
            
            # Update motion state text and color based on slip state
            if corrected_slip_state == 'NO_CONTACT':
                self.motion_state_text.set_text("NO CONTACT")
                self.motion_state_text.set_color('gray')
                if state_changed:
                    self.motion_state_text.set_bbox(dict(facecolor='lightgray', alpha=0.9, boxstyle='round', edgecolor='black'))
            elif corrected_slip_state == 'INITIAL_CONTACT':
                self.motion_state_text.set_text("INITIAL CONTACT")
                self.motion_state_text.set_color('blue')
                if state_changed:
                    self.motion_state_text.set_bbox(dict(facecolor='lightblue', alpha=0.9, boxstyle='round', edgecolor='blue'))
            elif corrected_slip_state == 'SLIP':
                self.motion_state_text.set_text("SLIP DETECTED")
                self.motion_state_text.set_color('red')
                if state_changed:
                    self.motion_state_text.set_bbox(dict(facecolor='mistyrose', alpha=0.9, boxstyle='round', edgecolor='red'))
            elif corrected_slip_state == 'MOTION':
                self.motion_state_text.set_text("MOTION DETECTED")
                self.motion_state_text.set_color('darkorange')
                if state_changed:
                    self.motion_state_text.set_bbox(dict(facecolor='peachpuff', alpha=0.9, boxstyle='round', edgecolor='orange'))
            else:  # STABLE
                self.motion_state_text.set_text("STABLE CONTACT")
                self.motion_state_text.set_color('green')
                if state_changed and motion_magnitude < 0.05:
                    self.motion_state_text.set_bbox(dict(facecolor='palegreen', alpha=0.9, boxstyle='round', edgecolor='green'))
            
            # Save images for each state when we first encounter it (and it's stable)
            if self.frame_count > 10 and not self.saved_states[corrected_slip_state]:
                if self.frame_count % 3 == 0:  # Only try every 3rd frame to avoid rapid state changes
                    self.saved_states[corrected_slip_state] = True
                    self.save_state_image(corrected_slip_state)
            
            self.prev_state = corrected_slip_state
            self.prev_motion_magnitude = motion_magnitude
            
            # Arrow styling based on state
            if corrected_slip_state == 'SLIP':
                arrow_color = 'red'
                motion_scale = 2.5
                arrow_alpha = 0.9
            elif corrected_slip_state == 'MOTION':
                arrow_color = 'darkorange'
                motion_scale = 3.0
                arrow_alpha = 0.8
            elif corrected_slip_state == 'STABLE' and motion_magnitude > 0.05:
                arrow_color = 'gold'
                motion_scale = 3.5
                arrow_alpha = 0.7
            else:
                arrow_color = 'lightgrey'
                motion_scale = 5.0
                arrow_alpha = 0.4
            
            # Whether to show arrows - more sensitive detection of motion
            show_arrows = (motion_magnitude > 0.02 or 
                          force_magnitude > 0.05 or 
                          corrected_slip_state in ['SLIP', 'MOTION'])
            
            if show_arrows:
                # Sample pressure at the grid points to determine where to show arrows
                pressure_at_grid = np.zeros(len(self.grid_points_x))
                for i, (x, y) in enumerate(zip(self.grid_points_x, self.grid_points_y)):
                    y_idx = int(y)
                    x_idx = int(x)
                    if 0 <= y_idx < self.rows_hi and 0 <= x_idx < self.cols_hi:
                        pressure_at_grid[i] = P_hires[y_idx, x_idx]
                
                # Only show arrows where there's pressure
                contact_points = pressure_at_grid > 0.1
                
                # Normalize and scale vectors for visualization
                if motion_magnitude > 1e-6:
                    motion_vec_normalized = delta_cop_vec / motion_magnitude
                    # Note: swap [y,x] to [x,y] for plotting
                    motion_x = motion_vec_normalized[1] * motion_scale * motion_magnitude
                    motion_y = motion_vec_normalized[0] * motion_scale * motion_magnitude
                    
                    # Fill in arrow vectors only for contact points
                    U = np.zeros_like(self.grid_points_x)
                    V = np.zeros_like(self.grid_points_y)
                    
                    # Set vectors only where there's contact
                    U[contact_points] = motion_x
                    V[contact_points] = motion_y
                    
                    # Update arrows appearance
                    self.motion_arrows.set_UVC(U, V)
                    self.motion_arrows.set_color(arrow_color)
                    self.motion_arrows.set_alpha(arrow_alpha)
                    
                    # Update main motion arrow (larger, centered)
                    self.main_motion_arrow.set_UVC([motion_x * 1.5], [motion_y * 1.5])
                    self.main_motion_arrow.set_color('darkred' if corrected_slip_state == 'SLIP' else 'darkorange')
                    
                elif force_magnitude > 1e-6:
                    # No significant motion vector, try using force vector
                    force_vec_normalized = force_vec / force_magnitude
                    force_x = -force_vec_normalized[0] * 3.0  # Negate to show direction of object motion
                    force_y = -force_vec_normalized[1] * 3.0  # Negate to show direction of object motion
                    
                    # Fill in arrow vectors only for contact points
                    U = np.zeros_like(self.grid_points_x)
                    V = np.zeros_like(self.grid_points_y)
                    
                    # Set vectors only where there's contact
                    U[contact_points] = force_x
                    V[contact_points] = force_y
                    
                    self.motion_arrows.set_UVC(U, V)
                    self.motion_arrows.set_color(arrow_color)
                    self.motion_arrows.set_alpha(arrow_alpha)
                    
                    self.main_motion_arrow.set_UVC([force_x * 1.5], [force_y * 1.5])
                    self.main_motion_arrow.set_color('darkred' if corrected_slip_state == 'SLIP' else 'darkorange')
                else:
                    # Neither motion nor force vector available, hide arrows
                    self.motion_arrows.set_alpha(0)
                    self.main_motion_arrow.set_UVC([0], [0])
            else:
                # No significant motion, hide arrows
                self.motion_arrows.set_alpha(0)
                self.main_motion_arrow.set_UVC([0], [0])
        
        # Update title with state and grasp type
        grasp_type = features.get('grasp_type', 'NONE')
        delta_cop = features.get('delta_CoP_magnitude', 0.0)
        self.title.set_text(f"Tactile Sensor Analysis - State: {slip_state} | Grasp: {grasp_type} | dCoP: {delta_cop:.3f}")
        
        # Update pressure map title
        self.axs[0].set_title(f"Pressure Map & CoP (Max: {np.max(P_hires):.2f})")

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        
        # Save frame for video
        if self.record_video:
            self.save_frame_for_video()
            
        return self.running

    def save_state_image(self, state):
        """Save an image of the current state."""
        output_dir = "state_images"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        filename = f"{output_dir}/state_{state.lower()}.png"
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved image for state: {state} as {filename}")
        
    def save_frame_for_video(self):
        """Save the current frame for video creation."""
        # Make sure output directory exists
        if not os.path.exists(self.temp_frame_dir):
            os.makedirs(self.temp_frame_dir)
            
        # Save figure with tight layout first to ensure proper capture of content
        frame_file = f"{self.temp_frame_dir}/frame_{self.frame_count:04d}.png"
        
        # Set DPI to ensure even dimensions (multiple of 2)
        # Standard figure size with dpi=100 makes dimensions divisible by 2
        self.fig.savefig(frame_file, dpi=100, bbox_inches='tight')
        
    def create_video(self):
        """Create a video from saved frames."""
        import subprocess
        import os
        
        # Check if any frames were saved
        if not os.path.exists(self.temp_frame_dir) or len(os.listdir(self.temp_frame_dir)) == 0:
            print(f"No frames found in {self.temp_frame_dir} directory.")
            return
            
        # Check if ffmpeg is available
        try:
            # First check if ffmpeg is installed
            have_ffmpeg = True
            try:
                subprocess.run(['which', 'ffmpeg'], check=True, capture_output=True)
            except subprocess.CalledProcessError:
                have_ffmpeg = check_and_install_ffmpeg()
                if not have_ffmpeg:
                    return
                
            # Get dimensions of first frame to check if we need scaling
            frame_files = sorted([f for f in os.listdir(self.temp_frame_dir) if f.startswith('frame_')])
            if not frame_files:
                print("No frame files found.")
                return
                
            # Use a filter to ensure even dimensions (required for yuv420p)
            cmd = [
                'ffmpeg', '-y',  # Overwrite output file if it exists
                '-framerate', '15',  # Frames per second
                '-i', f'{self.temp_frame_dir}/frame_%04d.png',
                '-c:v', 'libx264',
                '-profile:v', 'high',
                '-crf', '20',
                '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # Ensure even dimensions
                '-pix_fmt', 'yuv420p',
                'tactile_visualization.mp4'
            ]
            
            print("Creating video...")
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("Video saved as 'tactile_visualization.mp4'")
            
        except subprocess.CalledProcessError as e:
            print(f"Failed to create video: {e}")
            if hasattr(e, 'output'):
                print(f"Output: {e.output}")
            if hasattr(e, 'stderr'):
                print(f"Error: {e.stderr}")
                
        except Exception as e:
            print(f"Error creating video: {e}")

    def handle_close(self, evt):
        self.running = False
        print("Plot window closed.")
        
        # Create video before closing if we were recording frames
        if self.record_video and self.frame_count > 10:
            self.create_video()

    def close(self):
        # Create video before closing if we were recording frames
        if self.record_video and self.frame_count > 10:
            self.create_video()
        plt.close(self.fig)


# Add import for subprocess and os
import subprocess
import os

# Add utility functions at the module level
def check_and_install_ffmpeg():
    """Check if ffmpeg is available, and offer to install it if not."""
    import subprocess
    import sys
    
    try:
        # Check if ffmpeg is available
        subprocess.run(['which', 'ffmpeg'], check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        print("\nFFmpeg is not installed, which is required for video creation.")
        
        # Check system type
        if sys.platform.startswith('linux'):
            install_cmd = "sudo apt-get update && sudo apt-get install -y ffmpeg"
            package_manager = "apt"
        elif sys.platform == 'darwin':  # macOS
            install_cmd = "brew install ffmpeg"
            package_manager = "Homebrew"
        else:
            print("Automatic installation not supported on this platform.")
            print("Please install ffmpeg manually.")
            return False
            
        try:
            # Ask user for permission to install
            response = input(f"\nWould you like to install ffmpeg using {package_manager}? (y/n): ")
            if response.lower() == 'y':
                print(f"\nInstalling ffmpeg using: {install_cmd}")
                subprocess.run(install_cmd, shell=True, check=True)
                print("FFmpeg installation completed successfully!")
                return True
            else:
                print("FFmpeg installation skipped. Video creation will not be available.")
                return False
        except Exception as e:
            print(f"Error during ffmpeg installation: {e}")
            return False
    
    return False

# --- Main Execution ---
if __name__ == "__main__":
    try:
       run_tracker_with_force_estimation() # Call the new main function
    except KeyboardInterrupt:
       print("\nSimulation interrupted.")
    finally:
        plt.close('all')