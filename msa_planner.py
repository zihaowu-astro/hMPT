# Tool for optimizing JWST MSA configurations.
# Support grid search and local optimization for MSA pointings including RA, DEC, and PA.

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import differential_evolution, Bounds
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.special import erfc 
from astropy.coordinates import SkyCoord
import astropy.units as u
from tqdm import tqdm

# Constants
V2_REF = 378.563110
V3_REF = -428.402740
PHI = 41.42543
SHUTTER_X, SHUTTER_Y = 0.2679, 0.5294  # arcsec

N_K, N_I, N_J = 4, 365, 171
PAIRS_PER_K = N_I * N_J
NUMS_PER_K = 2 * PAIRS_PER_K

UNCONSTRAINED = 0
ENTIRE_OPEN = 0.035
MIDPOINT = 0.059
CONSTRAINED = 0.072
TIGHTLY_CONSTRAINED = 0.091



class MSAModel:
    """
    Model for JWST MSA (Micro-Shutter Array). 
    Handles shutter operability, source positioning, throughput, and observation status.
    """
    def __init__(self, shutter_mask_file, distortion_file,
                 ra_sources, dec_sources,  flux_sources=None, radius_source=0.06,
                 buffer=ENTIRE_OPEN, flux_threshold=0.5, slit_length=1):
        """
        buffer: margin from shutter edges (arcsec). Options: UNCONSTRAINED, ENTIRE_OPEN, MIDPOINT
        """
        self.shutter_mask = self.load_shutter_operability(shutter_mask_file)
        self.msa_apa = self.load_msa_apa(distortion_file)
        self.spline_models = distortion_spline(self.msa_apa)
        self.v2, self.v3 = read_msa_v2v3(distortion_file)
        self.buffer = buffer

        self.ra_sources = ra_sources
        self.dec_sources = dec_sources
        if flux_sources is None:
            self.flux_sources = np.ones_like(ra_sources)
        else:
            self.flux_sources = flux_sources
        self.source_radius = radius_source
        self.flux_threshold = flux_threshold
        self.slit_length = slit_length

    def load_shutter_operability(self, filename):
        """Load shutter operability flags and build a boolean mask of usable shutters."""
        flags = np.loadtxt(filename, dtype=int)
        usable = np.where(flags==0,1,0).reshape(4,171,365)

        # To consider the impact of requiring 3 open shutters vertically
        kernel = np.zeros((1, 3, 3))
        kernel[0, :, 1]=1
            
        # We're shifting the quadrants to unit offset, so that we can set all of the quad=0 to be bad.
        shutter_mask = np.full([5,171,365],False)
        shutter_mask[1:] = np.where(usable,False,True)
        return shutter_mask
    
    def load_msa_apa(self, filename):
        """Load and rotate MSA distortion data."""
        msa = np.loadtxt(filename)
        msa = msa.reshape([4,171,365,2])
        v23_ref = np.array([V2_REF,V3_REF])
        msa = msa[:,:,:]-v23_ref

        apa_rot = np.pi*(180.0-PHI)/180.0
        msa_apa = np.dot(msa, rotation_matrix(apa_rot))
        return msa_apa
    
    def apply_shutter_mask(self, shutters, usable):
        """Check if shutters are usable given the operability mask."""
        with np.errstate(invalid='ignore'):
            return usable[np.where(shutters[0]<0,0,shutters[0]), np.rint(shutters[1]).astype(int), np.rint(shutters[2]).astype(int)]
        
    def apply_shutter_length(self, shutters, usable, length=3):
        with np.errstate(invalid='ignore'):
            mask = np.full(len(shutters[0]), True) 
            for i in range(-length//2, length//2+1):
                row_indices = np.rint(shutters[1]).astype(int) + i
                col_indices = np.rint(shutters[2]).astype(int)
                quad_indices = np.where(shutters[0]<0, 0, shutters[0])
                
                valid_indices = ((row_indices >= 0) & (row_indices < usable.shape[1]) & 
                            (col_indices >= 0) & (col_indices < usable.shape[2]))
                
                current_mask = np.full(len(shutters[0]), False)
                current_mask[valid_indices] = usable[quad_indices[valid_indices], 
                                                    row_indices[valid_indices], 
                                                    col_indices[valid_indices]]
                
                mask &= current_mask
            return mask

    
    def apply_shutter_centration(self, shutters, buffer):
        """Check whether sources fall near shutter centers within buffer margins."""
        row_limit = 0.5-(buffer/SHUTTER_Y)
        col_limit = 0.5-(buffer/SHUTTER_X)
        return (np.abs(shutters[1]-np.rint(shutters[1]))<row_limit)&(np.abs(shutters[2]-np.rint(shutters[2]))<col_limit)
    
    def estimate_slit_flux(self, shutters, sigma=0.06):
        """Estimate fraction of flux transmitted through a shutter for Gaussian-shaped sources."""
        # If the object is a Gaussian of width sigma, how much light gets through the shutter?
        # First, compute where the objects fall in fractional shutter units
        shutter_loc_x = shutters[1]-np.rint(shutters[1]) # This is the 0..170 direction, rows
        shutter_loc_y = shutters[2]-np.rint(shutters[2]) # This is the 0..364 direction, columns
        # Now convert these to arcseconds
        shutter_loc_x = shutter_loc_x*SHUTTER_Y
        shutter_loc_y = shutter_loc_y*SHUTTER_X
        # The shutter bounds are at +-0.23 vertically, +-0.1 horizontally
        return integrate_gaussian(shutter_loc_x, sigma, -0.23, 0.23)*integrate_gaussian(shutter_loc_y, sigma, -0.10, 0.10)

    def evaluate_pointing(self, ra_msa, dec_msa, ang_v3, theta=90   ):
        """Evaluate shutter placement, centering, and flux throughput for a given pointing.
        Theta is the APT DVA param, and an unknown function of APA. See radec_to_Axy docstring."""
        axy = radec_to_Axy(self.ra_sources, self.dec_sources, ra_msa, dec_msa, ang_v3, theta=theta)
        shutters = find_shutter_from_Axy(axy, self.spline_models)
        available = self.apply_shutter_mask(shutters, self.shutter_mask)
        available &= self.apply_shutter_length(shutters, self.shutter_mask, length=self.slit_length)
        centered = self.apply_shutter_centration(shutters, self.buffer)
        throughput = self.estimate_slit_flux(shutters, sigma=self.source_radius) * np.where(available & centered, 1.0, 0.0)
        return throughput, shutters # should mask the shutters?
    
    def obs_status(self, ra_msa, dec_msa, ang_v3, theta=90):
        """Determine which sources are successfully observed given a pointing.
        Theta is the APT DVA param, and an unknown function of APA. See radec_to_Axy docstring.
        Perhaps for grid search if APA is not changing much, theta can be fixed, or maybe the small corection is not important. 
        theta=90 is no correction."""
        throughput, _ = self.evaluate_pointing(ra_msa, dec_msa, ang_v3, theta=theta)
        flux = throughput * self.flux_sources
        detected = flux >= self.flux_threshold
        return detected

class MSAOptimizer:
    def __init__(self, msa_model, weights=None, objective='number'):
        self.mas_model = msa_model
        if weights is None:
            self.weights = np.ones(len(msa_model.ra_sources))
        else:
            self.weights = weights

        self.objective = objective
        if objective == 'number':
            self._objective_function = self._objective_function_number
        elif objective == 'flux':
            self._objective_function = self._objective_function_flux
        else:
            raise ValueError("Objective must be 'number' or 'flux'")

    def _objective_function_number(self, params):
        """ Objective function to be minimized (negative of number of detected sources)."""
        try:
            detected = self.mas_model.obs_status(*params)
            return -np.sum(detected * self.weights)  # number of successful sources (to be maximized)
        except Exception as e:
            return 1e6
    
    def _objective_function_flux(self, params):
        """ Objective function to be minimized (negative of weighted flux of detected sources)."""
        try:
            throughput, _ = self.mas_model.evaluate_pointing(*params)
            flux = throughput * self.mas_model.flux_sources
            return -np.sum(flux * self.weights)  # weighted flux of successful sources (to be maximized)
        except Exception as e:
            return 1e6

    def grid_search(self, ra0, dec0, pa0, dra=0.05, ddec=0.05, dpa=30, n_steps=(50,50,50),
                    verbose=True):
        """Perform a grid search around a given pointing to find optimal parameters."""
        dra_arr, ddec_arr, dang_arr = np.meshgrid(
            np.linspace(-dra, dra, n_steps[0]),
            np.linspace(-ddec, ddec, n_steps[1]), 
            np.linspace(-dpa, dpa, n_steps[2])
        )
        
        ra_arr = ra0 + dra_arr.flatten()
        dec_arr = dec0 + ddec_arr.flatten()
        ang_arr = pa0 + dang_arr.flatten()
        
        scores = []
        
        for i in tqdm(range(len(ra_arr))):
            success_mask = self.mas_model.obs_status(ra_arr[i], dec_arr[i], ang_arr[i])
            scores.append(np.sum(success_mask * self.weights))

        indices = np.argsort(scores)[::-1]
        scores = np.array(scores)[indices]
        ra_arr, dec_arr, ang_arr = ra_arr[indices], dec_arr[indices], ang_arr[indices]
        if verbose:
            print(f"Top 10 scores from grid search:")
            for i in range(10):
                print(f"  {i+1}: Score={scores[i]}, (RA, Dec, V3_PA)=({ra_arr[i]:.8f}, {dec_arr[i]:.8f}, {ang_arr[i]:.2f})")

        results = Table()
        results['score'] = scores
        results['ra'] = ra_arr
        results['dec'] = dec_arr
        results['pa'] = ang_arr
        return results
    
    def optimize_solutions(self, ra_msa, dec_msa, pa_v3, 
                           dra=0.002, ddec=0.002, dpa=5, maxiter=500,
                           verbose=True):
        """Search neighborhood of a given pointing using differential evolution."""
        x0 = [ra_msa, dec_msa, pa_v3]
        initial_score = -self._objective_function(x0)
        
        bounds = Bounds(
            lb=[x0[0] - dra, x0[1] - ddec, x0[2] - dpa],
            ub=[x0[0] + dra, x0[1] + ddec, x0[2] + dpa])

        result = differential_evolution(self._objective_function, 
                                       bounds=list(zip(bounds.lb, bounds.ub)),
                                       maxiter=maxiter, popsize=15, seed=42)
        
        if result.success or hasattr(result, 'x'):
            optimized_params = result.x
            optimized_score = -result.fun
            if verbose:
                if optimized_score <= initial_score:
                    print("  Optimized score is not better than initial score.")
                else:
                    print(f"  Final/Initial score: {optimized_score:.0f}/{initial_score:.0f}")
                    print(f"  Initial params: (RA, Dec, PA)=({x0[0]:.8f}, {x0[1]:.8f}, {x0[2]:.4f})")
                    print(f"  Final params: (RA, Dec, PA)=({optimized_params[0]:.8f}, {optimized_params[1]:.8f}, {optimized_params[2]:.4f})")
                
            return optimized_score, optimized_params
        else:
            print(f"  Optimization failed: {result.message if hasattr(result, 'message') else 'Unknown error'}")
            return initial_score, np.array(x0)
        
    def optimize_top_solutions(self, grid_search_results, n_top=10, 
                               dra=0.002, ddec=0.002, dpa=5, maxiter=500,
                               verbose=True):
        """Optimize the top N solutions from a grid search."""
        ra_arr, dec_arr, pa_arr = grid_search_results['ra'], grid_search_results['dec'], grid_search_results['pa']
        optimized_scores_all, optimized_params_all = [], []
        
        for i in range(n_top):
            if verbose:
                print(f"Optimizing No.{i+1} best solution:")
            ra_init, dec_init, pa_init = ra_arr[i], dec_arr[i], pa_arr[i]
            optimized_score, optimized_params = self.optimize_solutions(ra_init, dec_init, pa_init,
                                                                        dra=dra, ddec=ddec, dpa=dpa,
                                                                        maxiter=maxiter, verbose=verbose)
            optimized_scores_all.append(optimized_score)
            optimized_params_all.append(optimized_params)
        
        # sort by score
        sorted_indices = np.argsort(optimized_scores_all)[::-1]
        optimized_scores_all = [optimized_scores_all[i] for i in sorted_indices]
        optimized_params_all = [optimized_params_all[i] for i in sorted_indices]


        results = Table()
        results['score'] = [optimized_scores_all[i] for i in range(len(optimized_scores_all))]
        results['ra'] = [optimized_params_all[i][0] for i in range(len(optimized_params_all))]
        results['dec'] = [optimized_params_all[i][1] for i in range(len(optimized_params_all))]
        results['pa'] = [optimized_params_all[i][2] for i in range(len(optimized_params_all))]
        return results

# Helper functions for distortions
def _reshape_one_quadrant(flat_k):
    """Reshape flat distortion data for one quadrant into structured arrays."""
    arr = np.asarray(flat_k).reshape(-1)
    if arr.size != NUMS_PER_K:
        raise ValueError(f"Expected {NUMS_PER_K} numbers per quadrant")
    pairs = arr.reshape(PAIRS_PER_K, 2)
    pairs = pairs.reshape(N_J, N_I, 2)
    pairs = np.transpose(pairs, (1, 0, 2))
    return pairs[:, :, 0], pairs[:, :, 1]

def read_msa_v2v3(path):
    """Read MSA v2,v3 distortion data from a file."""
    raw = np.loadtxt(path)
    raw = raw.reshape(-1, 2)
    if raw.shape[0] < 4 * PAIRS_PER_K:
        raise ValueError("File shorter than expected")
    v2 = np.zeros((N_K, N_I, N_J), float)
    v3 = np.zeros((N_K, N_I, N_J), float)
    idx = 0
    for k in range(N_K):
        block = raw[idx: idx + PAIRS_PER_K, :].reshape(-1)
        idx += PAIRS_PER_K
        v2[k], v3[k] = _reshape_one_quadrant(block)
    return v2, v3

def integrate_gaussian(mean, sigma, low, high):
    low = (low-mean)/sigma
    high = (high-mean)/sigma
    return 0.5*(erfc(low*np.sqrt(0.5))-erfc(high*np.sqrt(0.5)))

def rotation_matrix(r):
    return np.array([np.array([np.cos(r),np.sin(r)]), np.array([-np.sin(r),np.cos(r)])])

def V23_to_Axy(v2,v3):
    """Convert V2,V3 coordinates to aperture coordinates (ax, ay)."""
    v23 = np.array([np.atleast_1d(v2),np.atleast_1d(v3)])
    return np.dot(v23.T, rotation_matrix((180.0-PHI)*np.pi/180.0))

def Axy_to_V23(ax,ay):
    """Convert aperture coordinates (ax, ay) to V2,V3 coordinates."""
    axy = np.array([np.atleast_1d(ax),np.atleast_1d(ay)])
    return np.dot(axy.T, rotation_matrix(-(180.0-PHI)*np.pi/180.0))

def radec_to_Axy(ra, dec, ra_pointing, dec_pointing, pa_v3, theta=0):
    """Map (RA, Dec) deg to (ax, ay) for pointing (ra_pointing, dec_pointing, pa_v3).
    --update: theta is the APT differential velocity aberration param, retrieved from APT as File->Export->xml file->Search for theta.
    Theta is a function of the APA in APT, where some date is implicitly assumed to compute the JWST's velocity vector.
    For full consistency this should probably be applied in the v2v3_to_radec function as well.
    """
    dra = (ra - ra_pointing)*np.pi/180.0
    dec = dec*np.pi/180.0
    dec_ns = dec_pointing*np.pi/180.0
    denom = (np.sin(dec)*np.sin(dec_ns)+np.cos(dec)*np.cos(dec_ns)*np.cos(dra))
    #print(denom)   # This is almost exactly 1
    denom = denom*np.pi/3600.0/180.0
    x = np.cos(dec)*np.sin(dra)/denom    # West to east distance, in arcsec
    y = (np.sin(dec)*np.cos(dec_ns)-np.cos(dec)*np.sin(dec_ns)*np.cos(dra))/denom   # south to north distance

    M_DVA = 1 / (1 - 30/3e5 * np.cos((theta-pa_v3)*np.pi/180)) # theta references to v3pa
    x *= M_DVA
    y *= M_DVA

    v3pa = pa_v3*np.pi/180.0
    v2 = np.cos(v3pa)*x - np.sin(v3pa)*y
    v3 = +np.sin(v3pa)*x + np.cos(v3pa)*y   

    axy = V23_to_Axy(v2,v3)

    return axy

def v2v3_to_radec(v2, v3, ra_p_deg, dec_p_deg, pa_v3_deg):
    """
    Map (v2, v3) arcsec to (RA, Dec) deg for pointing (ra_p, dec_p, pa_v3).
    """

    # Zihao I think the - V2_REF and - V3_REF are not needed here, because v2 and v3 are already relative to the pointing. 
    # Runnning radec_to_Axy -> Axy_to_V23 -> v2v3_to_radec does not return the original RA, Dec.
    dv2 = v2 - V2_REF
    dv3 = v3 - V3_REF
    th = np.deg2rad(pa_v3_deg)

    xr_arcsec = +dv2 * np.cos(th) + dv3 * np.sin(th)
    yr_arcsec = -dv2 * np.sin(th) + dv3 * np.cos(th)

    ARCSEC2RAD = np.deg2rad(1.0 / 3600.0)
    xr = xr_arcsec * ARCSEC2RAD
    yr = yr_arcsec * ARCSEC2RAD

    ra_p = np.deg2rad(ra_p_deg)
    dec_p = np.deg2rad(dec_p_deg)

    denom = np.cos(dec_p) - yr * np.sin(dec_p)
    ra = ra_p + np.arctan2(xr, denom)

    num = np.sin(dec_p) + yr * np.cos(dec_p)
    dec = np.arcsin(num / np.sqrt(1.0 + xr * xr + yr * yr))

    return np.rad2deg(ra), np.rad2deg(dec)

def distortion_spline(msa_apa):
    """Create spline interpolators for MSA distortion data."""
    spline_models = []
    for q in range(4):
        # Extract detector coordinates and corresponding shutter positions
        ax_data = msa_apa[q, :, :, 0].flatten()  # ax coordinates
        ay_data = msa_apa[q, :, :, 1].flatten()  # ay coordinates
        
        # Create corresponding row, col coordinates
        rows, cols = np.meshgrid(np.arange(171), np.arange(365), indexing='ij')
        row_data = rows.flatten()
        col_data = cols.flatten()

        # Create points array for interpolation
        points = np.column_stack([ax_data, ay_data])
        
        # Build interpolators for row and column
        row_interpolator = CloughTocher2DInterpolator(points, row_data)
        col_interpolator = CloughTocher2DInterpolator(points, col_data)

        # Get bounds for this quadrant
        ax_bounds = (ax_data.min(), ax_data.max())
        ay_bounds = (ay_data.min(), ay_data.max())
        
        spline_models.append({
            'row_interp': row_interpolator,
            'col_interp': col_interpolator,
            'ax_bounds': ax_bounds,
            'ay_bounds': ay_bounds
        })

    return spline_models

def find_shutter_from_Axy(axy, distortion_models):
    """Find shutter indices (quadrant, row, col) from aperture coordinates (ax, ay)."""
    axy = axy.reshape(-1, 2)

    row, col = np.full(len(axy), np.nan), np.full(len(axy), np.nan)
    quad = np.zeros(len(axy), dtype=int)
    
    for q in range(4):
        model = distortion_models[q]
        
        # Check bounds
        ax_in_bounds = ((axy[:, 0] >= model['ax_bounds'][0]) & 
                    (axy[:, 0] <= model['ax_bounds'][1]))
        ay_in_bounds = ((axy[:, 1] >= model['ay_bounds'][0]) & 
                    (axy[:, 1] <= model['ay_bounds'][1]))
        in_bounds = ax_in_bounds & ay_in_bounds
        
        if np.any(in_bounds):
            # Direct interpolation
            pr = model['row_interp'](axy[in_bounds])
            pc = model['col_interp'](axy[in_bounds])
            
            # Check validity
            valid = ((pr >= -0.5) & (pr <= 170.5) & 
                    (pc >= -0.5) & (pc <= 364.5) & 
                    ~np.isnan(pr) & ~np.isnan(pc))
            
            # Apply vignetting masks
            if q == 0: 
                vignette_mask = (pr >= 12-0.5) & (pc >= 9-0.5)
            elif q == 1: 
                vignette_mask = (pr <= 170.5-12) & (pc >= 9-0.5)
            elif q == 2: 
                vignette_mask = (pr >= 12-0.5) & (pc <= 365.5-9)
            elif q == 3: 
                vignette_mask = (pr <= 170.5-13) & (pc <= 365.5-6)
            
            valid = valid & vignette_mask
            
            # Update results
            valid_indices = np.where(in_bounds)[0][valid]
            row[valid_indices] = pr[valid]
            col[valid_indices] = pc[valid]
            quad[valid_indices] = q + 1
    
    return quad, row, col


def show_msa_result(msa_model, ra_msa, dec_msa, pa_msa, dec_width=0.05, verbose=True):
    """Visualize MSA configuration and source placements."""
    ra_out, dec_out = v2v3_to_radec(msa_model.v2, msa_model.v3, 
                                        ra_msa, dec_msa, pa_msa)
    
    ra_sources, dec_sources = msa_model.ra_sources, msa_model.dec_sources
    success_mask = msa_model.obs_status(ra_msa, dec_msa, pa_msa)


    if verbose:
        aperture_pa =  (180 - PHI) + pa_msa
        coord = SkyCoord(ra=ra_msa*u.deg, dec=dec_msa*u.deg, frame='icrs')
        print(f"MSA pointing: RA={ra_msa:.8f}, Dec={dec_msa:.8f}, V3_PA={pa_msa:.6f}")
        print(f'              {coord.to_string("hmsdms", sep=" ", precision=4)}, Aperture PA: {np.mod(aperture_pa, 360):.6f} deg')
        print(f"Number of sources: {len(ra_sources)}")
        print(f"Number of sources in open shutters: {np.sum(success_mask)}")

    plt.figure(figsize=(8,8))
    plt.scatter(ra_out.flatten(), dec_out.flatten(), s=0.005, 
               c=msa_model.shutter_mask[1:].swapaxes(1,2).flatten(), cmap='grey')
    plt.scatter(ra_sources, dec_sources, s=20, c='k', label='candidates')
    plt.scatter(ra_sources[success_mask], dec_sources[success_mask], 
               s=20, c='C2', label='in open shutters')
    
    ra_width = dec_width / np.cos(np.deg2rad(dec_msa))
    plt.xlim(ra_msa + ra_width, ra_msa - ra_width)
    plt.ylim(dec_msa - dec_width, dec_msa + dec_width)
    plt.xlabel('RA (deg)')
    plt.ylabel('Dec (deg)')
    plt.legend()
    
    # Add dispersion direction arrow
    ang_disp = np.deg2rad(pa_msa - (180 + PHI) + 90)
    plt.annotate('Dispersion', 
                xy=(ra_msa - 1.2e-2 * np.sin(ang_disp)/np.cos(np.deg2rad(dec_msa)), 
                    dec_msa - 1.2e-2 * np.cos(ang_disp)), 
                xytext=(ra_msa, dec_msa),
                arrowprops=dict(arrowstyle='->', color='grey', lw=2, mutation_scale=20),
                fontsize=12, color='grey', ha='center', va='center',
                rotation=np.mod(90 + ang_disp*180/np.pi+90, 180)-90)
    
    plt.show()



def make_msa_config(shutters, shutter_mask_file, output_csv):
    """output a csv file representing the MSA configuration that can be imported into APT"""

    flags = np.loadtxt(shutter_mask_file, dtype=int)
    flags = flags.reshape(4,171,365)

    import csv

    quad, row, col = shutters 
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)
    quad = np.asarray(quad).astype(int)

    arr = np.zeros((4,171,365))
    for q, r, c in zip(quad, row, col):
        arr[q-1,r,c] = 1
    
    with open(output_csv, "w", newline='') as f:
        writer = csv.writer(f)

        for ir in range(365):
            line = np.full(171*2, '1')
            for jr in range(171):
                q = 3 
                i_slitmap = arr[q, jr, ir]
                i_msa_mask = flags[q, jr, ir]
                if i_msa_mask == 0: # failed closed
                    if i_slitmap == 1:
                        print(f"Warning: shutter at Q{q+1} R{jr} C{ir} is marked open in input but failed closed in operability mask.")
                    shut_stat = 'x'
                elif i_msa_mask == 1: # failed open
                    shut_stat = 's'
                elif i_msa_mask >= 2: # operable
                    if i_slitmap == 1:
                        shut_stat = '0'  # 0 means open for some reason
                    else:
                        shut_stat = '1'

                line[jr] = shut_stat

            for jr in range(0, 171):
                q = 0  
                i_slitmap = arr[q, jr, ir]
                i_msa_mask = flags[q, jr, ir]
                if i_msa_mask == 0: # failed closed
                    if i_slitmap == 1:
                        print(f"Warning: shutter at Q{q+1} R{jr} C{ir} is marked open in input but failed closed in operability mask.")
                    shut_stat = 'x'
                elif i_msa_mask == 1: # failed open
                    shut_stat = 's'
                elif i_msa_mask >= 2: # operable
                    if i_slitmap == 1:
                        shut_stat = '0'  # 0 means open for some reason
                    else:
                        shut_stat = '1'

                line[jr+171] = shut_stat

            writer.writerow(line)

        for ir in range(365):
            line = np.full(171*2, '1')
            for jr in range(171):
                q = 1 
                i_slitmap = arr[q, jr, ir]
                i_msa_mask = flags[q, jr, ir]
                if i_msa_mask == 0: # failed closed
                    if i_slitmap == 1:
                        print(f"Warning: shutter at Q{q+1} R{jr} C{ir} is marked open in input but failed closed in operability mask.")
                    shut_stat = 'x'
                elif i_msa_mask == 1: # failed open
                    shut_stat = 's'
                elif i_msa_mask >= 2: # operable
                    if i_slitmap == 1:
                        shut_stat = '0'  # 0 means open for some reason
                    else:
                        shut_stat = '1'

                line[jr] = shut_stat

            for jr in range(0, 171): 
                q = 2 
                i_slitmap = arr[q, jr, ir]
                i_msa_mask = flags[q, jr, ir]
                if i_msa_mask == 0: # failed closed
                    if i_slitmap == 1:
                        print(f"Warning: shutter at Q{q+1} R{jr} C{ir} is marked open in input but failed closed in operability mask.")
                    shut_stat = 'x'
                elif i_msa_mask == 1: # failed open
                    shut_stat = 's'
                elif i_msa_mask >= 2: # operable
                    if i_slitmap == 1:
                        shut_stat = '0'  # 0 means open for some reason
                    else:
                        shut_stat = '1'

                line[jr+171] = shut_stat

            writer.writerow(line)

    return

def check_model(mpt_output_file, pointing, spline_models, theta, plot=True):
    """Compare MPT output with model predictions and plot discrepancies.
    mpt_output_file is retrieved from APT as File->Export->MSA target info.
    theta is the APT differential velocity aberration param, retrieved from APT as File->Export->xml file->Search for theta.
    """

    import pandas as pd
    import seaborn as sns

    mpt_output = pd.read_csv(mpt_output_file)
    mpt_cols = mpt_output[' Column (Disp)']
    mpt_cols_offset = mpt_output[' Offset (x)']
    mpt_rows = mpt_output[' Row (Spat)']
    mpt_rows_offset = mpt_output[' Offset (y)']
    mpt_quads = mpt_output[' Quadrant']
    mpt_ra = mpt_output[' Source RA (Degrees)']
    mpt_dec = mpt_output[' Source Dec (Degrees)']
    hmpt_axy = radec_to_Axy(mpt_ra, mpt_dec, pointing[0], pointing[1], pointing[2], theta)
    hmpt_quads, hmpt_rows, hmpt_columns = find_shutter_from_Axy(hmpt_axy, spline_models)

    quad_mask = ~(hmpt_quads < 1)

    print(f'{np.sum(1-quad_mask)} sources were marked out of bounds (quad<1)')
    print(f'{np.sum((hmpt_quads[quad_mask]-1)==mpt_quads[quad_mask])} quadrants are different between hMPT and MPT')


    mpt_rows_float = np.array(mpt_rows-1) + np.array(mpt_rows_offset)
    mpt_cols_float = np.array(mpt_cols-1) + np.array(mpt_cols_offset)

    hmpt_rows = np.array(hmpt_rows)
    hmpt_cols = np.array(hmpt_columns)

    delta_rows = hmpt_rows[quad_mask]+0.5 - mpt_rows_float[quad_mask]
    delta_cols = hmpt_cols[quad_mask]+0.5 - mpt_cols_float[quad_mask]

    print(f'{np.sum(np.abs(delta_rows)>0.5)} sources were off by >0.5 in row')
    print(f'{np.sum(np.abs(delta_cols)>0.5)} sources were off by >0.5 in col')

    if plot:
        df = pd.DataFrame({
            'delta_row': delta_rows,
            'delta_col': delta_cols,
            'quadrant': mpt_quads[quad_mask],
        })

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # 1 row, 2 columns

        # Boxplot: delta_row vs quadrant
        sns.boxplot(x='quadrant', y='delta_row', data=df, ax=axes[0])
        axes[0].set_title('Boxplot of row error (hMPT - MPT) by Quadrant')
        axes[0].set_xlabel('Quadrant')
        axes[0].set_ylabel('delta_row')
        axes[0].set_ylim(-0.1,0.1)


        # Boxplot: delta_col vs quadrant
        sns.boxplot(x='quadrant', y='delta_col', data=df, ax=axes[1])
        axes[1].set_title('Boxplot of col error (hMPT - MPT) by Quadrant')
        axes[1].set_xlabel('Quadrant')
        axes[1].set_ylabel('delta_col')
        axes[1].set_ylim(-0.1,0.1)

        plt.tight_layout()
        plt.show()
    else:
        quads = mpt_quads[quad_mask]
        print(f'Median row error by quad: {np.median(delta_rows[quads==1]):.4f}, {np.median(delta_rows[quads==2]):.4f}, {np.median(delta_rows[quads==3]):.4f}, {np.median(delta_rows[quads==4]):.4f}')
        print(f'Median col error by quad: {np.median(delta_cols[quads==1]):.4f}, {np.median(delta_cols[quads==2]):.4f}, {np.median(delta_cols[quads==3]):.4f}, {np.median(delta_cols[quads==4]):.4f}')
    
    return 

def create_padded_catalog(catalog, pointing):
    """Takes a Table or pd DF catalog and creates a padded catalog to circumvent EoE"""

    import pandas as pd

    ra_center, dec_center = pointing[0], pointing[1]

    ra = catalog['RA'].to_numpy().copy()
    dec = catalog['DEC'].to_numpy().copy()
    
    # How many are on each side of center?
    n_left  = np.sum(ra <  ra_center)
    n_right = np.sum(ra >  ra_center)
    n_below = np.sum(dec < dec_center)
    n_above = np.sum(dec > dec_center)
    
    delta_ra = n_right - n_left
    delta_dec = n_above - n_below

    # Gather "fakes" in arrays
    fake_rows = []

    # Add fake center
    fake_rows.append([-1, ra_center, dec_center, 0.0, 3])

    # Pad RA sides
    if delta_ra > 0:  # need left padding
        n = delta_ra
        ids = -np.arange(2, 2 + n)
        fake_rows.extend(np.column_stack([ids,
                                          np.full(n, ra_center - 1),
                                          np.full(n, dec_center),
                                          np.zeros(n),
                                          np.full(n, 3)
                                         ]))
    elif delta_ra < 0:  # need right padding
        n = -delta_ra
        ids = -np.arange(2, 2 + n)
        fake_rows.extend(np.column_stack([ids,
                                          np.full(n, ra_center + 1),
                                          np.full(n, dec_center),
                                          np.zeros(n),
                                          np.full(n, 3)
                                         ]))
    next_id = 2 + abs(delta_ra)

    # Pad DEC sides
    if delta_dec > 0:  # need below padding
        n = delta_dec
        ids = -np.arange(next_id, next_id + n)
        fake_rows.extend(np.column_stack([ids,
                                          np.full(n, ra_center),
                                          np.full(n, dec_center - 1),
                                          np.zeros(n),
                                          np.full(n, 3)
                                         ]))
    elif delta_dec < 0:  # need above padding
        n = -delta_dec
        ids = -np.arange(next_id, next_id + n)
        fake_rows.extend(np.column_stack([ids,
                                          np.full(n, ra_center),
                                          np.full(n, dec_center + 1),
                                          np.zeros(n),
                                          np.full(n, 3)
                                         ]))

    # Assemble real data as a numpy array
    catalog_data = catalog[['ID','RA','DEC','Redshift','Number']].values

    # Stack everything together
    all_rows = np.vstack([catalog_data] + fake_rows)

    # Re-create DataFrame
    newcat = pd.DataFrame(
        all_rows, columns=['ID','RA','DEC','Redshift','Number'])

    # Ensure all columns the same dtype as original; cast ID/Number to int
    newcat['ID'] = newcat['ID'].astype(int)
    newcat['Number'] = newcat['Number'].astype(int)

    return newcat




if __name__ == "__main__":
    # Example usage with random sources
    N = 50
    np.random.seed(42)
    ra, dec = 53 + np.random.rand(N)*0.08, -27 + np.random.rand(N)*0.08
    msa_model = MSAModel('esa_msa_map_APT_2025.5.3.dat', './msa_v2v3.dat',
                        ra_sources=ra, dec_sources=dec, flux_sources=None,
                        radius_source=0.06, flux_threshold=0.5, buffer=UNCONSTRAINED)

    # Optimize
    optimizer = MSAOptimizer(msa_model)
    results = optimizer.grid_search(53, -27, 30, 
                                    dra = 0.05, ddec = 0.05, dpa = 30,
                                    n_steps=(50,50,50))
    print(f'Maximum objects in open shutters: {np.max(results["score"])}/{len(ra)}')

    optimized_results = optimizer.optimize_top_solutions(results, n_top=10, maxiter=300)
    print(f'Maximum objects in open shutters: {optimized_results["score"][0]}/{len(ra)}')

    show_msa_result(msa_model, optimized_results['ra'][0], optimized_results['dec'][0], optimized_results['pa'][0])
