"""Droplet coordinates, volumes and protein mass concentrations.

Lengths are nm, bead masses are Da, volumes are nm**3 and densities are
mg/mL. Convert BOTH MDAnalysis XTC positions and box lengths from Angstrom
to nm before calling. Arrays are not modified; calculations use float64.

The three physical analyses are density_core (radial plateau), density_gibbs
(equimolar reference volume) and density_envelope (density isosurface and
closed cavities). volume_cal retains the simpler occupied-bin descriptor.
Only orthorhombic periodic boxes are supported.

References: Willard & Chandler, J. Phys. Chem. B 114, 1954 (2010),
doi:10.1021/jp909219k; Li & Bourg, Atmos. Chem. Phys. 23, 2525 (2023),
doi:10.5194/acp-23-2525-2023, section 2.3.
"""

from collections import deque
import math
import warnings

import numpy as np
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree


# 1 Da / nm**3 = 1.660539... mg/mL (not 1660.539...).
DA_NM3_TO_MG_ML = 1.66053906892


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _positions(values, allow_empty=False):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("Positions must have shape (number_of_beads, 3).")
    if not allow_empty and len(values) == 0:
        raise ValueError("At least one bead is required.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Positions must be finite.")
    return values


def _vector(values, name):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain three finite values.")
    return values


def _box(values):
    values = _vector(values, "box_size")
    if np.any(values <= 0):
        raise ValueError("box_size must contain three positive orthorhombic lengths.")
    return values


def _masses(values, count):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 0:
        values = np.full(count, float(values))
    if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("masses must be a positive bead mass or one positive mass per bead, in Da.")
    return values


def extract_position_array(position_array_all, chain_id: list[int], chain_num, chain_length):
    """Copy selected, equal-length chains in the requested order.

    chain_id is ZERO based, as in cluster_analysis.max_cluster. The input
    must be ordered as contiguous chains, with shape (chain_num*chain_length, 3).
    IDs refer to this input array, not necessarily the entire Universe if a
    subgroup was used for clustering. Duplicate IDs are rejected to avoid
    double-counting mass. Empty selection returns shape (0, 3).
    """
    positions = _positions(position_array_all, allow_empty=True)
    chain_num = _integer(chain_num, "chain_num", minimum=0)
    chain_length = _integer(chain_length, "chain_length")
    if len(positions) != chain_num * chain_length:
        raise ValueError("Position count does not equal chain_num * chain_length.")
    ids = np.asarray(chain_id)
    if ids.ndim != 1:
        raise ValueError("chain_id must be a one-dimensional sequence of indices.")
    if ids.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if ids.dtype.kind not in "iu":
        raise ValueError("chain_id must contain integer indices, not floats or booleans.")
    if np.any(ids < 0) or np.any(ids >= chain_num):
        raise ValueError("chain_id is out of range for this coordinate array.")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("chain_id must not contain duplicates.")
    return positions.reshape(chain_num, chain_length, 3)[ids].reshape(-1, 3).copy()


def treat_pbc(position_array, box_size, *, chain_length=None, cutoff=0.8):
    """Reassemble one finite connected droplet, then center its geometry.

    Uses a periodic neighbor graph and minimum-image displacements, not an
    arithmetic mean of wrapped positions. cutoff is the contact search radius
    (nm), NOT the HPS bond length (0.38 nm). If chain_length is provided,
    consecutive beads in each equal-length chain are also connected explicitly.
    Neighbor queries are streamed; no N-by-N distance array is constructed.

    Disconnected selections and periodic winding/percolation raise ValueError:
    their image choices cannot define a unique finite droplet from this graph.
    Returned rows preserve their order and their mean is box_size/2. The result
    is NOT wrapped again: a long asymmetric tail can lie outside the box after
    centering, and wrapping it would split the droplet again.
    """
    positions = _positions(position_array)
    box = _box(box_size)
    cutoff = _positive(cutoff, "cutoff")
    if cutoff >= box.min() / 2:
        raise ValueError("cutoff must be smaller than half the shortest box length.")
    count = len(positions)
    if chain_length is not None:
        chain_length = _integer(chain_length, "chain_length")
        if count % chain_length:
            raise ValueError("Position count must be divisible by chain_length.")

    wrapped = np.mod(positions, box)
    tree = cKDTree(wrapped, boxsize=box)
    shifts = np.zeros((count, 3), dtype=np.int64)
    visited = np.zeros(count, dtype=bool)
    visited[0] = True
    queue = deque([0])
    while queue:
        bead = queue.popleft()
        neighbors = tree.query_ball_point(wrapped[bead], cutoff)
        if chain_length is not None:
            if bead % chain_length:
                neighbors.append(bead - 1)
            if (bead + 1) % chain_length:
                neighbors.append(bead + 1)
        neighbors = np.asarray(neighbors, dtype=np.intp)
        displacement = wrapped[neighbors] - wrapped[bead]
        fractional = displacement / box
        if np.any(np.isclose(np.abs(fractional), 0.5, rtol=0, atol=1e-10)):
            raise ValueError("A bond/contact is half a box long; its periodic image is ambiguous.")
        proposed = shifts[bead] - np.rint(fractional).astype(np.int64)
        known = visited[neighbors]
        if np.any(shifts[neighbors[known]] != proposed[known]):
            raise ValueError("Selected contact graph winds through the periodic box; not a finite droplet.")
        new = neighbors[~known]
        shifts[new] = proposed[~known]
        visited[new] = True
        queue.extend(np.unique(new).tolist())
    if not np.all(visited):
        raise ValueError(
            "Selected beads are disconnected at this cutoff. Check the cluster selection, "
            "coordinate units, chain_length or contact cutoff."
        )
    unwrapped = wrapped + shifts * box
    if np.any(np.ptp(unwrapped, axis=0) >= box):
        raise ValueError("The reconstructed object spans a box length; a finite droplet is required.")
    return unwrapped + (box / 2 - unwrapped.mean(axis=0))


def volume_cal(position_array, chain_id=None, chain_length=None, bin_size=0.5, *, origin=None) -> float:
    """Return the occupied-bin descriptor N_occupied * bin_size**3, in nm**3.

    With chain_id omitted, input is already the selected droplet. With chain_id
    supplied, input MUST be the all-chain array (chain_length is required).
    Do not re-select global IDs after extract_position_array/treat_pbc.
    This is a scale-dependent descriptor, not the envelope or Gibbs volume.
    origin controls grid translation; it defaults to (0, 0, 0).
    """
    positions = _positions(position_array, allow_empty=True)
    spacing = _positive(bin_size, "bin_size")
    origin = np.zeros(3) if origin is None else _vector(origin, "origin")
    if chain_id is not None:
        length = _integer(chain_length, "chain_length")
        if len(positions) % length:
            raise ValueError("Position count must be divisible by chain_length.")
        positions = extract_position_array(positions, chain_id, len(positions) // length, length)
    bins = np.floor((positions - origin) / spacing)
    #print(bins.shape, np.unique(bins, axis=0).shape)
    return float(len(np.unique(bins, axis=0)) * spacing**3)


def radial_density_profile(position_array, masses, *, bin_size=0.5, center=None,
                           r_max=None, box_size=None):
    """Mass/complete spherical-shell volume, including empty shells.

    Accepts (N, 3) or equal-size, already-centered frames (T, N, 3). center
    may be one vector or (T, 3); by default each frame uses its geometric mean.
    For a droplet plus dilute phase, supply the droplet center explicitly.
    With box_size, distances use minimum images and r_max cannot exceed half
    the shortest box length: shells beyond that would be truncated by PBC.
    Without box_size, the supplied coordinates must describe an unwrapped
    object, and the region beyond its supplied beads is assumed empty.
    """
    frames = np.asarray(position_array, dtype=np.float64)
    if frames.ndim == 2:
        frames = frames[None, ...]
    if frames.ndim != 3 or frames.shape[0] == 0 or frames.shape[1] == 0 or frames.shape[2] != 3:
        raise ValueError("Positions must have shape (N, 3) or (T, N, 3), with T, N > 0.")
    if not np.all(np.isfinite(frames)):
        raise ValueError("Positions must be finite.")
    masses = _masses(masses, frames.shape[1])
    spacing = _positive(bin_size, "bin_size")
    if center is None:
        centers = frames.mean(axis=1)
    else:
        centers = np.asarray(center, dtype=np.float64)
        if centers.shape == (3,):
            centers = np.broadcast_to(centers, (len(frames), 3))
        if centers.shape != (len(frames), 3) or not np.all(np.isfinite(centers)):
            raise ValueError("center must have shape (3,) or (T, 3).")
    box = None if box_size is None else _box(box_size)
    distances = []
    for frame, frame_center in zip(frames, centers):
        relative = frame - frame_center
        if box is not None:
            relative -= box * np.rint(relative / box)
        distances.append(np.linalg.norm(relative, axis=1))
    if r_max is None:
        r_max = box.min() / 2 if box is not None else max(r.max() for r in distances) + 3 * spacing
    r_max = _positive(r_max, "r_max")
    if box is not None and r_max > box.min() / 2:
        raise ValueError("r_max exceeds half the shortest box length; full spherical shells do not fit.")
    bin_count = int(np.ceil(r_max / spacing))
    if bin_count > 1_000_000:
        raise ValueError("Too many radial bins; increase bin_size.")
    edges = np.arange(bin_count + 1, dtype=float) * spacing
    edges[-1] = r_max
    shell_volumes = 4 * np.pi / 3 * np.diff(edges**3)
    shell_mass = np.zeros(bin_count)
    shell_count = np.zeros(bin_count)
    for radii in distances:
        shell_mass += np.histogram(radii, bins=edges, weights=masses)[0]
        shell_count += np.histogram(radii, bins=edges)[0]
    shell_mass /= len(frames)
    shell_count /= len(frames)
    # Volume-weighted mean radius of each shell, rather than its lower edge.
    shell_radii = 0.75 * np.diff(edges**4) / np.diff(edges**3)
    return {
        "radii_nm": shell_radii, "edges_nm": edges,
        "shell_volumes_nm3": shell_volumes, "shell_mass_da": shell_mass,
        "shell_bead_count": shell_count,
        "density_mg_ml": DA_NM3_TO_MG_ML * shell_mass / shell_volumes,
        "frame_count": len(frames), "centers_nm": np.array(centers, copy=True),
    }


def density_core(position_array, masses, *, bin_size=0.5, center=None,
                 r_max=None, box_size=None, core_radius=None, rho_dilute=None):
    """Estimate the dense-phase plateau in mg/mL, with a radial profile.

    By default fits rho(r) = rho_out + (rho_in-rho_out)*
    (1-tanh((r-r0)/w))/2, integrating this model over each shell. rho_dilute
    may fix the outer plateau (use 0 for selected-cluster-only coordinates).
    A resolved inner plateau and an outer tail are required; failures raise
    ValueError instead of returning an arbitrary fitted density.

    Alternatively provide core_radius from a visually verified plateau.
    This directly returns the mean bead mass inside that sphere / sphere
    volume, including empty space, without fitting. It does not claim that
    the sphere is a plateau. Both modes assume a roughly spherical droplet;
    a central cavity or a strongly aspherical droplet needs spatial analysis.
    Average equilibrium-frame profiles for physical interpretation.
    """
    profile = radial_density_profile(position_array, masses, bin_size=bin_size,
                                     center=center, r_max=r_max, box_size=box_size)
    if core_radius is not None:
        radius = _positive(core_radius, "core_radius")
        if radius > profile["edges_nm"][-1]:
            raise ValueError("core_radius exceeds the radial analysis region.")
        frames = np.asarray(position_array, dtype=np.float64)
        if frames.ndim == 2:
            frames = frames[None, ...]
        bead_masses = _masses(masses, frames.shape[1])
        core_mass = 0.0
        for frame, frame_center in zip(frames, profile["centers_nm"]):
            relative = frame - frame_center
            if box_size is not None:
                box = _box(box_size)
                relative -= box * np.rint(relative / box)
            core_mass += bead_masses[np.linalg.norm(relative, axis=1) <= radius].sum()
        core_mass /= len(frames)
        volume = 4 * np.pi / 3 * radius**3
        return {
            "density_mg_ml": DA_NM3_TO_MG_ML * core_mass / volume,
            "core_mass_da": float(core_mass), "core_volume_nm3": float(volume),
            "core_radius_nm": radius, "method": "specified_core_sphere", "profile": profile,
        }

    # remove the first three point close to the center of the max cluster, reduce abnormal values
    radii, edges = profile["radii_nm"][3:], profile["edges_nm"][3:]
    density, volumes = profile["density_mg_ml"][3:], profile["shell_volumes_nm3"][3:]
    # print(density) # density profile along the radius
    if len(radii) < 8 or np.count_nonzero(density) < 4:
        raise ValueError("Too few populated radial shells for a two-phase fit; use more frames or core_radius.")
    fixed_outer = None
    if rho_dilute is not None:
        fixed_outer = float(rho_dilute)
        if not np.isfinite(fixed_outer) or fixed_outer < 0:
            raise ValueError("rho_dilute must be finite and nonnegative (mg/mL).")
    # Gauss-Legendre integration prevents interpreting shell averages as point samples.
    nodes, weights = np.polynomial.legendre.leggauss(12)
    sample_r = (edges[1:, None] + edges[:-1, None]) / 2 + np.diff(edges)[:, None] / 2 * nodes
    radial_weights = weights * sample_r**2 * np.diff(edges)[:, None] / 2
    normalization = np.diff(edges**3) / 3

    def model(_r, contrast, midpoint, width, outer=0.0):
        samples = outer + contrast * (1 - np.tanh((sample_r - midpoint) / width)) / 2
        return (samples * radial_weights).sum(axis=1) / normalization

    outer_guess = max(0.0, float(np.average(density[-3:], weights=volumes[-3:])))
    if fixed_outer is not None:
        outer_guess = fixed_outer
    inner_guess = float(np.average(density[:max(2, len(density)//4)],
                                  weights=volumes[:max(2, len(density)//4)]))
    contrast_guess = max(inner_guess - outer_guess, float(density.max()) / 2, 1e-6)
    midpoint_guess = float(radii[np.argmin(np.abs(density - outer_guess - contrast_guess / 2))])
    min_width = bin_size / 20
    guess = [contrast_guess, np.clip(midpoint_guess, bin_size, edges[-1] - bin_size), bin_size]
    lower, upper = [1e-9, bin_size / 2, min_width], [np.inf, edges[-1], edges[-1]]
    if fixed_outer is None:
        guess.append(outer_guess)
        lower.append(0.0)
        upper.append(np.inf)
        fit_model = model
    else:
        def fit_model(r, contrast, midpoint, width):
            return model(r, contrast, midpoint, width, fixed_outer)
    try:
        parameters, _ = curve_fit(fit_model, radii, density, p0=guess, bounds=(lower, upper),
                                  sigma=1 / np.sqrt(volumes), maxfev=20000)
    except (RuntimeError, ValueError) as error:
        raise ValueError("Radial two-phase fit failed; inspect the profile or specify core_radius.") from error
    contrast, midpoint, width = parameters[:3]
    outer = parameters[3] if fixed_outer is None else fixed_outer
    fitted = fit_model(radii, *parameters)
    core_limit = midpoint - np.arctanh(0.9) * width
    outer_limit = midpoint + np.arctanh(0.9) * width
    if np.count_nonzero(radii < core_limit) < 2 or np.count_nonzero(radii > outer_limit) < 2:
        raise ValueError("The fitted profile has no resolved inner/outer plateau; enlarge the sampled region or use spatial analysis.")
    relative_rmse = float(np.sqrt(np.average((density - fitted)**2, weights=volumes)) / contrast)
    if relative_rmse > 0.35:
        warnings.warn("Large radial fit residual; inspect the profile before interpreting a plateau density.",
                      RuntimeWarning, stacklevel=2)
    return {
        "density_mg_ml": float(outer + contrast), "rho_dense_mg_ml": float(outer + contrast),
        "rho_dilute_mg_ml": float(outer), "interface_midpoint_nm": float(midpoint),
        "interface_width_nm": float(width), "interface_5_95_nm": float(2 * np.arctanh(0.9) * width),
        "core_radius_nm": float(core_limit), "relative_fit_rmse": relative_rmse,
        "fitted_density_mg_ml": fitted, "method": "radial_tanh_plateau", "profile": profile,
    }


def density_gibbs(position_array, masses, rho_dense, rho_dilute=0.0, *, reference_volume=None):
    """Equimolar volume from mass balance, with densities in mg/mL.

    V_e = (conversion*M - rho_dilute*V_ref)/(rho_dense-rho_dilute).

    For selected droplet coordinates, use rho_dilute=0 only if the omitted
    dilute background is negligible. For nonzero rho_dilute, reference_volume
    (nm**3) is required, and positions/masses MUST contain ALL beads of the
    analyzed species in that reference domain, including the dilute phase.
    A single droplet is assumed; otherwise V_e is their combined equivalent
    volume. Supply rho_dense independently from a plateau or a matched slab.

    density_mg_ml is the INPUT dense reference density, not an independently
    measured M/V density. liquid_reference_mass_da is rho_dense*V_e/conversion;
    it is not the mass of just those beads geometrically inside R_e.
    """
    positions = _positions(position_array)
    masses = _masses(masses, len(positions))
    dense = _positive(rho_dense, "rho_dense")
    dilute = float(rho_dilute)
    if not np.isfinite(dilute) or dilute < 0 or dilute >= dense:
        raise ValueError("Require 0 <= rho_dilute < rho_dense.")
    if reference_volume is None:
        if dilute != 0:
            raise ValueError("Nonzero rho_dilute requires reference_volume and all beads in that domain.")
        reference = 0.0
    else:
        reference = _positive(reference_volume, "reference_volume")
    total_mass = float(masses.sum())
    excess = DA_NM3_TO_MG_ML * total_mass - dilute * reference
    volume = excess / (dense - dilute)
    if volume <= 0 or (reference_volume is not None and volume > reference * (1 + 1e-12)):
        raise ValueError("Mass and reference densities imply a nonphysical equimolar volume.")
    return {
        "volume_nm3": float(volume), "radius_nm": float(np.cbrt(3 * volume / (4 * np.pi))),
        "density_mg_ml": dense, "rho_dilute_mg_ml": dilute,
        "total_mass_da": total_mass, "excess_mass_da": float(excess / DA_NM3_TO_MG_ML),
        "liquid_reference_mass_da": float(dense * volume / DA_NM3_TO_MG_ML),
        "reference_volume_nm3": reference_volume, "method": "gibbs_mass_balance",
    }


def density_envelope(position_array, masses, *, rho_dense=None, rho_dilute=0.0,
                     density_threshold=None, bin_size=0.5, smoothing_sigma=1.0,
                     max_grid_cells=8_000_000, return_grid=False):
    """Gaussian-density envelope, closed voids and spatially matched densities.

    Uses a local, padded grid, not the complete simulation box. bin_size is
    numerical spacing; smoothing_sigma is the Gaussian STANDARD DEVIATION in
    nm. Fix sigma and refine spacing for convergence. Threshold is in mg/mL,
    supplied explicitly or (rho_dense+rho_dilute)/2. No density is assumed.

    The largest 6-connected high-density component defines the droplet.
    Exterior low-density space is flood-filled with 26-connectivity; enclosed
    low-density regions are voids. Open channels remain exterior. Smaller
    dense components enclosed by this envelope are included as material.
    Volumes count voxels; raw bead-center membership in the same masks defines
    masses. Thus mass outside the envelope (including dangling tails) is not
    divided by the dense region's volume. At this scale a 'void' means low
    protein density, not necessarily a physically empty/vacuum cavity.

    Coordinates must already be unwrapped, e.g. using treat_pbc. The grid
    has open boundaries even if the original simulation was periodic.
    return_grid adds masks, field and origin for visualization/QA.
    """
    positions = _positions(position_array)
    masses = _masses(masses, len(positions))
    spacing = _positive(bin_size, "bin_size")
    sigma = _positive(smoothing_sigma, "smoothing_sigma")
    max_grid_cells = _integer(max_grid_cells, "max_grid_cells")
    dilute = float(rho_dilute)
    if not np.isfinite(dilute) or dilute < 0:
        raise ValueError("rho_dilute must be finite and nonnegative.")
    if density_threshold is None:
        if rho_dense is None:
            raise ValueError("Supply rho_dense or density_threshold in mg/mL.")
        dense = _positive(rho_dense, "rho_dense")
        if dense <= dilute:
            raise ValueError("rho_dense must exceed rho_dilute.")
        density_threshold = (dense + dilute) / 2
    threshold = _positive(density_threshold, "density_threshold")
    if threshold <= dilute:
        raise ValueError("density_threshold must exceed the dilute reference density.")
    if spacing > sigma / 2:
        warnings.warn("For grid convergence, consider bin_size <= smoothing_sigma/2.",
                      RuntimeWarning, stacklevel=2)
    padding = int(np.ceil(4 * sigma / spacing)) + 2
    origin = np.floor(positions.min(axis=0) / spacing) * spacing - padding * spacing
    shape = tuple(int(np.floor((x - start) / spacing)) + padding + 1
                  for x, start in zip(positions.max(axis=0), origin))
    if math.prod(shape) > max_grid_cells:
        raise ValueError(f"Local grid {shape} exceeds max_grid_cells={max_grid_cells}; "
                         "check PBC/units, increase bin_size or explicitly raise the limit.")
    indices = np.floor((positions - origin) / spacing).astype(np.intp)
    field = np.zeros(shape, dtype=np.float64)
    np.add.at(field, tuple(indices.T), masses * DA_NM3_TO_MG_ML / spacing**3)
    ndimage.gaussian_filter(field, sigma=sigma / spacing, mode="constant", cval=0,
                            truncate=4.0, output=field)
    occupied = field >= threshold
    for axis in range(3):
        if np.any(np.take(occupied, [0, -1], axis=axis)):
            raise ValueError("Isosurface reaches the grid boundary; increase padding or threshold.")
    labels, region_count = ndimage.label(occupied, structure=ndimage.generate_binary_structure(3, 1))
    if region_count == 0:
        raise ValueError("No dense region at this threshold/smoothing scale; inspect the density field.")
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    main = labels == int(np.argmax(counts))
    del labels
    outer_mask = ndimage.binary_fill_holes(main, structure=ndimage.generate_binary_structure(3, 3))
    material_mask = occupied & outer_mask
    void_mask = outer_mask & ~occupied
    voxel_volume = spacing**3
    outer_volume = float(np.count_nonzero(outer_mask) * voxel_volume)
    material_volume = float(np.count_nonzero(material_mask) * voxel_volume)
    void_volume = outer_volume - material_volume
    outer_membership = outer_mask[tuple(indices.T)]
    material_membership = material_mask[tuple(indices.T)]
    outer_mass = float(masses[outer_membership].sum())
    material_mass = float(masses[material_membership].sum())
    result = {
        "volume_outer_nm3": outer_volume, "volume_material_nm3": material_volume,
        "volume_void_nm3": void_volume, "void_fraction": void_volume / outer_volume,
        "density_outer_mg_ml": DA_NM3_TO_MG_ML * outer_mass / outer_volume,
        "density_material_mg_ml": DA_NM3_TO_MG_ML * material_mass / material_volume,
        "mass_outer_da": outer_mass, "mass_material_da": material_mass,
        "mass_void_da": outer_mass - material_mass,
        "mass_outside_da": float(masses[~outer_membership].sum()),
        "outer_membership": outer_membership, "material_membership": material_membership,
        "density_threshold_mg_ml": threshold, "smoothing_sigma_nm": sigma,
        "bin_size_nm": spacing, "grid_shape": shape, "dense_region_count": int(region_count),
        "method": "gaussian_density_envelope",
    }
    if return_grid:
        result.update(density_grid_mg_ml=field, grid_origin_nm=origin,
                      outer_mask=outer_mask, material_mask=material_mask, void_mask=void_mask)
    return result
