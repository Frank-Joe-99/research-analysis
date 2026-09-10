import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import zoom
from scipy.stats import binned_statistic_2d


# Plot style
mpl.rcParams.update({
    'font.family': 'Arial',
    'font.sans-serif': ['Arial'],
    'mathtext.fontset': 'custom',
    'mathtext.default': 'regular',
    'svg.fonttype': 'none',
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
})

LABELS = [
    r'$\beta$ content (%)', 'Helix content (%)', 'H-bond number', 'Rg (nm)',
    'SASA (nm$^2$)', 'Hydrophobic SASA (nm$^2$)',
]
FILE_SUFFIXES = ['-beta', '-helix', '-hbond', '-rg', '-sasa', '-hydrosasa', '-']
COLOR_STOPS = [
    (0.00, '#7D1415'), (0.16, '#EE2024'), (0.32, '#FF8C00'),
    (0.48, '#FFFF00'), (0.64, '#32CD32'), (0.78, '#00FFFF'),
    (0.85, '#0080FF'), (0.92, '#253494'), (0.93, '#FFFFFF'),
    (1.00, '#FFFFFF'),
]


def array_deall(array_mat, temperature=310, pseudocount=False):
    """Convert a normalized 2D population matrix to a free-energy surface."""
    gas_constant = 0.0019863  # kcal / (mol K)

    if pseudocount is False or pseudocount is None:
        positive_values = array_mat[array_mat > 0]
        if positive_values.size == 0:
            raise ValueError('No populated bin is available for FEL calculation.')
        pseudocount = positive_values.min()

    pseudocount = float(pseudocount)
    if pseudocount <= 0:
        raise ValueError('The pseudocount (-d) must be greater than zero.')

    print(pseudocount)
    energy = -gas_constant * temperature * (
        np.log(array_mat + pseudocount) - np.log(pseudocount)
    )
    return energy.T, round(energy.min())


def data_get_xy(file_path):
    """Load two-column x/y data, ignoring Gromacs-style comment lines."""
    data = np.loadtxt(file_path, comments=['#', '@'], ndmin=2)
    if data.ndim != 2 or data.shape[1] != 2:
        raise ValueError('The input file must contain exactly two numeric columns: x and y.')
    return data


# The following legacy helpers were unused by the plotting workflow and are
# intentionally disabled to keep this script focused.
# def data_get(file_path1, file_path2):
#     ...
#
def data_block_average(data, block_size=2):
    length, col_num = data.shape                                                                                     
    n_blocks = length // block_size                                                                                  
    result = data[:n_blocks * block_size].reshape(n_blocks, block_size, col_num).mean(axis=1)                        
    return result


def _get_axis_bins(values, axis_name):
    """Ask for a bin width and create bin edges for one coordinate axis."""
    data_min = float(np.min(values))
    data_max = float(np.max(values))
    axis_min = data_min * 0.9
    axis_max = data_max * 1.1

    print(f'{data_min} {data_max} {data_max - data_min}')
    print(f'auto {axis_name} range: {axis_min} {axis_max}')
    bin_width = float(input(f'{axis_name} bin width: '))
    if bin_width <= 0:
        raise ValueError(f'{axis_name} bin width must be greater than zero.')

    # Keep the original bin-count convention: one additional bin is retained.
    bin_count = int((axis_max - axis_min) / bin_width) + 1
    edges = axis_min + np.arange(bin_count + 1) * bin_width
    return axis_min, axis_max, bin_width, bin_count, edges


def array_get(array):
    """Bin two-dimensional data and retain source row numbers for each bin.

    The function's inputs and return values match the original implementation.
    """
    x_data, y_data = np.asarray(array).T
    x_min, x_max, x_bin_width, x_bins, x_edges = _get_axis_bins(x_data, 'x')
    y_min, y_max, y_bin_width, y_bins, y_edges = _get_axis_bins(y_data, 'y')
    row_numbers = np.arange(1, len(array) + 1)

    binned = binned_statistic_2d(
        x_data,
        y_data,
        row_numbers,
        statistic='count',
        bins=[x_edges, y_edges],
        expand_binnumbers=True,
    )
    array_mat = binned.statistic

    # Preserve the original return structure: each list starts with 0.0.
    list_t_info = np.zeros((x_bins, y_bins, 1)).tolist()
    x_indices, y_indices = binned.binnumber
    valid = (
        (x_indices >= 1) & (x_indices <= x_bins) &
        (y_indices >= 1) & (y_indices <= y_bins)
    )
    for x_index, y_index, row_number in zip(
        x_indices[valid] - 1, y_indices[valid] - 1, row_numbers[valid]
    ):
        list_t_info[x_index][y_index].append(row_number)

    return (
        array_mat, list_t_info, x_min, x_max, x_bin_width,
        y_min, y_max, y_bin_width,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description='Calculate and plot a two-dimensional Free Energy Landscape.'
    )
    parser.add_argument('-xy', required=True, help='Input data file with x and y columns.')
    parser.add_argument('-t', type=float, default=310, help='Temperature in K (default: 310).')
    parser.add_argument('-c', type=float, default=None, help='Minimum colorbar value.')
    parser.add_argument('-d', type=float, default=None, help='Pseudocount for comparable FELs.')
    parser.add_argument('-e', action='store_true', help='Save the raw FEL matrix.')
    parser.add_argument('-i', default='bilinear', help='Deprecated and unused; retained for compatibility.')
    parser.add_argument('-n', type=int, default=16, help='Number of color levels (default: 16).')
    parser.add_argument('-p', default='svg', help='Output format, e.g. svg, png, jpg, or eps.')
    parser.add_argument('-s', action='store_true', help='Interactively print source row numbers in a bin.')
    return parser.parse_args()


def choose_label(axis_name):
    print(
        'please choose the x & y label:\n'
        '1. beta Structure \n2. helix Structure \n3. hbond number \n'
        '4. Rg \n5. SASA \n6. hydrophobic SASA\n7. else\n'
    )
    choice = int(input(f'{axis_name} label number: '))
    if 1 <= choice <= len(LABELS):
        return LABELS[choice - 1], FILE_SUFFIXES[choice - 1]
    return input(f'{axis_name} label: '), FILE_SUFFIXES[-1]


def save_energy_data(data):
    np.savetxt('./FEL-energy.dat', data, delimiter='\t')
    print('energy data is saved as: FEL-energy.dat')


def plot_fel(data_fin, value_min, x_min, x_max, y_min, y_max, levels, xlabel, ylabel):
    """Create the FEL contour plot and its continuous colorbar."""
    data_interp = zoom(data_fin, (8, 6), order=1, mode='grid-constant', grid_mode=True)
    ny, nx = data_interp.shape
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, nx),
        np.linspace(y_min, y_max, ny),
    )

    cmap = colors.LinearSegmentedColormap.from_list('fel_discrete', COLOR_STOPS, levels)
    energy_levels = np.linspace(value_min, 0.0, levels + 1)
    norm = colors.BoundaryNorm(energy_levels, cmap.N)

    fig, ax = plt.subplots(figsize=(1.8, 1.8))
    ax.tick_params(direction='in', pad=3, length=2, which='both',
                   bottom=True, top=True, left=True, right=True)
    contour = ax.contourf(x_grid, y_grid, data_interp, levels=energy_levels,
                          cmap=cmap, norm=norm, antialiased=False)

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(AutoLocator())
        axis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis='both', which='major', labelsize=8)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=0.05, pad=0.05)
    continuous_cmap = colors.LinearSegmentedColormap.from_list(
        'fel_continuous', COLOR_STOPS, 256
    )
    colorbar_mappable = mpl.cm.ScalarMappable(
        norm=colors.Normalize(vmin=value_min, vmax=0.0), cmap=continuous_cmap
    )
    colorbar_mappable.set_array([])
    cbar = fig.colorbar(colorbar_mappable, cax=cax)
    cbar.ax.yaxis.set_major_locator(AutoLocator())
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    cbar.ax.tick_params(which='major', labelsize=7, pad=3, length=2)
    cbar.ax.tick_params(which='minor', length=1)
    cbar.ax.set_title('(kcal/mol)', fontsize=7)

    return fig, contour


def print_bin_time_info(list_t_info, x_min, y_min, x_bin_width, y_bin_width):
    """Interactively show source-row numbers for selected bins."""
    while True:
        x_position = float(input('x position: '))
        y_position = float(input('y position: '))
        x_index = int((x_position - x_min) // x_bin_width)
        y_index = int((y_position - y_min) // y_bin_width)
        try:
            time_info = list_t_info[x_index][y_index]
        except IndexError:
            print('The selected position is outside the bin range.')
            continue

        print('there are ', len(time_info), ' points in position (',
              x_position, ',', y_position, ')')
        print(time_info[1:21])
        if input('Continue print different position? [y/n]: ').lower() == 'n':
            break


def main():
    args = parse_args()
    if args.n < 1:
        raise ValueError('-n must be at least 1.')

    data_array = data_get_xy(args.xy)
    (array_mat, list_t_info, x_min, x_max, x_bin_width,
     y_min, y_max, y_bin_width) = array_get(data_array)

    if array_mat.sum() == 0:
        raise ValueError('No input data falls within the selected bin ranges.')

    data_fin, value_min = array_deall(array_mat / array_mat.sum(), args.t, args.d)
    if args.c is not None:
        value_min = args.c
    if value_min >= 0:
        raise ValueError('The colorbar minimum (-c) must be less than 0.')

    if args.e:
        save_energy_data(data_fin)

    xlabel, x_suffix = choose_label('X')
    ylabel, y_suffix = choose_label('Y')
    fig, _ = plot_fel(data_fin, value_min, x_min, x_max, y_min, y_max,
                      args.n, xlabel, ylabel)

    output_file = f'FEL-plot{x_suffix}{y_suffix}.{args.p}'
    fig.savefig(output_file, bbox_inches='tight')
    plt.close(fig)
    print(f'plot is saved as: {output_file}')

    if args.s:
        print_bin_time_info(list_t_info, x_min, y_min, x_bin_width, y_bin_width)


if __name__ == '__main__':
    main()
