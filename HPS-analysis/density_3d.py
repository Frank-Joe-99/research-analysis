# 3D density tool in DROPPS package by Yun Zhou @ Fudan
# Development started on Sep 10 2026

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class

from openmm.unit import nanosecond, picosecond
import numpy as np
import tqdm
from bisect import bisect_left, bisect_right

from dropps.fileio.xvg_reader import write_xvg
from dropps.analysis.volume import (
    extract_position_array, treat_pbc, volume_cal,
    density_core, density_gibbs, density_envelope,
)

prog = "density_3d"
desc = '''This program calculate the density profile of a spontaneous LLPS simulation.'''

def getargs_density(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")

    parser.add_argument('-c', '--cluster-input', type=str, required=True,
                        help="Chain indices in the max cluster as a input file. Default: npz file")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")

    parser.add_argument('-o', '--output', type=str, required=True, 
                        help="XVG file to write density profile.")

    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")
    
    args = parser.parse_args(argv)
    return args

# calculate the densit of the droplet
def density_3d(args):
    output_file_name = args.output if args.output.endswith(".xvg") else args.output + ".xvg"

    # load trajectory into memory
    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    dimensions = trajectory.Universe.dimensions
    if not np.allclose(dimensions[3:], 90):
        raise ValueError("当前函数只支持正交盒子")
    box_nm = dimensions[:3].astype(np.float64) / 10.0

    # read the npz file (form the cluster analysis),
    # return data['frame_indices']: list[int]
    # and data['max_clusters'] : list[list[int]]
    try:
        data = np.load(args.cluster_input, allow_pickle=True)
    except:
        print("## An exception occurred when trying to open npz file %s." % args.cluster_input)
        quit()
    frame_idx, max_clusters = data['frame_indices'], data['max_clusters']

    # We treat time for analysis and generate frame for analysis
    start_time = args.start_time * nanosecond if args.start_time is not None else trajectory.time_init()
    end_time = args.end_time * nanosecond if args.end_time is not None else trajectory.time_end()

    if start_time < trajectory.time_init() or end_time > trajectory.time_end():
        print(f"ERROR: Trajectory containing {trajectory.time_init()} - {trajectory.time_end()} "
              + f"while you demanding {start_time} - {end_time}.")
        print(f"YOU MUST BE KIDDING ME.")
        quit()

    start_frame = int((start_time - trajectory.time_init()).value_in_unit(picosecond) / trajectory.time_step().value_in_unit(picosecond))
    end_frame = int((end_time - trajectory.time_init()).value_in_unit(picosecond) / trajectory.time_step().value_in_unit(picosecond))

    index_start = bisect_left(frame_idx, start_frame)
    index_end = bisect_right(frame_idx, end_frame) - 1
    frames = frame_idx[index_start:index_end + 1]
    max_clusters_sets =  max_clusters[index_start:index_end + 1]

    # We now generate atom groups for dense phase determination and density calculations
    # 暂时屏蔽selection，用 max_cluser 中的 chain indices 代表 selection
    # if args.selection_fit is None or args.selection_calculate is None:
    #     trajectory.index.print_all()

    # if args.selection_fit is not None:
    #     print(f"## Will use group {args.selection_fit} for dense phase determination.")
    #     fit_group, fit_group_name = trajectory.getSelection(f"group {args.selection_fit}")
    # else:
    #     fit_group, fit_group_name = trajectory.getSelection_interactive("group for dense phase determination")
    
    # if args.selection_calculate is not None:
    #     print(f"## Will use groups {','.join([f"{i}" for i in args.selection_calculate])} for density calculations.")
    #     density_groups = [trajectory.getSelection(f"group {gid}")[0] for gid in args.selection_calculate]
    #     density_groups_names = [f"group{i}" for i in args.selection_calculate]
    # else:
    #     density_groups, density_groups_names = trajectory.getSelection_interactive_multiple("groups for density calculation")


    # Initialize the main density profile array (time, density_1, density_2, density_3)
    # time = frame_index * args.time_unit
    density_profiles = []

    universe = trajectory.Universe
    all_masses = universe.atoms.masses # numpy.ndarray
    chain_num = trajectory.num_chains()
    chain_length = len(universe.atoms) // chain_num

    for index in tqdm.tqdm(range(len(frames)), desc='# Calculating'):
        # start_time_ptr = time.perf_counter()
        # Load selected frames into MDAnalysis
        temp_frame = int(frames[index])
        temp_cluster = max_clusters_sets[index] # chain index list of the max_cluster

        temp_traj = universe.trajectory[temp_frame]
        temp_time = temp_traj.time / 1000.0 # ns
        temp_positions_nm = temp_traj.positions.astype(np.float64) / 10.0

        cluster_masses = all_masses.reshape(chain_num, chain_length)[temp_cluster].reshape(-1) # mass array of the max_cluster
        cluster_positions = extract_position_array(temp_positions_nm, temp_cluster, chain_num, chain_length,) # position array of the max_cluster

        # print(len(temp_cluster), temp_frame)
        try:
            # treat pbc and get the new position array: centered
            centered = treat_pbc(cluster_positions, box_nm, chain_length=chain_length, cutoff=0.8,)

            # 原始占据体积描述符
            v_occupied = volume_cal(centered, bin_size=1.2)
            rho_occupied = cluster_masses.sum() * 1.66053906892 / v_occupied
            # print(f'voloum: {v_occupied}, rho_occupied: {rho_occupied}')

            # 球形核心平台密度（沿着axis direction, using tanh function fit）
            # core = density_core(centered, cluster_masses, center=box_nm/2, bin_size=0.2, rho_dilute=0.0)
            # rho_in = core["density_mg_ml"]
            # r0 = core["interface_midpoint_nm"]
            # width = core["interface_width_nm"]
            # print(rho_in, r0, width)

            # Gibbs 等摩尔体积
            # gibbs = density_gibbs(centered, cluster_masses, rho_dense=rho_in, )
            # volume_e = gibbs["volume_nm3"]
            # radius_e = gibbs["radius_nm"]
            # density_e = gibbs["density_mg_ml"]
            # print(density_e, radius_e, volume_e)

            # 外包络、密相区域和空腔
            geometry = density_envelope(centered, cluster_masses, rho_dense=rho_occupied, rho_dilute=0.0,
                                        bin_size=0.5, smoothing_sigma=1.0, return_grid=False,)
            # print(geometry["density_outer_mg_ml"], geometry["density_material_mg_ml"])

            # end_time_ptr = time.perf_counter()
            # print(end_time_ptr - start_time_ptr, 'second per loop calculation')

            result = [temp_time, #rho_in, 
                    geometry["density_outer_mg_ml"], 
                    geometry["density_material_mg_ml"], 
                    rho_occupied]
            density_profiles.append(result)
        except ValueError as e:
            print(f' FrameID-{temp_frame}: {e}')
            continue

    # print(density_profiles)
    # print(np.array(density_profiles)[:, :1].reshape(-1).shape)
    # print(np.array(density_profiles)[:,1:].shape)
    # We now create output file.
    try:
        xlabel = f"Time (ns)"
        ylabel = f"Mass density (mg/mL)"
        title  = f"Mass density"
        legends = ["Outer envelope density", "Exclude cavity density", "Estimate density"]

        write_xvg(output_file_name, np.array(density_profiles)[:, :1].reshape(-1), 
                  np.array(density_profiles)[:,1:].T, title=title, xlabel=xlabel, ylabel=ylabel, legends=legends)


    except Exception as e:
        print(f"## An exception occurred when trying to open text file {output_file_name} for output: {e}")
        quit()

    print(f"## Density profile written to {output_file_name}.")   


from dropps.share.command_class import single_command
density_3d_commands = single_command("density_3d", getargs_density, density_3d, desc)
