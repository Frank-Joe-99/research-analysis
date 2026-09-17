# cluster analysis in DROPPS package by Yun Zhou / Zichao Wang @ Fudan
# Development started on Sep 10 2026

from argparse import ArgumentParser
from dropps.share.trajectory import trajectory_class
from dropps.share.forcefield import getff, forcefield_list

from tqdm import tqdm
import numpy as np

from MDAnalysis.lib.nsgrid import FastNS

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)

from pathlib import Path
import os


class cluster_analysis:
    """
    统计 chain contact map 中的连通 cluster。

    连通性只依赖链间接触数，与链长无关；混合链长体系也使用同一个
    cutoff 阈值，cluster size 仍表示链的条数，而不是残基总数。
    """

    def __init__(self, chain_connectivity, num_chain, cutoff=2) -> None:
        self.chain_connectivity = chain_connectivity
        self.num_chain = num_chain
        self.cutoff = cutoff
        self.in_cluster = [False] * num_chain
        self.cluster_idx = [-1] * num_chain
        self.clusters = []
        self.run()
        self.cluster_size = self.cluster_size_analysis()
        self.cluster_size_distribution = self.cluster_size_distribution_analysis()

    def find_clusters(self, chain_index, cluster_id):
        """迭代查找与指定 chain 连通的所有 chain。"""
        stack = [chain_index]
        self.cluster_idx[chain_index] = cluster_id
        self.in_cluster[chain_index] = True

        while stack:
            current = stack.pop()
            self.clusters[cluster_id].append(current)
            for neighbour in np.flatnonzero(
                self.chain_connectivity[current] >= self.cutoff
            ):
                if neighbour != current and self.cluster_idx[neighbour] == -1:
                    self.cluster_idx[neighbour] = cluster_id
                    self.in_cluster[neighbour] = True
                    stack.append(neighbour)

    def run(self):
        """遍历所有 chain，得到全部 clusters。"""
        for chain_index in range(self.num_chain):
            if self.cluster_idx[chain_index] == -1:
                cluster_id = len(self.clusters)
                self.clusters.append([])
                self.find_clusters(chain_index, cluster_id)
        return self.clusters

    def cluster_size_analysis(self):
        cluster_size = [len(cluster) for cluster in self.clusters]
        self.max_cluster = max(self.clusters, key=len, default=[])
        return cluster_size

    def cluster_size_distribution_analysis(self):
        """Return ``(cluster_size, fraction_of_chains)`` for the current frame."""
        sizes = np.asarray(self.cluster_size, dtype=np.intp)
        if sizes.size == 0:
            return np.empty((0, 2), dtype=float)

        cluster_counts = np.bincount(sizes)
        cluster_sizes = np.flatnonzero(cluster_counts)
        chain_fractions = cluster_sizes * cluster_counts[cluster_sizes] / self.num_chain
        return np.column_stack((cluster_sizes, chain_fractions))

def _build_lookup_fixed(chains, max_idx):
    """
    从 contact.py 复制并调整：全局 atom/bead 编号 -> 链编号、链内位置。

    链编号是 chains 中的下标；未选中的粒子记为 -1。
    拓扑在分析过程中不变，因此该查找表只在帧循环外建立一次。
    每条链都独立生成 0..len(chain)-1 的位置编号，天然支持不同链长，
    不需要统一长度或填充。保留函数名中的 fixed 以兼容已有调用。
    """
    idx_to_chain = np.full(max_idx + 1, -1, dtype=np.int32)
    idx_to_pos = np.full(max_idx + 1, -1, dtype=np.int32)

    for cid, idxs in enumerate(chains):
        idxs = np.asarray(idxs, dtype=np.int64)
        valid = (idxs >= 0) & (idxs <= max_idx)
        idx_to_chain[idxs[valid]] = cid
        idx_to_pos[idxs[valid]] = np.arange(len(idxs), dtype=np.int32)[valid]

    return idx_to_chain, idx_to_pos


def _safe_lookup(arr, idx):
    """沿用 contact.py 的批量查表方式，同时屏蔽负数和越界编号。"""
    out = np.full(idx.shape, -1, dtype=np.int32)
    valid = (idx >= 0) & (idx < arr.shape[0])
    out[valid] = arr[idx[valid]]
    return out


def _canonicalize_pairs(pairs, undirected=True, unique=True, drop_self=True):
    """从 contact.py 复制：规范化形状为 (N, 2) 的粒子接触对。"""
    pairs = np.asarray(pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("pairs must be a (N,2) array-like of indices")

    if pairs.size == 0:
        return pairs

    if drop_self:
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    if undirected:
        pairs = np.sort(pairs, axis=1)
    if unique and pairs.size:
        pairs = np.unique(pairs, axis=0)
    return pairs


def accumulate_chain_contact_map_fixed(
    pairs, chains, chain_lookup, unique_pairs=True,
):
    """
    将 contact.py 的二维残基图聚合改成逐链对计数，支持混合链长。

    输入 pairs 是当前帧通过距离筛选后的全局 atom/bead 编号对。
    返回对称的 (num_chain, num_chain) int64 矩阵：元素 [i, j]
    是链 i 与链 j 的接触数，对角线和没有接触的链对均为 0。

    chains 中各条链可以选中不同数量的粒子；函数名中的 fixed 仅为
    兼容已有调用保留，不再要求等长链。每个残基对应一个选中的 bead；
    若直接输入全原子选择，这里得到的是 atom-atom 接触数。
    链 i、j 的临时图形状为 (len(chains[i]), len(chains[j]))。
    各链对复用同一个 bool 缓冲区，其容量按实际接触链对的需要增长，
    不把短链填充至最长链长度，也不同时保存所有链对的残基图。
    """
    num_chain = len(chains)
    chain_lengths = [len(chain) for chain in chains]

    chain_contact_map_temp = np.zeros((num_chain, num_chain), dtype=np.int64)
    pairs = _canonicalize_pairs(
        pairs, undirected=True, unique=unique_pairs, drop_self=True,
    )
    if pairs.size == 0:
        return chain_contact_map_temp

    idx_to_chain, idx_to_pos = chain_lookup
    chain_a = _safe_lookup(idx_to_chain, pairs[:, 0])
    chain_b = _safe_lookup(idx_to_chain, pairs[:, 1])
    pos_a = _safe_lookup(idx_to_pos, pairs[:, 0])
    pos_b = _safe_lookup(idx_to_pos, pairs[:, 1])

    # cluster 只使用链间接触；链内接触不应进入矩阵的对角线。
    inter = (chain_a >= 0) & (chain_b >= 0) & (chain_a != chain_b)
    if not np.any(inter):
        return chain_contact_map_temp
    chain_a, chain_b = chain_a[inter], chain_b[inter]
    pos_a, pos_b = pos_a[inter], pos_b[inter]

    # 统一以较小链编号作为临时图的行；交换链时同步交换链内位置。
    swap = chain_a > chain_b
    chain_i = np.minimum(chain_a, chain_b).astype(np.int64)
    chain_j = np.maximum(chain_a, chain_b).astype(np.int64)
    row_pos = np.where(swap, pos_b, pos_a)
    col_pos = np.where(swap, pos_a, pos_b)

    # 将同一个链对的接触排在一起，只遍历确实有接触的链对。
    # 链对编码使用 int64，避免 chain_i * num_chain 的 int32 溢出。
    pair_ids = chain_i * num_chain + chain_j
    order = np.argsort(pair_ids)
    sorted_pair_ids = pair_ids[order]
    boundaries = np.concatenate((
        [0], np.flatnonzero(np.diff(sorted_pair_ids)) + 1, [len(order)],
    ))

    # 使用一维缓冲区，按当前链对的 Li x Lj 重新解释形状。
    # 只在容量不足时扩容，避免为每一种长度组合缓存一张二维矩阵。
    residue_contact_buffer = np.empty(0, dtype=bool)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        i, j = divmod(int(sorted_pair_ids[start]), num_chain)
        length_i, length_j = chain_lengths[i], chain_lengths[j]
        map_size = length_i * length_j
        if residue_contact_buffer.size < map_size:
            residue_contact_buffer = np.zeros(map_size, dtype=bool)
        residue_contact_map = residue_contact_buffer[:map_size].reshape(
            length_i, length_j,
        )

        members = order[start:stop]
        rows, cols = row_pos[members], col_pos[members]
        residue_contact_map[rows, cols] = True
        contact_number = residue_contact_map.sum(dtype=np.int64)

        chain_contact_map_temp[i, j] = contact_number
        chain_contact_map_temp[j, i] = contact_number

        # 临时图的行、列分别属于两条不同的链，只填一次，不需要除以 2。
        # 清除本链对用过的位置，使下一链对即使改变图的形状也从全零开始。
        residue_contact_map[rows, cols] = False
        # 释放当前二维视图，确保后续扩容时旧缓冲区不会被该视图保留。
        del residue_contact_map

    return chain_contact_map_temp


prog = "cluster"
desc = '''This performant program loads all chains in your simulation and analysis the cluster distribution of the spontaneous LLPS process.'''

def getargs_cluster(argv):
    parser = ArgumentParser(prog=prog, description=desc)

    parser.add_argument('-s', '--run-input', type=str, required=True, 
                        help="TPR file containing all information for a simulation run.")
    
    parser.add_argument('-f', '--input', type=str, required=True, 
                    help="XTC file which is taken as input trajectory.")
    
    parser.add_argument('-n', '--index', type=str, required=False,
                        help="Index file containing non-default groups.")

    parser.add_argument('-cs', '--cutoff-scheme', type=str, choices=["global", "residue"], required=True, default="global", 
                        help="Scheme for determining inter-residue contacts.")
    
    parser.add_argument('-c', '--cutoff', type=float, default=0.8,
                        help="Cutoff distance for contact calculation, unit is nanometer.")

    parser.add_argument('-cc', '--cutoff-cluster', type=int, default=5,
                        help="Cutoff number for cluster calculation, default is 5.")

    parser.add_argument('-cm', '--cutoff-multiplier', type=float, default=1.2,
                        help="Factor which is multiplied to sigma value for residue-wise contact cutoff.")
    
    parser.add_argument('-ff', '--forcefield', choices=forcefield_list, default='HPS',
                        help="Forcefield selection")
    
    parser.add_argument('-b', '--start-time', type=int,
                        help="Time (ns) of the first frame to calculate.")

    parser.add_argument('-e', '--end-time', type=int,
                        help="Time (ns) of the last frame to calculate.")

    parser.add_argument('-dt', '--delta-time', type=float,
                        help="Intervals (ns) between two calculated frames.")
    
    parser.add_argument('-ocd', '--output-cluster-distribution', type=str, default='cluster_distribution.npz',
                        help="numpy NPZ file to write the cluster size distribution as a function of time.")
    
    args = parser.parse_args(argv)
    return args


def cluster_distribution_fast(args):

    output_parameters = [args.output_cluster_distribution,
                        ]
    if all(x is None for x in output_parameters):
        print("ERROR: No output file specified.")
        quit()

    if args.cutoff_cluster < 1:
        raise ValueError("cutoff_cluster must be at least 1")

    # 加载拓扑和轨迹读取器；后续逐帧读取，不将整个 XTC 放入内存。
    try:
        trajectory = trajectory_class(args.run_input, args.index, args.input)
    except:
        print("## An exception occurred when trying to open trajectory file %s." % args.input)
        quit()

    # 本程序固定使用周期边界条件；FastNS 内部处理最小镜像距离。
    print("## Distance will be calculated using periodic boundary conditions.")

    # We determine cutoff scheme and generate cutoff distance vector
    if args.cutoff_scheme == "global":
        print(f"## Will use a glocal cutoff of {args.cutoff} nm.")
        cutoff_vector = [args.cutoff * 10.0 for atom in range(trajectory.num_atoms())]
    elif args.cutoff_scheme == "residue":
        print(f"Will use a residue responsive cutoff will a sigma multiplier of {args.cutoff_multiplier}")
        
        current_file_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        forcefields_dir = Path(current_file_dir) / "share" / "forcefields"
        files1 = list(forcefields_dir.glob("*.ff"))
        cwd = Path.cwd()
        files2 = list(cwd.glob("*.ff"))

        # Get base names for conflict checking
        names1 = set(f.name for f in files1)
        names2 = set(f.name for f in files2)

        # Check for conflicts
        conflicts = names1 & names2
        if conflicts:
            print("ERROR: Conflict detected! The following .ff file(s) exist in both system and working directory:")
            for name in conflicts:
                print(f"  {name}")
            quit()

        # Combine and make a list of paths
        all_files = files1 + files2

        # Print basename list
        if not all_files:
            print("ERROR: No forcefields found.")
            quit()
        else:
            print("## Available forcefields:")
            for idx, f in enumerate(all_files, start=1):
                print(f"{idx}: {f.name}")

            # Let user select

            if args.forcefield is not None:
                filenames = [file for file in all_files if os.path.basename(file) == args.forcefield + ".ff"]

                if len(filenames) == 0:
                    print(f"ERROR: Unknown forcefield {args.forcefield}.")
                    quit()
                selected_file_path = Path(filenames[0])
            else:

                while True:
                    try:
                        choice = int(input("Select a file by index: "))
                        if 1 <= choice <= len(all_files):
                            break
                        else:
                            print("Invalid choice. Try again.")
                    except ValueError:
                        print("Please enter a valid integer.")

                selected_file_path = all_files[choice - 1]
            print(f"## Selected forcefield: {selected_file_path.name}")

            # Save path in parameter
            parameter_file_path = selected_file_path

        forcefield = getff(parameter_file_path)
        cutoff_vector = [args.cutoff_multiplier * sigma * 10.0 for sigma in trajectory.sigmas]
    else:
        print(f"ERROR: Cannot process cutoff scheme {args.cutoff_scheme}")
        quit()

    cutoff_vector = np.array(cutoff_vector).astype(np.float32)
    # 仅保留 O(N) 的 cutoff 向量，候选粒子对的 cutoff 在每帧按需计算。
    # 原版的 (N, N) cutoff_matrix 在这里完全省去。

    # We treat time for analysis and generate frame for analysis
    start_frame, end_frame, interval_frame = trajectory.time2frame(args.start_time, args.end_time, args.delta_time)
    frame_list = range(start_frame, end_frame, interval_frame)

    # 本程序分析同一组链中的全部链间接触；两组的输入顺序可以不同。
    # 程序当前使用所有的链进行分析（目前单一组分或者多组分不做区分，全部加载进来）
    reference_group, _ = trajectory.getSelection("group 0")
    selection_group, _ = trajectory.getSelection("group 0")
    
    reference_indices = np.sort(np.asarray(reference_group.indices, dtype=np.int64))
    selection_indices = np.sort(np.asarray(selection_group.indices, dtype=np.int64))
    if reference_indices.size == 0 or selection_indices.size == 0 :
        raise ValueError("The reference/selection group contains no atoms")
    if not np.array_equal(reference_indices, selection_indices):
        raise ValueError(
            "Cluster analysis requires reference and selection to contain the same atoms"
        )

    # 所有链共同参与 cluster 分析；按真实拓扑分链，允许任意链长组合。
    # 两组已验证为相同粒子集合，只需拆分一次，避免生成不同的局部链编号。
    reference_chains = trajectory.index.splitch_indices(reference_indices)
    chain_length_list_reference = [len(chain) for chain in reference_chains]
    chain_lengths = np.asarray(chain_length_list_reference, dtype=np.int64)
    lengths, length_counts = np.unique(chain_lengths, return_counts=True)
    length_summary = ", ".join(
        f"{length} selected beads: {count} chains"
        for length, count in zip(lengths, length_counts)
    )
    print(f"## Processed {len(reference_chains)} chains ({length_summary}).")
    print(f"## Reference and selection contain the same atoms; each contact is searched once.")

    # 从 contact.py 移植：搜索组局部编号 -> 全局 atom/bead 编号 -> 链/链内位置。
    # 同一组只搜索一次；查找表在帧循环外建立，后续各帧复用。
    search_atom_indices = reference_indices
    search_atoms = trajectory.Universe.atoms[search_atom_indices]
    chain_lookup = _build_lookup_fixed(reference_chains, int(search_atom_indices.max()))
    # cluster 内的编号为 0..num_chain-1；保存其对应的原始拓扑 chain ID。
    chain_ids = np.asarray([
        trajectory.get_chainID(int(chain[0])) for chain in reference_chains
    ], dtype=np.int64)

    selected_cutoffs = cutoff_vector[search_atom_indices]
    if not np.all(np.isfinite(selected_cutoffs)) or np.any(selected_cutoffs <= 0):
        raise ValueError("All selected contact cutoffs must be finite and positive")
    # 只取选中粒子的最大 cutoff。先舍入再取搜索半径，避免舍入增大阈值后漏搜。
    search_cutoff = float(np.round(selected_cutoffs.max(), 4))
    if search_cutoff <= 0:
        raise ValueError("Contact cutoff becomes zero after rounding to four decimals")

    # We start calculation
    print(f"## We will start calculating.")
    num_chain = len(reference_chains)
    num_frames = len(frame_list)
    if num_frames == 0:
        raise ValueError("No frames selected; check the start/end time and frame interval")
    cluster_size_distribution =  np.zeros((num_chain + 1, num_frames), dtype=np.float16)

    # 时间使用 float64，避免长轨迹或较小帧间隔在 float16 下溢出/丢失精度。
    frame_indices = np.empty(num_frames, dtype=np.float64)
    max_cluster = []
    chain_contact_map_max = []
    chain_contact_map_average = []
    print(f"## Neighborhood search: {len(search_atom_indices)} selected atoms, cutoff {search_cutoff / 10.0} nm.")
    print(f"## Each frame produces a {num_chain} x {num_chain} chain contact matrix.")
    print(f"## Calculating for requested frames.")

    for num_frame, frame_index in enumerate(tqdm(frame_list)):
        ts = trajectory.Universe.trajectory[frame_index]
        pos = np.asarray(search_atoms.positions, dtype=np.float32)

        # 固定启用 PBC，使用当前帧的真实周期盒。
        box = ts.dimensions
        if box is None:
            raise ValueError("Periodic neighbor search requires trajectory box dimensions")

        # 替换原来的全距离矩阵：只返回最大 cutoff 内的候选粒子对。
        ns = FastNS(search_cutoff, pos, box=box, pbc=True)
        search_result = ns.self_search()
        pairs = search_result.get_pairs()
        pairs_atomid = search_atom_indices[pairs]
        pair_distances = search_result.get_pair_distances()

        if args.cutoff_scheme == "residue":
            pair_cutoffs = np.round(cutoff_vector[pairs_atomid].mean(axis=1), 4)
        else:
            pair_cutoffs = search_cutoff
        # 保持 cluster 原版的严格 < 和四位小数舍入规则；contact.py 使用的是 <=。
        pairs_atomid = pairs_atomid[pair_distances < pair_cutoffs]
        del ns, search_result, pairs, pair_distances, pair_cutoffs

        # 每个相邻链对使用 Li x Lj 的矩形残基图，求和后写入对称链级矩阵。
        # 不按链长分组搜索，不同长度链之间的接触也会进入同一张链级矩阵。
        # self_search 已返回不重复的无向粒子对，可跳过额外的粒子对去重。
        chain_contact_map_temp = accumulate_chain_contact_map_fixed(
            pairs_atomid, reference_chains, chain_lookup, unique_pairs=False,
        )
        del pairs_atomid
        chain_contact_map_max.append(chain_contact_map_temp.max())
        chain_contact_map_average.append(chain_contact_map_temp.mean())

        # cluster analysis
        analyzer = cluster_analysis(
            chain_connectivity=chain_contact_map_temp,
            num_chain=num_chain, cutoff=args.cutoff_cluster,)
        
        # get the cluster size distribution
        distribution = analyzer.cluster_size_distribution
        
        cluster_sizes = distribution[:, 0].astype(np.intp)
        distributions = distribution[:, 1]
        cluster_size_distribution[cluster_sizes, num_frame] = distributions

        max_cluster.append(analyzer.max_cluster)
        # 保存本帧分析结果后释放矩阵，不保留所有帧的链级接触矩阵。
        del analyzer, chain_contact_map_temp

    output_path = Path(args.output_cluster_distribution)
    # chain_lengths[k] 与 chain_ids[k] 对应同一条链，便于解释混合链长结果。
    np.savez(
        output_path, frame_indices=frame_list,
        cluster_distributions=cluster_size_distribution, chain_ids=chain_ids,
        chain_lengths=chain_lengths,
    )
    max_clusters_array = np.empty(len(max_cluster), dtype=object)
    max_clusters_array[:] = max_cluster
    np.savez(
        output_path.with_name('max_cluster_' + output_path.name),
        frame_indices=frame_list,
        max_clusters=max_clusters_array,
        chain_ids=chain_ids,
        chain_lengths=chain_lengths,
    )
    #print(chain_contact_map_max)
    np.savez(
        output_path.with_name('chain_contact_map_max_' + output_path.name),
        chain_contact_map_max,
    )
    #print(chain_contact_map_average)
    np.savez(
        output_path.with_name('chain_contact_map_average_' + output_path.name),
        chain_contact_map_average,
    )


from dropps.share.command_class import single_command
cluster_analysis_commands = single_command("cluster", getargs_cluster, cluster_distribution_fast, desc)


if __name__ == "__main__":
    # 支持直接运行该 fast 文件进行手动测试，无需修改已有的命令注册。
    import sys
    cluster_distribution_fast(getargs_cluster(sys.argv[1:]))
