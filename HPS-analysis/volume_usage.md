# 液滴体积与密度函数

`volume.py` 是可逐帧调用的数值模块。输入坐标/盒长使用 **nm**，bead 质量使用 **Da**；输出体积为 **nm³**，质量浓度为 **mg/mL**。HPS 使用隐式溶剂，因此这里只计算传入的蛋白组分质量浓度。

当前实现不负责读取 cluster NPZ 或写 XVG；这些函数可以在轨迹分析循环里调用。现有 `density_3d.py` 命令仍是另一个未完成的脚本；本模块提供的可用入口见下面的示例。

## 选链与 PBC

```python
import numpy as np
from dropps.analysis.volume import (
    extract_position_array, treat_pbc, volume_cal,
    density_core, density_gibbs, density_envelope,
)

# trajectory 为现有 trajectory_class 实例，已经切换到要处理的帧。
# 以下假设聚类时用的是所有蛋白链，链等长且按链连续排列。
# chain_ids 是本帧 max_clusters 的一个元素，编号从 0 开始。
universe = trajectory.Universe
chain_num = trajectory.num_chains()
chain_length = len(universe.atoms) // chain_num
chain_ids = np.asarray(chain_ids, dtype=np.intp)

# XTC 读入 MDAnalysis 后，坐标和 dimensions 长度均为 Angstrom。
positions_nm = universe.atoms.positions.astype(np.float64) / 10.0
dimensions = universe.dimensions
if not np.allclose(dimensions[3:], 90):
    raise ValueError("当前函数只支持正交盒子")
box_nm = dimensions[:3].astype(np.float64) / 10.0

cluster_positions = extract_position_array(
    positions_nm, chain_ids, chain_num, chain_length,
)
# 质量必须与选出坐标的顺序一致。
cluster_masses = np.asarray(universe.atoms.masses, dtype=np.float64).reshape(
    chain_num, chain_length
)[chain_ids].reshape(-1)

centered = treat_pbc(
    cluster_positions, box_nm,
    chain_length=chain_length,
    cutoff=0.8,
)
np.testing.assert_allclose(centered.mean(axis=0), box_nm / 2)
```

`cutoff=0.8` 是 **接触距离**，相邻 bead 的 HPS 平衡键长仍为 **0.38 nm**。函数沿周期邻居图重组液滴，提供 `chain_length` 后还会显式连接每条链的连续 bead；不会把不同链的首尾连成一条链。

只移动已经 wrap 的坐标均值会把跨界液滴错误拉散，因而这里先重组再移动。输出是独立的 float64 数组，不修改 Universe；保持原行顺序，质心使用等权几何中心。重组后不再次 wrap，否则可能重新切断长尾。如果选区在当前 cutoff 下不连通，或接触图绕周期盒形成闭环，函数报 `ValueError`，不会默默给出伪造的有限液滴。

若聚类只用了某个 selection，链 ID 是该 selection 内的编号：请把相同 selection 的坐标按相同链顺序传入，不要直接用全系统编号。多种链长的混合体系应先依据拓扑索引构造坐标选区；当前 `extract_position_array` 的接口只支持等长链。

## 三类物理分析

### 核心平台密度

```python
core = density_core(
    centered, cluster_masses,
    center=box_nm / 2,
    bin_size=0.5,
    rho_dilute=0.0,
)
rho_in = core["density_mg_ml"]
r0 = core["interface_midpoint_nm"]
width = core["interface_width_nm"]
```

默认对径向质量密度拟合

$$
\rho(r)=\rho_\mathrm{out}+\frac{\rho_\mathrm{in}-\rho_\mathrm{out}}2
\left[1-\tanh\frac{r-r_0}{w}\right].
$$

每个球壳的分母为完整球壳体积，包括没有 bead 的部分；拟合时对模型也做球壳积分。返回 `profile` 中的半径、边界、壳层体积、质量和密度，以及 `fitted_density_mg_ml`、`relative_fit_rmse` 供检查。`interface_5_95_nm` 是 `2*atanh(0.9)*w`，不要与 `w` 混用。

单帧密度会有噪声。对于 bead 数目相同、已经各自居中的多帧，可以直接传入 `(T,N,3)` 坐标数组，函数先平均球壳质量再拟合。如果链成员随时间变化，应按固定径向边界分别计算 `radial_density_profile` 并做适当的时间分块统计；不要把不同帧坐标取并集。

`rho_dilute=0` 仅适用于已选出液滴且外部蛋白背景可忽略的近似。若拟合真实稀相浓度，应传入足够大区域内的全部蛋白坐标、指定液滴中心、`box_size=box_nm` 和 `r_max`，省略 `rho_dilute` 以同时拟合。`r_max` 不能超过最短盒长的一半，否则球壳会被周期盒截断。

拟合不具备可分辨的平台时会报错，不能据此认定小团簇已经形成体相。显著非球形或中心有空腔时，径向平台也可能没有所需物理含义。若已通过剖面确认某个球形区域位于均匀密相，可改用：

```python
core = density_core(
    centered, cluster_masses,
    center=box_nm / 2,
    core_radius=5.0,  # 示例数值，应换为从当前剖面确认的半径
)
```

该模式按此球内实际 bead 质量除以完整球体积，返回 `core_mass_da`、`core_volume_nm3` 和 `density_mg_ml`；不会假装这个用户指定区域已经通过平台拟合验证。

### Gibbs 等摩尔体积

稀相可忽略时：

```python
gibbs = density_gibbs(centered, cluster_masses, rho_dense=rho_in)
volume_e = gibbs["volume_nm3"]
radius_e = gibbs["radius_nm"]
```

函数采用质量守恒：

$$
V_e=\frac{cM-\rho_\mathrm{out}V_\mathrm{ref}}
{\rho_\mathrm{in}-\rho_\mathrm{out}},\qquad
c=1.66053906892\;\frac{\mathrm{mg/mL}}{\mathrm{Da/nm^3}}.
$$

`rho_dense` 来自平台密度或相同条件下的 slab 参考值。`density_mg_ml` 是这一输入参考密度，不是第二次独立测量：不能用 `M/V_e` 再验证输入密度。半密度半径 `r0` 与等摩尔半径也不一定相等。

若稀相不可忽略，必须计入参考域内**该组分的所有 bead**：

```python
# 适用于整个盒子里只有一个液滴、给出的两相密度对应同一种蛋白组分。
gibbs = density_gibbs(
    positions_nm, universe.atoms.masses,
    rho_dense=rho_in,
    rho_dilute=rho_out,  # 从包含稀相的剖面测量
    reference_volume=float(np.prod(box_nm)),
)
```

不能只传最大团簇质量，却扣掉整个盒子的稀相背景。存在多个液滴时，整个盒子的质量守恒只给出它们的总等摩尔体积，不能自动归给最大液滴。Gibbs 体积也不解析空腔的形状。

### 外包络、密相区域和空腔

```python
geometry = density_envelope(
    centered, cluster_masses,
    rho_dense=rho_in, rho_dilute=0.0,
    bin_size=0.5,
    smoothing_sigma=1.0,
    return_grid=False,
)
print(geometry["volume_outer_nm3"], geometry["volume_void_nm3"])
print(geometry["density_outer_mg_ml"], geometry["density_material_mg_ml"])
```

在液滴附近建立网格，把质量分配到体素后用高斯滤波，阈值默认为两相密度中点，也可显式传入 `density_threshold`。`smoothing_sigma` 为高斯核标准差，决定物理观察尺度；`bin_size` 决定数值网格精度。固定平滑尺度后改变网格间距进行收敛检查。

最大高密度连通区域决定外界面；从网格外部能连通的低密度区属于外部，包括开口通道。封闭低密度区作为空腔。高密度区域使用 6 邻接，外部 flood-fill 使用 26 邻接，避免把对角开口错误封死；这一离散分类也要检查网格收敛。

返回量：

| 键 | 定义 |
|---|---|
| `volume_outer_nm3` | 包含封闭空腔的外包络体积 |
| `volume_material_nm3` | 外包络内高于阈值的区域体积 |
| `volume_void_nm3` | 二者之差 |
| `void_fraction` | 空腔体积 / 外包络体积 |
| `density_outer_mg_ml` | 外包络内实际 bead 质量 / 外包络体积 |
| `density_material_mg_ml` | 高密度区域内实际 bead 质量 / 该区域体积 |
| `mass_outside_da` | 在外包络外的 bead 质量，包括被排除的长尾 |
| `outer_membership`, `material_membership` | 每个输入 bead 的空间归属，行顺序不变 |

这里 `material` 是由平滑密度阈值定义的区域，不是所有 bead 的硬球体积；`void` 是低蛋白浓度空间，不能仅凭隐式溶剂模型把它解释为空真空。分子间微小空隙随平滑尺度变化。

质量按 bead 中心所在区域逐颗计数，故半游离链可以部分计入。可以把 membership reshape 为 `(len(chain_ids), chain_length)`，结合每 bead 质量求每条链的质量分数。即使输入选取了整条链，外部链段也不会被错误地计入核心质量。

设置 `return_grid=True` 可取出 `density_grid_mg_ml`、`grid_origin_nm` 和三个布尔区域，便于可视化检查。网格原点指体素角点；体素中心为 `origin + (index + 0.5)*bin_size`。默认最多分配 800 万体素，超限报错，可显式调整 `max_grid_cells`。

## 原始占据体积描述符

```python
v_occupied = volume_cal(centered, bin_size=1.0)
rho_occupied = cluster_masses.sum() * 1.66053906892 / v_occupied

# 也支持原来的三个位置参数：此处必须传全链数组。
v_from_all = volume_cal(positions_nm, chain_ids, chain_length, bin_size=1.0)
```

`volume_cal` 保留 `N_occupied * bin_size**3` 的原始思路，只保存实际出现的体素坐标，不建立 590³ 的全盒数组。它会把液滴内部没有 bead 的体素全部排除，包含表面离散误差，因此单独作为依赖分辨率的占据指标。`origin` 可指定网格原点以研究平移敏感性。不要把已选出的 cluster 再按全系统 chain IDs 筛选一次。

## 验证

在此目录执行：

```powershell
& 'G:\uv-dropps3-test\dropps3\Scripts\python.exe' -m unittest discover -s tests -p test_volume.py -v
```

测试包括跨三个盒面重组、多链和长链的连通性、周期贯穿检测、单位换算、解析 tanh 球体及非零背景的 Gibbs 质量守恒、高斯等值面解析体积、封闭空腔/开口通道、尾部质量归属和网格细化。

方法依据：[Willard–Chandler 瞬时密度界面](https://doi.org/10.1021/jp909219k)，[液滴径向密度及等摩尔半径](https://acp.copernicus.org/articles/23/2525/2023/index.html)。本实现的体素离散和平滑参数需在实际轨迹上进行灵敏度验证。
