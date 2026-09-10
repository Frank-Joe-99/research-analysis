# REMD 反应坐标自由能面绘制脚本

`FEL-statistics-plot.py` 用于根据副本交换分子动力学（Replica Exchange Molecular Dynamics，REMD）轨迹中提取的两个反应坐标，计算并绘制二维自由能面（Free Energy Landscape，FEL）。脚本先对输入的二维坐标进行网格统计，再按照玻尔兹曼关系将归一化布居转换为自由能，并输出带有连续色条的等高线图。

> 本脚本只负责二维数据的分箱、自由能计算和绘图，不负责 REMD 轨迹交换、轨迹重构或反应坐标提取。请先将 REMD 分析得到的坐标整理为输入文件。

## 环境依赖

- Python 3
- NumPy
- SciPy
- Matplotlib

例如：

```bash
pip install numpy scipy matplotlib
```

## 输入文件

使用 `-xy` 指定输入文件。文件必须包含**恰好两列数值**，每一行代表一个采样点：

```text
x_coordinate    y_coordinate
0.20            3.10
0.25            3.05
0.30            2.98
```

以 `#` 或 `@` 开头的行会被忽略，因此可以直接使用包含 Gromacs 风格注释的分析结果文件。两列数据应分别对应 X、Y 方向的反应坐标；对于 REMD，建议使用已经按照目标温度汇总、重排或重加权后的采样数据。脚本本身不会自动处理不同温度副本的数据。

## 使用方法

```bash
python FEL-statistics-plot.py -xy coordinate.dat
```

运行后，脚本会依次要求输入 X、Y 方向的分箱宽度，并要求选择两个坐标的显示标签。例如：

```text
x bin width: 1
 y bin width: 1
X label number: 1
Y label number: 4
```

坐标范围根据输入数据自动设置为：

- 下限：数据最小值 × 0.9
- 上限：数据最大值 × 1.1

也可以在命令行中指定常用参数：

```bash
python FEL-statistics-plot.py \
    -xy coordinate.dat \
    -t 300 \
    -d 1e-6 \
    -n 20 \
    -p png \
    -e
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `-xy FILE` | 必填 | 两列 X/Y 坐标输入文件 |
| `-t TEMP` | `310` | 温度，单位为 K |
| `-c VALUE` | 自动 | 色条最小值，单位为 kcal/mol，必须小于 0 |
| `-d VALUE` | 自动 | 自由能计算使用的 pseudocount，必须大于 0 |
| `-e` | 关闭 | 保存自由能矩阵为 `FEL-energy.dat` |
| `-i VALUE` | `bilinear` | 为兼容旧版本保留，当前未使用 |
| `-n N` | `16` | 色阶数量 |
| `-p FORMAT` | `svg` | 输出格式，如 `svg`、`png`、`jpg`、`eps` |
| `-s` | 关闭 | 绘图后交互查询某个网格中的原始数据行号 |

## 坐标标签

运行时可以选择预设标签：

1. β structure（β 含量，%）
2. Helix structure（螺旋含量，%）
3. H-bond number（氢键数）
4. Rg（nm）
5. SASA（nm²）
6. Hydrophobic SASA（nm²）
7. 其他：手动输入标签

选择预设标签时，标签名称还会用于生成输出文件名后缀。

## 输出文件

默认输出：

```text
FEL-plot.svg
```

输出文件名格式为：

```text
FEL-plot<X坐标后缀><Y坐标后缀>.<格式>
```

例如选择 Rg 作为 X 坐标、SASA 作为 Y 坐标并输出 PNG 时，文件名为：

```text
FEL-plot-rg-sasa.png
```

使用 `-e` 时，另外保存：

```text
FEL-energy.dat
```

该文件为制表符分隔的二维自由能矩阵。自由能单位为 kcal/mol。

## 计算说明

二维坐标首先被统计到网格中，并归一化为布居概率 `P`。自由能按照下式计算：

```text
G(x, y) = -R T ln[(P(x, y) + p) / p]
```

其中：

- `R = 0.0019863 kcal/(mol·K)`；
- `T` 为 `-t` 指定的温度；
- `p` 为 pseudocount；未指定 `-d` 时，使用非零网格布居中的最小值。

零布居网格通过 pseudocount 处理，并对应较高自由能；图中色条范围默认为计算得到的最低自由能到 0 kcal/mol。若需要比较多个自由能面，建议为不同数据集使用相同的 `-t`、`-d`、分箱宽度和 `-c` 设置。

## 交互查询网格数据

加入 `-s` 可以在绘图完成后查询指定坐标所在网格中的输入数据行号：

```bash
python FEL-statistics-plot.py -xy coordinate.dat -s
```

输入 `x position` 和 `y position` 后，脚本会显示该网格中的采样点数量及最多前 20 个原始行号；输入 `n` 可退出查询。

## 注意事项

- 输入文件必须有且仅有两列数值数据。
- 分箱宽度通过交互输入，必须为正数；过大的分箱会降低自由能面的空间分辨率，过小的分箱会产生大量低采样或空网格。
- `-d` 是概率矩阵中的 pseudocount，不是能量单位；跨数据集比较时应保持一致。
- `-c` 只改变绘图色条的下限，不会改变 `FEL-energy.dat` 中的计算结果。
- 若输入数据没有落入自动生成的网格，脚本会终止并提示错误。
