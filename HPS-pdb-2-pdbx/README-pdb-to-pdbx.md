# TER 分段 HPS PDB → PDBx/mmCIF 转换器

`pdb_to_pdbx.py` 把每个 `TER` 分隔出的坐标段视为一条独立链，并生成唯一的四字符链 ID（默认 `C001`、`C002` …）。它特别适合链 ID 在传统 PDB 中循环复用的模拟文件。

## 使用

```powershell
python .\pdb_to_pdbx.py "input.pdb" `
  --output "output.cif" `
  --chain-count 216 `
  --chain-length 140
```

- `input.pdb`：输入文件（位置参数）。
- `--output` / `-o`：输出 PDBx/mmCIF 文件，必填；惯例使用 `.cif` 扩展名。
- `--chain-count`：每个 model 预期的 `TER` 段数。它是校验参数；省略时自动接受检测到的数量。
- `--chain-length`：每个 `TER` 段预期的**唯一残基数**，不是原子数；同样只作校验。
- `--chain-prefix C` 和 `--chain-width 3`：生成 `C001` 格式的 ID。比如 `--chain-prefix P --chain-width 4` 生成 `P0001`。
- `--auth-chain-source generated`：默认把新 ID 写入 `_atom_site.label_asym_id` 和 `_atom_site.auth_asym_id`，下游程序最不容易产生歧义。若要保留原 PDB 的单字符值，请改为 `original`；唯一的规范 ID 仍在 `label_asym_id` 中。
- `--mapping-output chains.tsv`：指定链映射表位置。默认与输出同名，例如 `output.chains.tsv`。

## 对给定 `mdrun.pdb` 的命令

```powershell
python .\pdb_to_pdbx.py "G:\VWshare-ssd\pY39-HPS\DROPPS-spontaneous-llps\wt-as\mdrun-2\mdrun.pdb" `
  --output ".\mdrun.pdbx.cif" `
  --chain-count 216 `
  --chain-length 140
```

该文件会生成 `C001` 至 `C216`，并写出 `mdrun.pdbx.chains.tsv`。转换器会保留坐标、占据率、B 因子、原始残基编号、插入码和模型编号；`CRYST1` 也会转换为基本晶胞与空间群字段。

## 范围和限制

这是“坐标文件转换器”，不是 wwPDB 的实验结构提交工具：PDB 中不存在的实验元数据、实体定义、键连接和化学成分信息无法自动恢复。若元素列不是有效元素符号（例如部分粗粒化模型把残基单字母码放在该列），输出中的 `_atom_site.type_symbol` 会安全地写为 `?`，而不会伪造元素类型。
