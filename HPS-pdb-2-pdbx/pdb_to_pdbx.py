#!/usr/bin/env python3
"""Convert a TER-delimited PDB coordinate file to a coordinate PDBx/mmCIF file.

The converter is intended for simulation/coarse-grained PDB files where the
one-character PDB chain ID is re-used.  Each TER-delimited segment becomes one
unique chain.  By default, the generated IDs are C001, C002, ... .

This is a coordinate converter, not a wwPDB deposition-preparation program:
information absent from a PDB file (experiment, chemistry, connectivity, and
full entity description) cannot be reconstructed automatically.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PERIODIC_SYMBOLS = {
    "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG",
    "AL", "SI", "P", "S", "CL", "AR", "K", "CA", "SC", "TI", "V", "CR",
    "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR",
    "KR", "RB", "SR", "Y", "ZR", "NB", "MO", "TC", "RU", "RH", "PD",
    "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA",
    "CE", "PR", "ND", "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER",
    "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU",
    "HG", "TL", "PB", "BI", "PO", "AT", "RN", "FR", "RA", "AC", "TH",
    "PA", "U", "NP", "PU", "AM", "CM", "BK", "CF", "ES", "FM", "MD",
    "NO", "LR", "RF", "DB", "SG", "BH", "HS", "MT", "DS", "RG", "CN",
    "NH", "FL", "MC", "LV", "TS", "OG", "D",
}


@dataclass(frozen=True)
class Atom:
    record: str
    serial: str
    atom_name: str
    alt_id: str
    residue_name: str
    source_chain_id: str
    residue_number: str
    insertion_code: str
    x: float
    y: float
    z: float
    occupancy: str
    b_factor: str
    element: str
    charge: str


@dataclass
class Segment:
    atoms: list[Atom]

    @property
    def source_chain_ids(self) -> list[str]:
        return sorted({atom.source_chain_id for atom in self.atoms})

    @property
    def residue_keys(self) -> list[tuple[str, str, str]]:
        keys: list[tuple[str, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for atom in self.atoms:
            key = (atom.residue_number, atom.insertion_code, atom.residue_name)
            if key not in seen:
                keys.append(key)
                seen.add(key)
        return keys


@dataclass
class Model:
    number: int
    segments: list[Segment]


class PdbParseError(ValueError):
    pass


def field(line: str, start: int, end: int) -> str:
    """Return a stripped PDB fixed-column field using zero-based end-exclusive offsets."""
    return line[start:end].strip() if len(line) > start else ""


def parse_float(value: str, what: str, line_number: int) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise PdbParseError(f"line {line_number}: invalid {what}: {value!r}") from exc


def parse_atom(line: str, line_number: int) -> Atom:
    if len(line) < 54:
        raise PdbParseError(
            f"line {line_number}: ATOM/HETATM record is shorter than column 54"
        )
    record = field(line, 0, 6)
    atom_name = field(line, 12, 16)
    residue_name = field(line, 17, 20)
    if not atom_name or not residue_name:
        raise PdbParseError(
            f"line {line_number}: atom name and residue name are both required"
        )
    return Atom(
        record=record,
        serial=field(line, 6, 11) or str(line_number),
        atom_name=atom_name,
        alt_id=field(line, 16, 17),
        residue_name=residue_name,
        source_chain_id=field(line, 21, 22),
        residue_number=field(line, 22, 26) or "?",
        insertion_code=field(line, 26, 27),
        x=parse_float(field(line, 30, 38), "X coordinate", line_number),
        y=parse_float(field(line, 38, 46), "Y coordinate", line_number),
        z=parse_float(field(line, 46, 54), "Z coordinate", line_number),
        occupancy=field(line, 54, 60) or "?",
        b_factor=field(line, 60, 66) or "?",
        element=field(line, 76, 78),
        charge=field(line, 78, 80),
    )


def parse_cryst1(line: str, line_number: int) -> dict[str, str]:
    try:
        return {
            "a": f"{float(field(line, 6, 15)):g}",
            "b": f"{float(field(line, 15, 24)):g}",
            "c": f"{float(field(line, 24, 33)):g}",
            "alpha": f"{float(field(line, 33, 40)):g}",
            "beta": f"{float(field(line, 40, 47)):g}",
            "gamma": f"{float(field(line, 47, 54)):g}",
            "space_group": field(line, 55, 66) or "P 1",
            "z": field(line, 66, 70) or "?",
        }
    except ValueError as exc:
        raise PdbParseError(f"line {line_number}: invalid CRYST1 record") from exc


def parse_pdb(path: Path) -> tuple[list[Model], dict[str, str] | None]:
    models: list[Model] = []
    segments: list[Segment] = []
    atoms: list[Atom] = []
    current_model_number = 1
    crystallography: dict[str, str] | None = None

    def finish_segment() -> None:
        nonlocal atoms
        if atoms:
            segments.append(Segment(atoms))
            atoms = []

    def finish_model() -> None:
        nonlocal segments
        finish_segment()
        if segments:
            models.append(Model(current_model_number, segments))
            segments = []

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            record = field(line, 0, 6)
            if record in {"ATOM", "HETATM"}:
                atoms.append(parse_atom(line, line_number))
            elif record == "TER":
                finish_segment()
            elif record == "MODEL":
                # A MODEL always starts a fresh coordinate model.  Tolerate files
                # that omit ENDMDL before the next MODEL.
                finish_model()
                model_token = field(line, 10, 14)
                try:
                    current_model_number = int(model_token)
                except ValueError:
                    current_model_number = len(models) + 1
            elif record == "ENDMDL":
                finish_model()
                current_model_number += 1
            elif record == "CRYST1" and crystallography is None:
                crystallography = parse_cryst1(line, line_number)

    finish_model()
    if not models:
        raise PdbParseError("no ATOM or HETATM records were found")
    return models, crystallography


def make_chain_id(index: int, prefix: str, width: int) -> str:
    return f"{prefix}{index:0{width}d}"


def cif_value(value: object | None) -> str:
    """Encode one value safely in CIF 1.1 token syntax."""
    if value is None or value == "":
        return "?"
    text = str(value)
    if text in {".", "?"}:
        return text
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_+\-.]*", text) and text.lower() not in {
        "data_", "loop_", "stop_", "save_", "global_",
    }:
        return text
    if "'" not in text and "\n" not in text:
        return f"'{text}'"
    if '"' not in text and "\n" not in text:
        return f'"{text}"'
    return f";\n{text}\n;"


def format_number(value: float) -> str:
    return f"{value:.3f}"


def element_symbol(raw_element: str) -> str:
    """Return a valid chemical symbol, otherwise CIF's unknown-value marker."""
    value = raw_element.strip().upper()
    return value.title() if value in PERIODIC_SYMBOLS else "?"


def validate_models(
    models: list[Model], expected_chain_count: int | None, expected_chain_length: int | None
) -> None:
    reference_count = len(models[0].segments)
    for model in models:
        if len(model.segments) != reference_count:
            raise PdbParseError(
                f"model {model.number} contains {len(model.segments)} TER segments; "
                f"model {models[0].number} contains {reference_count}"
            )
    if expected_chain_count is not None and reference_count != expected_chain_count:
        raise PdbParseError(
            f"expected {expected_chain_count} TER segments per model, found {reference_count}"
        )
    if expected_chain_length is not None:
        failures: list[str] = []
        for model in models:
            for chain_index, segment in enumerate(model.segments, start=1):
                residue_count = len(segment.residue_keys)
                if residue_count != expected_chain_length:
                    failures.append(
                        f"model {model.number}, segment {chain_index}: {residue_count} residues"
                    )
        if failures:
            preview = "; ".join(failures[:8])
            extra = "" if len(failures) <= 8 else f"; ... {len(failures) - 8} more"
            raise PdbParseError(
                f"expected {expected_chain_length} residues per TER segment; {preview}{extra}"
            )


def write_mmcif(
    output_path: Path,
    models: list[Model],
    crystallography: dict[str, str] | None,
    prefix: str,
    width: int,
    auth_chain_source: str,
    block_name: str,
) -> int:
    chain_ids = [make_chain_id(i, prefix, width) for i in range(1, len(models[0].segments) + 1)]
    if len(chain_ids) != len(set(chain_ids)):
        raise PdbParseError("generated chain IDs are not unique; change --chain-prefix or --chain-width")

    columns = [
        "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
        "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
        "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
        "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
        "_atom_site.pdbx_formal_charge", "_atom_site.auth_seq_id", "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id", "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(f"data_{cif_value(block_name)}\n#\n")
        handle.write("_entry.id " + cif_value(block_name) + "\n")
        handle.write("_audit_conform.dict_name mmcif_pdbx.dic\n")
        handle.write("_audit_conform.dict_version 5.0\n#\n")
        if crystallography:
            handle.write("_cell.length_a " + crystallography["a"] + "\n")
            handle.write("_cell.length_b " + crystallography["b"] + "\n")
            handle.write("_cell.length_c " + crystallography["c"] + "\n")
            handle.write("_cell.angle_alpha " + crystallography["alpha"] + "\n")
            handle.write("_cell.angle_beta " + crystallography["beta"] + "\n")
            handle.write("_cell.angle_gamma " + crystallography["gamma"] + "\n")
            handle.write("_cell.Z_PDB " + cif_value(crystallography["z"]) + "\n")
            handle.write("_symmetry.space_group_name_H-M " + cif_value(crystallography["space_group"]) + "\n#\n")
        handle.write("loop_\n" + "\n".join(columns) + "\n")
        for model in models:
            for segment_index, segment in enumerate(model.segments):
                generated_chain = chain_ids[segment_index]
                source_chain = segment.atoms[0].source_chain_id if segment.atoms else ""
                auth_chain = generated_chain if auth_chain_source == "generated" else source_chain
                residue_label_ids = {
                    key: str(index)
                    for index, key in enumerate(segment.residue_keys, start=1)
                }
                for atom in segment.atoms:
                    atom_id += 1
                    residue_key = (atom.residue_number, atom.insertion_code, atom.residue_name)
                    row = [
                        atom.record, atom_id, element_symbol(atom.element), atom.atom_name,
                        atom.alt_id or ".", atom.residue_name, generated_chain, "?",
                        residue_label_ids[residue_key], atom.insertion_code or "?",
                        format_number(atom.x), format_number(atom.y), format_number(atom.z),
                        atom.occupancy, atom.b_factor, atom.charge or "?", atom.residue_number,
                        atom.residue_name, auth_chain, atom.atom_name, model.number,
                    ]
                    handle.write(" ".join(cif_value(value) for value in row) + "\n")
        handle.write("#\n")
    return atom_id


def write_mapping(path: Path, segments: Iterable[Segment], prefix: str, width: int) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow([
            "ter_segment", "generated_chain_id", "source_chain_id", "residue_count", "atom_count",
        ])
        for index, segment in enumerate(segments, start=1):
            writer.writerow([
                index,
                make_chain_id(index, prefix, width),
                ",".join(segment.source_chain_ids) or "(blank)",
                len(segment.residue_keys),
                len(segment.atoms),
            ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert TER-delimited PDB coordinates into PDBx/mmCIF with unique chain IDs."
    )
    parser.add_argument("input_pdb", type=Path, help="input PDB file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="output PDBx/mmCIF (.cif) file")
    parser.add_argument(
        "--chain-count", type=int, default=None,
        help="expected number of TER-delimited chains per model (validation only)",
    )
    parser.add_argument(
        "--chain-length", type=int, default=None,
        help="expected number of unique residues per TER-delimited chain (validation only)",
    )
    parser.add_argument("--chain-prefix", default="C", help="prefix for generated chain IDs (default: C)")
    parser.add_argument(
        "--chain-width", type=int, default=3,
        help="zero-padded numeric width in generated IDs (default: 3; C001, C002, ...)",
    )
    parser.add_argument(
        "--auth-chain-source", choices=("generated", "original"), default="generated",
        help="write generated IDs (default) or original PDB IDs to auth_asym_id",
    )
    parser.add_argument(
        "--mapping-output", type=Path, default=None,
        help="CSV mapping output (default: output-name.chains.csv)",
    )
    parser.add_argument(
        "--block-name", default=None,
        help="mmCIF data-block name (default: output file stem)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.chain_count is not None and args.chain_count < 1:
        raise SystemExit("--chain-count must be positive")
    if args.chain_length is not None and args.chain_length < 1:
        raise SystemExit("--chain-length must be positive")
    if args.chain_width < 1:
        raise SystemExit("--chain-width must be positive")
    if not re.fullmatch(r"[A-Za-z0-9]+", args.chain_prefix):
        raise SystemExit("--chain-prefix must contain only letters and digits")
    if not args.input_pdb.is_file():
        raise SystemExit(f"input file does not exist or is not a file: {args.input_pdb}")
    if args.output.resolve() == args.input_pdb.resolve():
        raise SystemExit("output must not overwrite the input PDB file")

    try:
        models, crystallography = parse_pdb(args.input_pdb)
        validate_models(models, args.chain_count, args.chain_length)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mapping_path = args.mapping_output or args.output.with_suffix(".chains.csv")
        requested_block_name = args.block_name or args.output.stem
        # CIF data-block headers cannot contain whitespace or quoting.  Keep the
        # requested name recognizable while making any file name safe to use.
        block_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", requested_block_name).strip("._-")
        if not block_name:
            block_name = "converted_structure"
        atom_count = write_mmcif(
            args.output, models, crystallography, args.chain_prefix, args.chain_width,
            args.auth_chain_source, block_name,
        )
        write_mapping(mapping_path, models[0].segments, args.chain_prefix, args.chain_width)
    except (OSError, PdbParseError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(
        f"Wrote {args.output} ({atom_count} coordinate rows, {len(models)} model(s), "
        f"{len(models[0].segments)} TER-delimited chains/model)."
    )
    print(f"Wrote chain mapping: {mapping_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
