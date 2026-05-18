#!/usr/bin/env python3

import h5py
import numpy as np
from pathlib import Path
import sys
import re
from ase.io import read
import argparse
from datetime import datetime

# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------
THIS_FILE = Path(__file__).resolve()
ROOT = THIS_FILE.parent

while not (ROOT / "util").exists():
    if ROOT.parent == ROOT:
        raise RuntimeError("Could not find project root containing 'util'")
    ROOT = ROOT.parent

sys.path.insert(0, str(ROOT))

from util.crystal_parser import CrystalOutputParser
from util.mark_outliers import mark_outliers

DEFAULT_STRUCTURES_PATTERN = "structures"
H5_PATH = ROOT / "data" / "train_db.h5"

ATOMIC_OUTPUTS = None

HARTREE_TO_EV = 27.211386245988
FORCE_CONV = 51.422067  # Ha/Bohr -> eV/Angstrom

# ============================================================
# HELPERS
# ============================================================

def extract_distortion_id(outfile: Path):
    """
    Extract distortion ID information from parent folder names like:
        010042
        000123
        021999

    Returns
    -------
    dict
        {
            "raw": "010042",
            "prefix": "010",
            "local_index": 42,
            "full_index": 10042,
        }

    or None if nothing suitable was found.
    """
    for parent in [outfile.parent] + list(outfile.parents):
        name = parent.name

        # exactly 6 digits, e.g. 010042
        if re.fullmatch(r"\d{6}", name):
            raw = name
            prefix = raw[:3]
            local_index = int(raw[3:])
            full_index = int(raw)

            return {
                "raw": raw,
                "prefix": prefix,
                "local_index": local_index,
                "full_index": full_index,
            }

    return None


def write_string_dataset(group, name, value):
    group.create_dataset(name, data=np.bytes_(str(value)))


def record_failure(failed_group, distortion_key, outfile, reason, error_msg="", distortion_info=None, overwrite=False):
    if distortion_key in failed_group:
        if overwrite:
            del failed_group[distortion_key]
        else:
            return
    g = failed_group.create_group(distortion_key)
    write_string_dataset(g, "source_file", str(outfile))
    write_string_dataset(g, "source_name", outfile.stem if outfile is not None else "")
    write_string_dataset(g, "reason", reason)
    write_string_dataset(g, "error_message", error_msg)

    if distortion_info is not None:
        write_string_dataset(g, "distortion_id_raw", distortion_info["raw"])
        write_string_dataset(g, "distortion_prefix", distortion_info["prefix"])
        g.create_dataset("distortion_local_index", data=int(distortion_info["local_index"]))
        g.create_dataset("distortion_full_index", data=int(distortion_info["full_index"]))


def overwrite_group(parent, name):
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def safe_decode_array(arr):
    """
    Convert string/object arrays to fixed-width UTF-8 compatible arrays for HDF5.
    """
    arr = np.asarray(arr)
    if arr.dtype.kind in {"U", "O"}:
        return arr.astype("S")
    return arr


def sort_singlepoint_outs_by_id(outfiles):
    def keyfunc(p):
        info = extract_distortion_id(p)
        if info is None:
            return (999999999, str(p))
        return (info["full_index"], str(p))
    return sorted(outfiles, key=keyfunc)


def find_single_outfile(folder: Path):
    if not folder.exists():
        return None

    outfiles = [
        f for f in folder.glob("*.out")
        if "slurm" not in f.name.lower()
    ]

    if not outfiles:
        return None

    if len(outfiles) == 1:
        return outfiles[0]

    preferred = [
        f for f in outfiles
        if "_freq" in f.name.lower()
        or "_geoopt" in f.name.lower()
        or "_sp" in f.name.lower()
    ]

    if preferred:
        return sorted(preferred, key=lambda p: p.stat().st_mtime)[-1]

    return sorted(outfiles, key=lambda p: p.stat().st_mtime)[-1]


def find_clean_cif(folder):
    if not folder.exists():
        return None
    try:
        file = [f for f in folder.glob("*geoopt_clean.cif")]
    except Exception:
        print('No clean cif found')
        return None
    
    return file[0]


def find_geoopt_out(struct_dir: Path):
    geoopt_out = find_single_outfile(struct_dir / "geoopt")
    clean_cif = find_clean_cif(struct_dir)
    return geoopt_out, clean_cif


def find_freq_out(struct_dir: Path):
    return find_single_outfile(struct_dir / "freq")


def find_singlepoint_outs(struct_dir: Path):
    """
    Find CRYSTAL single-point output files in directories containing:
        - "mixed"
        - and ("single_points" or "sp")

    Returns
    -------
    list[Path]
        Sorted list of *.out files.
    """
    disto_dir = struct_dir / 'disto'

    candidate_dirs = []

    for d in disto_dir.rglob("*"):
        if not d.is_dir():
            continue

        name = d.name.lower()

        if (
            "singlepoints" in name or "sp" in name
        ):
            candidate_dirs.append(d)

    # print(candidate_dirs)
    outfiles = []

    for d in candidate_dirs:

        outs = [
            f for f in d.rglob("*.out")
            if not f.name.startswith("slurm")
        ]

        outfiles.extend(outs)

    return sorted(outfiles)


def parser_ok(parser, outfile: Path):
    try:
        if not parser.scf_converged():
            print(f"  skipping {outfile.name}: SCF not converged")
            return False
    except Exception as e:
        print(f"  warning: could not verify SCF convergence for {outfile.name}: {e}")
        return False
    return True


def load_reference_from_structure_file(cifpath):
    atoms = read(cifpath)
    positions = atoms.get_positions()
    numbers = atoms.get_atomic_numbers()
    cell = atoms.cell.array
    return positions, numbers, cell


def parse_structure_common(outfile: Path, is_reference=False, clean_cif=None):
    """
    Parse standard structure information from a CRYSTAL output file.

    Returns
    -------
    success : bool
    data_or_reason : dict | str
    error_msg : str
    """
    try:
        parser = CrystalOutputParser(outfile)
    except Exception as e:
        return False, "parser_init_failed", str(e)

    try:
        if not parser.scf_converged():
            return False, "scf_not_converged", ""
    except Exception as e:
        return False, "scf_check_failed", str(e)

    try:
        if is_reference:
            positions, atomic_numbers, lattice = load_reference_from_structure_file(clean_cif)
        else:
            positions, atomic_numbers, _ = parser.get_positions_and_species()
            lattice = parser.get_lattice()
    except Exception as e:
        return False, "positions_species_failed", str(e)

    try:
        lattice = parser.get_lattice()
    except Exception as e:
        return False, "lattice_failed", str(e)

    try:
        energy = parser.get_total_energy_hartree() * HARTREE_TO_EV
    except Exception as e:
        return False, "energy_failed", str(e)

    try:
        forces = parser.get_forces() * FORCE_CONV
    except Exception as e:
        print(f"  warning: no forces in {outfile.name}: {e}")
        forces = np.zeros((len(atomic_numbers), 3), dtype=float)

    try:
        stress = parser.get_stress()
    except Exception as e:
        # print(f"  warning: no stress in {outfile.name}: {e}")
        stress = np.zeros((3, 3), dtype=float)

    data = {
        "positions": np.asarray(positions, dtype=float),
        "atomic_numbers": np.asarray(atomic_numbers, dtype=int),
        "lattice": np.asarray(lattice, dtype=float),
        "dft_forces": np.asarray(forces, dtype=float),
        "stress": np.asarray(stress, dtype=float),
        "energy": float(energy),
    }
    return True, data, ""


def write_common_structure_group(group, data: dict):
    group.create_dataset("positions", data=data["positions"])
    group.create_dataset("atomic_numbers", data=data["atomic_numbers"])
    group.create_dataset("lattice", data=data["lattice"])
    group.create_dataset("dft_forces", data=data["dft_forces"])
    group.create_dataset("stress", data=data["stress"])
    group.create_dataset("energy", data=data["energy"])


def parse_and_write_reference(struct_group, geoopt_out: Path, clean_cif, is_reference):
    ok, result, err = parse_structure_common(geoopt_out, is_reference, clean_cif)
    if not ok:
        print(f"  failed reference parse: {geoopt_out.name} | {result} | {err}")
        return False

    ref_group = overwrite_group(struct_group, "reference")
    write_common_structure_group(ref_group, result)
    ref_group.attrs["source_file"] = str(geoopt_out)
    return True


def parse_and_write_primitive_reference(struct_group, geoopt_out: Path, freq_out: Path):
    if geoopt_out is None or freq_out is None:
        return False

    prim_group = overwrite_group(struct_group, "primitive_reference")

    # geometry from geoopt
    ok, geo_data, err = parse_structure_common(geoopt_out)
    if not ok:
        print(f"  warning: could not build primitive_reference geometry from {geoopt_out.name}: {geo_data} | {err}")
        return False

    prim_group.create_dataset("positions", data=geo_data["positions"])
    prim_group.create_dataset("atomic_numbers", data=geo_data["atomic_numbers"])
    prim_group.create_dataset("lattice", data=geo_data["lattice"])

    # vibrational data from freq
    parser = CrystalOutputParser(freq_out)
    if not parser_ok(parser, freq_out):
        print(f"  warning: freq file not usable: {freq_out.name}")
        return False

    try:
        born, species = parser.get_born_charges()
        prim_group.create_dataset("born_charges", data=np.asarray(born, dtype=float))
        prim_group.create_dataset("born_species", data=safe_decode_array(species))
    except Exception as e:
        print(f"  warning: no born charges in {freq_out.name}: {e}")

    try:
        freqs, all_freqs, intensities, degeneracies, irreps = parser.get_phonon_frequencies()
        prim_group.create_dataset("optical_phonon_frequencies", data=np.asarray(freqs, dtype=float))
        prim_group.create_dataset("all_phonon_frequencies", data=np.asarray(all_freqs, dtype=float))
        prim_group.create_dataset("intensities", data=np.asarray(intensities, dtype=float))
        prim_group.create_dataset("degeneracies", data=np.asarray(degeneracies, dtype=int))
        prim_group.create_dataset("irreps", data=safe_decode_array(irreps))
    except Exception as e:
        print(f"  warning: no phonon data in {freq_out.name}: {e}")

    prim_group.attrs["geoopt_source_file"] = str(geoopt_out)
    prim_group.attrs["freq_source_file"] = str(freq_out)
    return True


def parse_and_write_distortions(struct_group, singlepoint_outs, overwrite=False):
    distortions_group = struct_group.require_group("distortions")
    failed_group = struct_group.require_group("failed_distortions")

    n_found = len(singlepoint_outs)
    n_written = 0
    n_skipped = 0
    n_failed = 0

    used_keys = set(distortions_group.keys()) | set(failed_group.keys())

    for outfile in singlepoint_outs:
        distortion_info = extract_distortion_id(outfile)

        if distortion_info is not None:
            distortion_key = distortion_info["raw"]
        else:
            distortion_key = f"unknown_{n_failed + n_written + n_skipped:06d}"

        if distortion_key in distortions_group:
            if overwrite:
                del distortions_group[distortion_key]
            else:
                n_skipped += 1
                continue

        if distortion_key in failed_group:
            if overwrite:
                del failed_group[distortion_key]
            else:
                n_skipped += 1
                continue

        # prevent duplicate keys during same run
        if distortion_key in used_keys and overwrite:
            suffix = 1
            base_key = distortion_key
            new_key = f"{base_key}_dup{suffix}"
            while new_key in used_keys:
                suffix += 1
                new_key = f"{base_key}_dup{suffix}"
            distortion_key = new_key

        used_keys.add(distortion_key)

        ok, result, err = parse_structure_common(outfile)

        if not ok:
            record_failure(
                failed_group=failed_group,
                distortion_key=distortion_key,
                outfile=outfile,
                reason=result,
                error_msg=err,
                distortion_info=distortion_info,
                overwrite=overwrite,
            )
            n_failed += 1
            continue

        g = distortions_group.create_group(distortion_key)
        write_common_structure_group(g, result)
        g.attrs["source_file"] = str(outfile)
        g.attrs["source_name"] = outfile.stem

        if distortion_info is not None:
            g.attrs["distortion_id_raw"] = distortion_info["raw"]
            g.attrs["distortion_prefix"] = distortion_info["prefix"]
            g.attrs["distortion_local_index"] = int(distortion_info["local_index"])
            g.attrs["distortion_full_index"] = int(distortion_info["full_index"])

        n_written += 1

    distortions_group.attrs["n_distortions"] = len(distortions_group)
    failed_group.attrs["n_failed"] = len(failed_group)

    struct_group.attrs["n_singlepoints_found_last_scan"] = n_found
    struct_group.attrs["n_distortions_written_last_scan"] = n_written
    struct_group.attrs["n_distortions_skipped_last_scan"] = n_skipped
    struct_group.attrs["n_distortions_failed_last_scan"] = n_failed
    struct_group.attrs["n_distortions_total"] = len(distortions_group)
    struct_group.attrs["n_failed_distortions_total"] = len(failed_group)

    return {
        "found": n_found,
        "written": n_written,
        "skipped": n_skipped,
        "failed": n_failed,
        "total": len(distortions_group),
    }


def write_atomic_energies(h5):
    atomic_group = overwrite_group(h5, "atomic_energies")

    if not ATOMIC_OUTPUTS:
        print("Atomic energies: skipped (ATOMIC_OUTPUTS empty)")
        return

    n_written = 0
    n_missing = 0
    n_failed = 0

    for z, atom_outfile in ATOMIC_OUTPUTS.items():
        if not atom_outfile.exists():
            n_missing += 1
            continue

        parser = CrystalOutputParser(atom_outfile)

        if not parser_ok(parser, atom_outfile):
            n_failed += 1
            continue

        try:
            e_hartree = parser.get_total_energy_hartree()
        except Exception:
            try:
                e_hartree = parser.get_energy()
            except Exception:
                n_failed += 1
                continue

        e_ev = e_hartree * HARTREE_TO_EV

        z_group = atomic_group.create_group(str(z))
        z_group.create_dataset("energy_hartree", data=float(e_hartree))
        z_group.create_dataset("energy_ev", data=float(e_ev))
        z_group.attrs["source_file"] = str(atom_outfile)
        n_written += 1

    print(
        f"Atomic energies: written={n_written}, missing={n_missing}, failed={n_failed}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Build/update CRYSTAL train DB.")

    parser.add_argument(
        "--db",
        type=Path,
        default=H5_PATH,
        help="Output HDF5 train database.",
    )

    parser.add_argument(
        "--structures-root",
        type=Path,
        default=None,
        help=(
            "Optional single structures directory. "
            "If omitted, all directories under ROOT containing 'structures' are scanned."
        ),
    )

    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional structure names to process.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing structure groups completely.",
    )

    parser.add_argument(
        "--no-outliers",
        action="store_true",
        help="Skip mark_outliers() after DB build.",
    )

    return parser.parse_args()


def discover_structure_roots(root: Path, explicit_root: Path | None = None) -> list[Path]:
    """
    Find directories like:
        structures
        structures_pbe
        structures_pbesol

    If explicit_root is given, use only that directory.
    """
    if explicit_root is not None:
        explicit_root = explicit_root.resolve()
        if not explicit_root.exists():
            raise FileNotFoundError(f"Missing structures root: {explicit_root}")
        if not explicit_root.is_dir():
            raise NotADirectoryError(f"Not a directory: {explicit_root}")
        return [explicit_root]

    roots = sorted(
        p for p in root.iterdir()
        if p.is_dir() and "structures" in p.name.lower()
    )

    if not roots:
        raise FileNotFoundError(
            f"No directories containing 'structures' found under {root}"
        )

    return roots


def structure_db_key(struct_dir: Path, structures_root: Path) -> str:
    """
    Avoid collisions between:
        structures/SiO2
        structures_pbe/SiO2

    If the structure folder already contains a suffix, keep it.
    Otherwise append suffix from parent root for non-default roots.
    """
    name = struct_dir.name
    root_name = structures_root.name.lower()

    if root_name == "structures":
        return name

    suffix = root_name.replace("structures", "").strip("_")

    if not suffix:
        return name

    suffix_upper = suffix.upper()

    if name.upper().endswith(f"_{suffix_upper}"):
        return name

    return f"{name}_{suffix_upper}"


# ============================================================
# MAIN
# ============================================================

def build_dataset(
    h5_path: Path = H5_PATH,
    structures_root: Path | None = None,
    only: list[str] | None = None,
    overwrite: bool = False,
    run_outliers: bool = True,
):
    h5_path = Path(h5_path).resolve()
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    structure_roots = discover_structure_roots(ROOT, explicit_root=structures_root)
    only_set = set(only) if only else None

    summaries = []

    mode = "a"
    with h5py.File(h5_path, mode) as h5:
        structures_group = h5.require_group("structures")

        for root_dir in structure_roots:
            struct_dirs = sorted([p for p in root_dir.iterdir() if p.is_dir()])

            print("=" * 72)
            print(f"Scanning structure root: {root_dir}")
            print("=" * 72)

            for struct_dir in struct_dirs:
                source_name = struct_dir.name
                db_name = structure_db_key(struct_dir, root_dir)

                if only_set is not None and source_name not in only_set and db_name not in only_set:
                    continue

                print("-" * 72)
                print(f"Processing source={source_name} -> db_key={db_name}")
                print("-" * 72)

                if db_name in structures_group:
                    if overwrite:
                        print(f"  overwriting existing structure group: {db_name}")
                        del structures_group[db_name]
                        mat_group = structures_group.create_group(db_name)
                    else:
                        print(f"  updating existing structure group: {db_name}")
                        mat_group = structures_group[db_name]
                else:
                    mat_group = structures_group.create_group(db_name)

                geoopt_out, clean_cif = find_geoopt_out(struct_dir)
                freq_out = find_freq_out(struct_dir)
                singlepoint_outs = find_singlepoint_outs(struct_dir)

                has_reference = "reference" in mat_group
                has_primitive = "primitive_reference" in mat_group

                if geoopt_out is not None and clean_cif is not None:
                    if overwrite or "reference" not in mat_group:
                        print("  writing reference")
                        is_reference = True
                        ok_ref = parse_and_write_reference(
                            mat_group,
                            geoopt_out,
                            clean_cif,
                            is_reference,
                        )
                        has_reference = bool(ok_ref)
                    else:
                        print("  reference exists, skipping")

                if geoopt_out is not None and freq_out is not None:
                    if overwrite or "primitive_reference" not in mat_group:
                        print("  writing primitive_reference")
                        ok_prim = parse_and_write_primitive_reference(
                            mat_group,
                            geoopt_out,
                            freq_out,
                        )
                        has_primitive = bool(ok_prim)
                    else:
                        print("  primitive_reference exists, skipping")

                if singlepoint_outs:
                    dsum = parse_and_write_distortions(
                        mat_group,
                        singlepoint_outs,
                        overwrite=overwrite,
                    )
                else:
                    dsum = {
                        "found": 0,
                        "written": 0,
                        "skipped": 0,
                        "failed": 0,
                        "total": len(mat_group.get("distortions", {})),
                    }
                    mat_group.attrs["n_singlepoints_found_last_scan"] = 0
                    mat_group.attrs["n_distortions_written_last_scan"] = 0
                    mat_group.attrs["n_distortions_skipped_last_scan"] = 0
                    mat_group.attrs["n_distortions_failed_last_scan"] = 0

                mat_group.attrs["structure_name"] = db_name
                mat_group.attrs["source_structure_name"] = source_name
                mat_group.attrs["source_dir"] = str(struct_dir)
                mat_group.attrs["source_structures_root"] = str(root_dir)
                mat_group.attrs["last_updated"] = datetime.now().isoformat(timespec="seconds")

                summaries.append({
                    "name": db_name,
                    "source": source_name,
                    "root": root_dir.name,
                    "reference": has_reference,
                    "primitive": has_primitive,
                    "found": dsum["found"],
                    "written": dsum["written"],
                    "skipped": dsum["skipped"],
                    "failed": dsum["failed"],
                    "total": dsum["total"],
                })

        if overwrite or "atomic_energies" not in h5:
            write_atomic_energies(h5)

        h5.attrs["root"] = str(ROOT)
        h5.attrs["structure_roots"] = ",".join(str(p) for p in structure_roots)
        h5.attrs["energy_unit"] = "eV"
        h5.attrs["force_unit"] = "eV/Angstrom"
        h5.attrs["stress_note"] = "Stored as returned by parser.get_stress()"
        h5.attrs["created_from"] = "CRYSTALdataGen project structure"
        h5.attrs["last_updated"] = datetime.now().isoformat(timespec="seconds")
        h5.attrs["update_mode"] = "overwrite" if overwrite else "incremental"

    print("\nStructure summary")
    print("-" * 96)

    total_found = 0
    total_written = 0
    total_skipped = 0
    total_failed = 0
    total_ref = 0
    total_prim = 0

    for s in summaries:
        ref_flag = "yes" if s["reference"] else "no"
        prim_flag = "yes" if s["primitive"] else "no"

        print(
            f"{s['name']:<20} "
            f"root={s['root']:<15} "
            f"ref={ref_flag:<3} "
            f"prim={prim_flag:<3} "
            f"found={s['found']:<5} "
            f"new={s['written']:<5} "
            f"skip={s['skipped']:<5} "
            f"fail={s['failed']:<5} "
            f"total={s['total']:<5}"
        )

        total_found += s["found"]
        total_written += s["written"]
        total_skipped += s["skipped"]
        total_failed += s["failed"]

        if s["reference"]:
            total_ref += 1

        if s["primitive"]:
            total_prim += 1

    print("-" * 96)
    print(
        f"{'TOTAL':<20} "
        f"ref={total_ref:<3} "
        f"prim={total_prim:<3} "
        f"found={total_found:<5} "
        f"new={total_written:<5} "
        f"skip={total_skipped:<5} "
        f"fail={total_failed:<5}"
    )

    if run_outliers:
        print("\nmarking outliers\n")
        mark_outliers()
    else:
        print("\noutlier marking skipped\n")

    print(f"\nDone. Updated: {h5_path}")


if __name__ == "__main__":
    args = parse_args()

    build_dataset(
        h5_path=args.db,
        structures_root=args.structures_root,
        only=args.only,
        overwrite=args.overwrite,
        run_outliers=not args.no_outliers,
    )
    