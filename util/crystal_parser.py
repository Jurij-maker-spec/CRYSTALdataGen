import re
import numpy as np
from pathlib import Path
from ase.io import read
from pymatgen.core.periodic_table import Element


def fix_label_case(label):
    """Convert an atomic label like FE1 -> Fe1 or SI2 -> Si."""
    match = re.match(r"([A-Za-z]+)([0-9]*)", label)
    if not match:
        return label  # leave unchanged if not matching expected pattern
    elem, idx = match.groups()
    # Only capitalize first letter, lowercase the rest (Fe, Si, Co, etc.)
    elem_fixed = elem.capitalize()
    return elem_fixed


def species_to_Z(symbol):
    # Remove oxidation/charge decorations (C4-, B+, Fe2+, ...)
    pure = "".join([c for c in symbol if c.isalpha()])
    return Element(pure).Z


def extract_from_outfile(filename):
    """
    Extracts frequencies (cm^-1), IR intensities (KM/MOL),
    degeneracies, and irreducible representations (irreps)
    from a CRYSTAL output file.
    """
    pattern = re.compile(
        r'\s*(\d+)-\s*(\d+)\s+[0-9.E+-]+\s+([0-9.]+)\s+[0-9.]+\s+\(\s*([A-Za-z0-9]+)\s*\)\s+\w\s+\(\s*([0-9.E+-]+)\)'
    )
    freqs, all_freqs, intensities, degeneracies, irreps = [], [], [], [], []
    with open(filename, 'r') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                start_mode = int(match.group(1))
                end_mode = int(match.group(2))
                freq = float(match.group(3))
                all_freqs.append(freq)
                irrep = match.group(4)
                intensity = float(match.group(5))
                degeneracy = end_mode - start_mode + 1
                if intensity != 0.0:
                    freqs.append(freq)
                    intensities.append(intensity)
                    degeneracies.append(degeneracy)
                    irreps.append(irrep)
    freqs = np.array(freqs)
    intensities = np.array(intensities)
    degeneracies = np.array(degeneracies)
    irreps = np.array(irreps)
    if freqs.size == 0:
        print("⚠️ No nonzero intensities found — check your file format or regex pattern.")
    return freqs, all_freqs, intensities, degeneracies, irreps


class CrystalOutputParser:

    def __init__(self, filepath):
        self.filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            self.text = f.read()
        self.lines = self.filepath.read_text().splitlines()
        self.N_ATOMS = None

    def get_energy(self):
        for line in reversed(self.lines):
            if "TOTAL ENERGY" in line:
                # print('energy: success')
                energy = float(line.split()[3])  # Hartree
                return  energy
            
        raise ValueError("Energy not found")
    
    # replace get_energy() with the following method:
    def get_total_energy_hartree(self):
        """
        Parse the final total energy in Hartree from a CRYSTAL output.

        Priority:
        1) TOTAL ENERGY(DFT)(AU) line
        2) SCF ENDED ... E(AU) line
        """
        patterns = [
            r"TOTAL ENERGY\(DFT\)\(AU\)\(\s*\d+\)\s*([-\d.E+]+)",
            r"SCF ENDED\s*-\s*CONVERGENCE ON ENERGY\s*E\(AU\)\s*([-\d.E+]+)",
        ]

        for pattern in patterns:
            matches = re.findall(pattern, self.text)
            if matches:
                return float(matches[-1])

        raise ValueError(f"Could not parse total energy from {self.filepath}")

    def get_atomic_number_from_0d_input(self):
        """
        For isolated-atom CRYSTAL jobs, parse the atomic number from the geometry block.
        Expects a line like:
            13 0.0 0.0 0.0
        after the line containing the number of atoms (= 1).

        This is a simple heuristic intended for your atomic-reference files.
        """
        lines = self.text.splitlines()

        for i, line in enumerate(lines):
            parts = line.strip().split()

            # Look for a line that is exactly "1" = number of atoms in 0D input
            if len(parts) == 1 and parts[0] == "1":
                # Search a few lines forward for "Z x y z"
                for j in range(i + 1, min(i + 8, len(lines))):
                    p = lines[j].strip().split()
                    if len(p) == 4:
                        try:
                            z = int(float(p[0]))
                            float(p[1]); float(p[2]); float(p[3])
                            return z
                        except ValueError:
                            pass

        raise ValueError(f"Could not parse atomic number from {self.filepath}")

    def get_forces(self):
        forces = []
        reading = False

        for line in self.lines:
            if "CARTESIAN FORCES" in line:
                reading = True
                continue
            

            if reading:
                stripped = line.strip()
            
                if 'ATOM' in stripped:
                    continue
                if stripped == "" :
                    break
                parts = line.split()
                
                forces.append([float(parts[-3]),
                               float(parts[-2]),
                               float(parts[-1])])
        forces = np.array(forces)
        if self.N_ATOMS is not None:
            if forces.shape == (self.N_ATOMS, 3):
                # print('forces: success')
                return forces
            else:
                print('forces: fail')
                return np.zeros((self.N_ATOMS, 3))
        else:
            print('forces: fail')
            return 0    

    def get_stress(self):
        for i, line in enumerate(self.lines):
            if "STRESS TENSOR" in line:
                stress = []
                for j in range(1, 4):
                    stress.append(
                        [float(x) for x in self.lines[i+j].split()[:3]]
                    )
                return np.array(stress)
        raise ValueError("Stress not found")

    def get_lattice(self):
        for i, line in enumerate(self.lines):
            if "DIRECT LATTICE VECTORS CARTESIAN COMPONENTS" in line:
                lattice = []
                for j in range(1, 4):
                    lattice.append([float(x) for x in self.lines[i+1+j].split()])

                lattice = np.array(lattice)

                # if lattice.shape == (3, 3):
                    # print('lattice: success')
                return lattice
        raise ValueError("Lattice not found")

    def get_positions_and_species(self):
        positions = []
        atomic_numbers = []
        reading = False

        i  = 0

        for line in self.lines:

            # Start condition
            if "CARTESIAN COORDINATES" in line:
                reading = True
                continue

            if reading:

                stripped = line.strip()
                # Skip header / separator lines
                if stripped.startswith("*") or "ATOM" in stripped:
                    continue

                parts = stripped.split()
                # Stop if line does not look like atom line
                # Valid atom line must start with integer index
                if stripped == '' or not parts[0].isdigit():
                    # print(f'found {i} Atoms')
                    self.N_ATOMS = i
                    break
                
                # Parse
                try:
                    atomic_numbers.append(int(parts[1]))
                    positions.append([
                        float(parts[3]),
                        float(parts[4]),
                        float(parts[5])
                    ])
                    i +=1
                except (ValueError, IndexError):
                    break


        return np.array(positions), np.array(atomic_numbers), self.N_ATOMS

    def scf_converged(self):
        """
        Returns True if SCF finished properly.
        """
        text = "\n".join(self.lines)

        if "SCF ENDED" in text:
            return True

        if "CONVERGENCE ON ENERGY" in text:
            return True

        return False
    
    def get_born_charges(self):
        born = []
        reading = False

        species = []
        n_atoms = 0
        for i, line in enumerate(self.lines):
            if "ATOMIC BORN CHARGE TENSOR" in line:
                reading = True
                continue
            if reading:
                line_split = line.split()
                if 'ATOM' in line:
                    n_atoms += 1
                    atom = fix_label_case(line_split[2])
                    Z = species_to_Z(atom)
                    species.append(Z)
                    continue

                if len(line_split) == 0:
                    continue
                if line_split[0] == '1' and line_split[1] == '2':
                    continue

                if line_split[0] == '1' or line_split[0] == '2' or line_split[0] == '3':
                    parts = line.split()
                    if len(parts) >= 4:
                        born.append([float(parts[-3]),
                                    float(parts[-2]),
                                    float(parts[-1])])
                if "+++" in line.split():
                    break
                        

        born = np.array(born)
        species = np.array(species)
        # reshape to (Nat,3,3)
        if len(species) == n_atoms:
            print('born charges: success')
            born_reshape = born.reshape(n_atoms, 3, 3)
            return born_reshape, species
        else:
            print('born charges: fail')
            return 0, 0

    def get_dielectric_tensor(self):
        """
        Parse dielectric tensor from CRYSTAL freq output.

        Current target:
        SUM TENSOR OF THE VIBRATIONAL CONTRIBUTIONS TO THE STATIC DIELECTRIC TENSOR
        """
        import re
        import numpy as np

        float_re = re.compile(
            r"^[\s]*([+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)"
            r"\s+([+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)"
            r"\s+([+-]?\d+(?:\.\d*)?(?:[EeDd][+-]?\d+)?)\s*$"
        )

        for i, line in enumerate(self.lines):
            normalized = " ".join(line.upper().split())

            if (
                "SUM TENSOR OF THE VIBRATIONAL CONTRIBUTIONS" in normalized
                and "DIELECTRIC" in normalized
                and "TENSOR" in normalized
            ):
                tensor = []

                for candidate in self.lines[i + 1 : i + 20]:
                    m = float_re.match(candidate.replace("D", "E").replace("d", "e"))
                    if m:
                        tensor.append([float(m.group(1)), float(m.group(2)), float(m.group(3))])
                        if len(tensor) == 3:
                            return np.asarray(tensor, dtype=float)

                raise ValueError("Dielectric tensor header found, but no 3x3 tensor parsed")

        raise ValueError("Dielectric tensor not found")

    def get_phonon_frequencies(self):
        # freqs, intensities, degeneracies, irreps = extract_from_outfile(self)
        return extract_from_outfile(self.filepath)


