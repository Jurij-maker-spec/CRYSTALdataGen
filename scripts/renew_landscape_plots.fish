#!/home/jha/.local/bin/fish
conda activate mace_env

python plot_ref_db_sweep.py --structure SiO2 --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/SiO2 --plot landscape
python plot_ref_db_sweep.py --structure SiO2_PBE --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/SiO2 --plot landscape

python plot_ref_db_sweep.py --structure AlN --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/AlN --plot landscape
python plot_ref_db_sweep.py --structure AlN_PBE --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/AlN --plot landscape

python plot_ref_db_sweep.py --structure Al2O3 --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/Al2O3 --plot landscape
python plot_ref_db_sweep.py --structure Al2O3_PBE --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/Al2O3 --plot landscape

python plot_ref_db_sweep.py --structure TiO2_rutil --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/TiO2 --plot landscape
python plot_ref_db_sweep.py --structure TiO2_rutil_PBE --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/TiO2 --plot landscape

