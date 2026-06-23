#!/home/jha/.local/bin/fish
conda activate mace_env 

./general_plots.py top-n-ir --structure AlN --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/AlN
./general_plots.py top-n-ir --structure AlN_PBE --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/AlN
./general_plots.py top-n-ir --structure Al2O3 --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/Al2O3
./general_plots.py top-n-ir --structure Al2O3_PBE --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/Al2O3
./general_plots.py top-n-ir --structure SiO2 --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/SiO2
./general_plots.py top-n-ir --structure SiO2_PBE --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/SiO2
./general_plots.py top-n-ir --structure TiO2_rutil --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/TiO2_rutil/
./general_plots.py top-n-ir --structure TiO2_rutil_PBE --top-n 10 --selection spaced --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/TiO2_rutil/
