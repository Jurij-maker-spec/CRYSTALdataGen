#!/home/jha/.local/bin/fish
conda activate mace_env_3_12

python general_plots.py best-result-figures --structure AlN
python general_plots.py best-result-figures --structure SiO2
python general_plots.py best-result-figures --structure Al2O3
python general_plots.py best-result-figures --structure TiO2_rutil
