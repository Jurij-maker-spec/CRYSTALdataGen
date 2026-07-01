#!/home/jha/.local/bin/fish
conda activate mace_env_3_12
 
python general_plots.py frequency-evolution \
                        --structure SiO2 \
                        # --modes ir_active \
                        --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/SiO2 \
                        # --color-by intensity_abs_error
                            

python general_plots.py frequency-evolution \
                        --structure AlN \
                        --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/AlN
python general_plots.py frequency-evolution \
                        --structure Al2O3 \
                        --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/Al2O3
python general_plots.py frequency-evolution \
                        --structure TiO2_rutil \
                        --outdir /home/jha/jha/python_scripts/master_thesis/figures/results/TiO2_rutil \
                        
                            



