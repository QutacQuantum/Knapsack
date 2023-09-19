# Annealing

This directory provides the code the plotting the results of the annealing approaches.
* plots_annealing.ipynb contains most of the plots
* plots_qubits_needed_by_dwave contains the code needed to visualize the number of qubits needed by the D-Wave devices
* results_annealing.csv contains all annealing experiment data

The code used to generate these results can be found in QUARK folder. After installing the pip packages you can start the process like ``python src/main.py --config paper_config/IHS_config.yml``.
``paper_config/IHS_config.yml`` was used for the IHS results, while ``paper_config/annealing_config.yml`` was used for the normal annealing results.
Note that the DWave results probably cannot be reproduced as the Dwave devices have been removed from Amazon Braket in November 2022.