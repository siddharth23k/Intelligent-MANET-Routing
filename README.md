# ML-Driven Link Stability Prediction for Proactive Routing in MANET

## Overview
This project proposes a hybrid Machine Learning and Genetic Algorithm framework to improve routing reliability in Mobile Ad-hoc Networks (MANET).

## Components
1. NS-3 MANET simulation
2. Network dataset generation
3. Machine learning models
4. Ensemble prediction
5. Genetic algorithm routing optimization

## Technologies
- NS-3 Network Simulator
- Python
- Scikit-learn
- NetAnim
 
## Workflow
Simulation → Dataset → ML Training → Link Prediction → GA Routing

## Pipeline
1. **Simulate MANET topology**: Run the NS-3 script `simulation/manet-sim.cc` to generate the link dataset at `dataset/link_dataset.csv` and the NetAnim file `manet-animation.xml`.
2. **Explore the dataset**: Use `ml/plot_results.py` or the notebook `notebooks/01_explore_dataset.ipynb` to inspect distance and link-stability distributions.
3. **Train ML models**: Train the Random Forest and Neural Network link predictors with `ml/train_model.py` or `notebooks/02_train_and_evaluate_models.ipynb`, which save models to the `results/` directory.
4. **Evaluate ensemble**: Run `ml/ensemble_model.py` to compute the ensemble performance of the trained models on a held-out test set.
5. **Optimize routes with GA**: Use `ml/ga_routing.py` or `notebooks/03_ga_routing_demo.ipynb` to search for stable routes between a chosen source and destination using the learned link-stability model.

## Getting Started

1. **Create and activate a virtual environment (recommended)**  
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the NS-3 simulation**  
   Build and run `simulation/manet-sim.cc` inside your NS-3 setup to generate `dataset/link_dataset.csv` and `manet-animation.xml`.

## How to Run Experiments

1. **Dataset exploration**
   - Script: `python -m ml.plot_results`
   - Notebook: open `notebooks/01_explore_dataset.ipynb` in Jupyter and run all cells.

2. **Train ML models**
   - Script: `python -m ml.train_model`
   - Notebook: open `notebooks/02_train_and_evaluate_models.ipynb`.

3. **Evaluate ensemble**
   - Script: `python -m ml.ensemble_model`

4. **Run GA-based routing**
   - Script: `python -m ml.ga_routing`
   - Notebook: open `notebooks/03_ga_routing_demo.ipynb`.

All trained models and metrics are saved under the `results/` directory, and figures are written to `figures/`.

## Results
Random Forest Accuracy: 1.00  
Neural Network Accuracy: 0.998