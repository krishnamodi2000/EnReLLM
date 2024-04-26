# 9302 Research Project Experiment

## Overview

This repository contains the code for an experiment evaluating the effectiveness of Large Language Models (LLMs) in comparison to traditional Recommender System (RS) libraries for movie recommendations. The experiment focuses on specific genres and a subset of users from the MovieLens database.

## Requirements

To run the experiment, you'll need the following:

- Python (version 3.9)
- Required Python libraries (listed in requirements.txt)
- MovieLens dataset (downloaded and stored in data folder)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://git.cs.dal.ca/krishnam/researchproject9302.git 
   ```

2. Install the required Python libraries:

   ```bash
   pip install -r requirements.txt
   ```

## Replicating the Experiment

To replicate the experiment, follow these steps:

1. Navigate to the cloned repository directory:

   ```bash
   cd researchproject9302
   ```

2. Run the main experiment script:

   ```bash
   python Top10moviesRec.py
   ```

3. The experiment will generate evaluation metrics such as MAE, RMSE, and overlap percentage for rating and ranking comparisons.

## Results and Analysis

The experiment results and analysis are stored in the 'results' folder within the repository. Refer to the evaluation.csv files for detailed metrics and insights.

To create some graphs for visually analysing the evaluation.csv simply run the code cells in the presented order in '9302Analysis.ipynb' file. Feel free to create more graphs in your local machine for further analysis.

## Contact Information

For any questions or issues regarding the experiment or code, please contact Krishna Modi at [kr733081@dal.ca].