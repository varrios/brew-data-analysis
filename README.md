# brew-data-analysis

The **brew-data-analysis** project is a toolkit for analyzing brewing-related data. It enables the exploration, modeling, and visualization of data concerning brewing processes, ingredients, and final product characteristics.

## Table of Contents

- [About The Project](#about-the-project)
- [Analysis Highlights](#analysis-highlights)
- [Directory Structure](#directory-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Authors](#authors)

## About The Project

The goal of this project is to analyze homebrewing data to uncover trends and build predictive models. The analysis is based on a comprehensive public dataset of beer recipes sourced from **Brewer's Friend**.

The project utilizes Python and Jupyter Notebooks to process this data, perform exploratory analysis, build a machine learning model to predict beer styles, and generate insightful visual reports.

## Analysis Highlights

Our analysis is detailed across two primary reports, uncovering key insights from the dataset:

### Report 1: Exploratory Data Analysis (EDA)
This report focuses on cleaning, understanding, and visualizing the core components of the dataset. Key findings include:
-   **Data Cleaning:** Preprocessing steps to handle missing values and standardize data for accurate analysis.
-   **Parameter Distributions:** Visualizations of key beer attributes like Alcohol By Volume (ABV), International Bitterness Units (IBU), and Standard Reference Method (SRM) for color, revealing common ranges and outliers in homebrewing.
-   **Correlation Matrix:** Analysis of the relationships between different brewing parameters (e.g., how Original Gravity relates to ABV).

### Report 2: Predictive Modeling & Style Analysis
This report builds on the EDA by applying machine learning to understand and predict beer styles using two distinct approaches:

-   **Clustering (Unsupervised Analysis):** We first used clustering algorithms to group recipes based on their inherent similarities (e.g., SRM, ABV, IBU) *without* using the pre-defined style labels. This exploratory step helps identify natural groupings within the data and reveals how they align with established categories.
-   **Classification (Supervised Modeling):** A classification model was then trained to predict the official BJCP (Beer Judge Certification Program) style of a beer based on its recipe features. This is the primary predictive task of the project.
-   **Model Evaluation:** The report includes a detailed classification report and a confusion matrix to assess the classification model's performance, identifying which beer styles are most accurately predicted.

## Directory Structure

-   `data/` – Contains the raw and processed Brewer's Friend dataset.
-   `model/` – Holds analytical scripts and trained machine learning models.
-   `raports/` – Stores the Jupyter Notebooks (`raport_1.ipynb`, `report_2.ipynb`) and generated visualizations.
-   `utility/` – Includes helper functions for data loading and cleaning.
-   `main.py` – The main script to execute the full analysis pipeline.
-   `constants.py` – Stores constants and configuration settings for the project.

## Prerequisites

-   Python 3.7+
-   A `requirements.txt` file is provided. Key libraries include:
    -   pandas
    -   numpy
    -   matplotlib
    -   seaborn
    -   scikit-learn
    -   jupyter

## Installation

1.  Clone the repository:
    ```
    git clone https://github.com/varrios/brew-data-analysis.git
    ```
2.  Navigate to the project directory:
    ```
    cd brew-data-analysis
    ```
3.  Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

-   To run the main analysis script from your terminal:
    ```
    python main.py
    ```
-   For interactive analysis, you can run the Jupyter Notebooks located in the `reports/` directory:
    ```
    jupyter notebook
    ```
    Then, open `raport_1.ipynb` or `raport_2.ipynb`.

## Features

-   Loading and preprocessing of the Brewer's Friend dataset.
-   Exploratory Data Analysis (EDA) and rich visualization of key brewing parameters.
-   Building and evaluating predictive models for clustering and classifying beer styles.
-   Generating summary reports and visualizations to communicate findings.

## Authors

-   [@Anna Sztukowska](https://github.com/sztvk)
-   [@Michał Sugalski](https://github.com/mikkelangelas)
-   [@Lucjan Gackowski](https://github.com/varrios)

