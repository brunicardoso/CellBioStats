<p align="center">
  <img src="https://github.com/brunicardoso/CellBioStats/raw/main/assets/logo.png" alt="CellBioStats Logo" width="300"/>
</p>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/brunicardoso/CellBioStats/blob/main/CellBioStats.ipynb)

CellBioStats is an application that simplifies the creation of publication-quality [**SuperPlots**](https://rupress.org/jcb/article/219/6/e202001064/151717/SuperPlots-Communicating-reproducibility-and) and performs  statistical analysis on hierarchical data from common experimental results in **cell biology, molecular biology, and biochemistry**, helping you visualize complex datasets and avoid pitfalls like pseudoreplication.

## Features

* **Interactive SuperPlots**: Generates publication-quality SuperPlots, showing individual data points, replicate means, and overall treatment means with standard error (SEM).
* **Automated Statistical Analysis**: Automatically performs normality checks, variance homogeneity tests, and selects the appropriate statistical test for your data.
* **Hierarchical Statistics**: Avoids pseudoreplication by correctly performing statistical tests on **replicate means**, not on raw technical measurements.
* **Paired & Unpaired Data**: Handles both independent (unpaired) and repeated measures (paired) experimental designs. 
* **Data Upload**: Supports both `.csv` and `.xlsx` file formats.
* **Plot Customization**: Allows customization of plot aesthetics (font size, color schemes, marker size, and plot templates).
* **Downsampling for plot visualization**: Option to display a random subset of your data points to keep plots clean and responsive with very large datasets.
* **Export Results**: Download the detailed statistical summary as a `.txt` file and the plot as a `.png, .jpeg, .svg or .pdf` file directly from the app.

## Getting Started

You can run CellBioStats in three different ways.

### Option 1: Standalone Application (`.exe` for Windows)

This is the easiest method for most users. No installation is required.

1.  Go to the [**dist**](https://github.com/brunicardoso/CellBioStats/tree/main/dist) page of this repository.
2.  Download the latest `.exe` file.
3.  Double-click the file to run the application. 
BE AWARE THAT IT MIGHT BE FLAGGED AS A VIRUS BY SOME ANTI VIRUS
SOFTWARE AND YOU MIGHT NEED TO TEMPORARILY DISABLE THE DEFENSE SYSTEM TO RUN THE APP!.

### Option 2: Google Colab (Cloud-Based)

Run the app in your browser without any local installation.

1.  Click the "Open in Colab" badge at the top of this README.
2.  Run the cells in the notebook to start the application. 

Note that in Colab it won't be possible to download the plots as PNG and JPEG by clicking on the "Download Plot" button. But, you are still going to be able to download the PNG file by clicking on the camera icon. After choosing SVG or PDF extensions, the file can be found in the Colab folder panel.

### Option 3: Run Locally

If you have Python installed, you can run the app from the source code.

1.  **Clone the repository as below or just directly download the app.py file:**
    ```bash
    git clone [https://github.com/brunicardoso/CellBioStats.git](https://github.com/brunicardoso/CellBioStats.git)
    cd CellBioStats
    ```

2.  **Create a virtual environment and install dependencies.** Choose either **Conda** or **venv**.

    **Using Conda:**
    ```bash
    # Create a new conda environment
    conda create --name cellbiostats python=3.10

    # Activate the environment
    conda activate cellbiostats

    # Install the required packages
    pip install -r requirements.txt
    ```

    **Using venv:**
    ```bash
    # Create a virtual environment
    python -m venv venv

    # Activate the environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    venv\Scripts\activate

    # Install the required packages
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    python app.py
    ```

4.  Open your web browser and navigate to `http://127.0.0.1:8050`.

## How to Use the App

1.  **Upload Data**: Click the "Upload Data" button and select your `.csv` or `.xlsx` file. You can try it on our sample data https://github.com/brunicardoso/CellBioStats/raw/main
2.  **Map Columns**: Select the appropriate columns from your file for **Treatment**, **Value**, and **Replicate**.
3.  **Select Test Type**: (Optional) If your experiment uses a repeated measures or paired design, check the "Use paired/repeated statistical tests" box. [*Read this if you are not sure about choosing paired or unpaired tests*](https://rupress.org/jcb/article/219/6/e202001064/151717/SuperPlots-Communicating-reproducibility-and)
4.  **Generate**: Click the "Generate Analysis" button.
5.  **Customize**: Use the sliders and dropdowns in the "Plot Customization" panel to adjust the appearance of your plot.
6.  **Download**: Click the "Download Summary" button to save the statistical report, or use the camera icon on the plot to save it as a PNG image or the "Download Plot" button to save the plot with different resolutions and file extensions (.png, .jpeg, .svg, or .pdf).

## Input Data Format

The application expects your data to be in a specific format, like in the example below. You must have at least three columns:

* **Treatment Column**: Identifies the different experimental groups (e.g., 'Control', 'Drug A').
* **Value Column**: Contains the numeric measurement data (the dependent variable; e.g., cell size, expression level, etc).
* **Replicate Column**: Identifies the independent experimental replicates (e.g., experiment number of independent replications done in different days or animal ID).

Here is an example of a valid data structure:

| Treatment | Cell_size | Replicate |
| :-------- | :-------- | :-------- |
| Control   | 10.5      | 1         |
| Control   | 11.2      | 1         |
| Control   | 12.1      | 2         |
| Control   | 11.8      | 2         |
| Drug A    | 15.3      | 1         |
| Drug A    | 14.9      | 1         |
| Drug A    | 16.5      | 2         |
| Drug A    | 17.1      | 2         |

## Statistical Methodology

CellBioStats is designed to perform statistically sound analysis by respecting the hierarchical nature of typical biological data.

1.  **Data Aggregation**: All primary statistical tests are performed on the **means of each replicate**, not on the raw technical measurements. This avoids pseudoreplication and ensures that the statistical power reflects the number of independent experiments.
2.  **Assumption Checks**:
    * **Normality**: The Shapiro-Wilk test is run on the replicate means for each treatment group.
    * **Homoscedasticity (Equal Variances)**: Levene's test is run to check for equality of variances across groups.
3.  **Automated Test Selection**: Based on the number of groups, the experimental design (paired/unpaired), and the results of the assumption checks, the app automatically selects the most appropriate statistical test.

| # of Groups | Design   | Assumptions Met (Normal & Homoscedastic)          | Assumptions Not Met                                 |
| :---------- | :------- | :------------------------------------------------ | :-------------------------------------------------- |
| 2           | Unpaired | Student's t-test                                  | Mann-Whitney U test                                 |
| >2          | Unpaired | One-way ANOVA + Tukey HSD post-hoc                | Kruskal-Wallis + Mann-Whitney U post-hoc (Bonferroni) |
| 2           | Paired   | Paired t-test                                     | Wilcoxon signed-rank test                           |
| >2          | Paired   | Repeated Measures ANOVA + Paired t-test (Bonferroni) | Friedman test + Wilcoxon post-hoc (Bonferroni)      |

## Built With

* [**Dash**](https://dash.plotly.com/) 
* [**Plotly**](https://plotly.com/python/) 
* [**Pandas**](https://pandas.pydata.org/) 
* [**SciPy**](https://scipy.org/) & [**Statsmodels**](https://www.statsmodels.org/) 
* [**PyInstaller**](https://pyinstaller.org/) - For packaging the standalone `.exe` application.

## References

1. Lord, S. J., Velle, K. B., Mullins, R. D., & Fritz-Laylin, L. K. (2020). SuperPlots: Communicating reproducibility and variability in cell biology. Journal of Cell Biology, 219(6). https://doi.org/10.1083/jcb.202001064

2. Pollard, D. A., Pollard, T. D., & Pollard, K. S. (2019). Empowering statistical methods for cellular and molecular biologists. Molecular Biology of the Cell, 30(12), 1359–1368. https://doi.org/10.1091/mbc.e15-02-0076


## **Cite our work** 
Bruni-Cardoso, A. CellBioStats, an application for robust visualization and statistical analysis of data from cell and molecular biology experiments
[![DOI](https://zenodo.org/badge/doi.org/10.5281/zenodo.16394653.svg)](https://doi.org/10.5281/zenodo.16394653)
 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
