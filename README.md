<p align="center">
  <img src="https://github.com/brunicardoso/CellBioStats/raw/main/assets/logo.png" alt="CellBioStats Logo" width="300"/>
</p>

An interactive web application specifically designed for scientists in **cell biology, molecular biology, and biochemistry**. CellBioStats simplifies the creation of publication-quality **SuperPlots** and performs robust statistical analysis on hierarchical data, helping you visualize complex datasets and avoid common pitfalls like pseudoreplication.

## ✨ Features

* **Interactive SuperPlots**: Generates publication-quality SuperPlots using Plotly, clearly showing individual data points, replicate means, and overall treatment means with standard error.
* **Automated Statistical Analysis**: Automatically performs normality checks, variance homogeneity tests, and selects the appropriate statistical test for your data.
* **Hierarchical Statistics**: Avoids pseudoreplication by correctly performing statistical tests on **replicate means**, not on raw technical measurements.
* **Paired & Unpaired Data**: Handles both independent (unpaired) and repeated measures (paired) experimental designs common in cell biology.
* **Easy Data Upload**: Supports both `.csv` and `.xlsx` file formats.
* **Full Customization**: Easily customize plot aesthetics like font size, color schemes, marker size, and plot templates.
* **Data Downsampling**: Option to display a random subset of your data points to keep plots clean and responsive with very large datasets.
* **Export Results**: Download the detailed statistical summary as a `.txt` file and the plot as a `.png, .jpeg, .svg or .pdf` file directly from the app.

## 🚀 Getting Started

You can run CellBioStats in three different ways.

### Option 1: Standalone Application (`.exe` for Windows)

This is the easiest method for most users. No installation is required.

1.  Go to the [**Releases**](https://github.com/brunicardoso/CellBioStats/dist) page of this repository.
2.  Download the latest `.exe` file.
3.  Double-click the file to run the application.

### Option 2: Google Colab (Cloud-Based)

Run the app in your browser without any local installation.

1.  Click the "Open in Colab" badge at the top of this README.
2.  Run the cells in the notebook to start the application.

### Option 3: Run Locally (For Developers)

If you have Python installed, you can run the app from the source code.

1.  **Clone the repository:**
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

## 📊 How to Use the App

1.  **Upload Data**: Click the "Upload Data" button and select your `.csv` or `.xlsx` file.
2.  **Map Columns**: Select the appropriate columns from your file for **Treatment**, **Value**, and **Replicate**.
3.  **Select Test Type**: (Optional) If your experiment uses a repeated measures or paired design, check the "Use paired/repeated statistical tests" box.
4.  **Generate**: Click the "Generate Analysis" button.
5.  **Customize**: Use the sliders and dropdowns in the "Plot Customization" panel to adjust the appearance of your plot.
6.  **Download**: Click the "Download Summary" button to save the statistical report or use the camera icon on the plot to save it as a PNG image.

## 📝 Input Data Format

The application expects your data to be in a **long format**. You must have at least three columns:

* **Treatment Column**: Identifies the different experimental groups (e.g., 'Control', 'Drug A').
* **Value Column**: Contains the numeric measurement data (the dependent variable).
* **Replicate Column**: Identifies the independent experimental replicates (e.g., experiment number, animal ID, cell line batch).

Here is an example of a valid data structure:

| Treatment | Value | Replicate |
| :-------- | :---- | :-------- |
| Control   | 10.5  | 1         |
| Control   | 11.2  | 1         |
| Control   | 12.1  | 2         |
| Control   | 11.8  | 2         |
| Drug A    | 15.3  | 1         |
| Drug A    | 14.9  | 1         |
| Drug A    | 16.5  | 2         |
| Drug A    | 17.1  | 2         |

## 🔬 Statistical Methodology

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

## 🛠️ Built With

* [**Dash**](https://dash.plotly.com/) - The web framework for building the application.
* [**Plotly**](https://plotly.com/python/) - For creating interactive data visualizations.
* [**Pandas**](https://pandas.pydata.org/) - For data manipulation and analysis.
* [**SciPy**](https://scipy.org/) & [**Statsmodels**](https://www.statsmodels.org/) - For performing statistical tests.
* [**PyInstaller**](https://pyinstaller.org/) - For packaging the standalone `.exe` application.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
