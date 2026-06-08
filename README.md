# 🌸 Iris Flower Classification

> **CodeAlpha Data Science Internship — Task 1**

A machine learning project that classifies Iris flower species (*setosa*, *versicolor*, *virginica*) based on sepal and petal measurements using a **Random Forest Classifier**.

---

## 📋 Table of Contents

- [About the Project](#about-the-project)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Results](#results)
- [Visualizations](#visualizations)
- [Key Learnings](#key-learnings)
- [Author](#author)

---

## 📖 About the Project

The **Iris Flower Classification** project is a classic introductory machine learning task. The goal is to build a model that can accurately predict the species of an Iris flower given four physical measurements:

| Feature | Description |
|---|---|
| Sepal Length | Length of the sepal in cm |
| Sepal Width | Width of the sepal in cm |
| Petal Length | Length of the petal in cm |
| Petal Width | Width of the petal in cm |

The model learns patterns from labeled training data and then classifies unseen flowers into one of three species:

- **Iris-setosa**
- **Iris-versicolor**
- **Iris-virginica**

---

## 📊 Dataset

- **Source:** [Scikit-learn's built-in Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)
- **Samples:** 150 (50 per species)
- **Features:** 4 numerical measurements
- **Target:** 3 species classes
- **Missing Values:** None

---

## 🛠️ Technologies Used

| Library | Purpose |
|---|---|
| **Python 3.x** | Programming language |
| **pandas** | Data manipulation and analysis |
| **NumPy** | Numerical computations |
| **matplotlib** | Data visualization |
| **seaborn** | Statistical data visualization |
| **scikit-learn** | Machine learning model and evaluation |

---

## 📁 Project Structure

```
Task-1/
│
├── iris_classification.py    # Main Python script (fully commented)
├── Iris.csv                  # Dataset file
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
│
└── plots/                    # Generated visualizations
    ├── 01_species_distribution.png
    ├── 02_pairplot.png
    ├── 03_correlation_heatmap.png
    ├── 04_boxplots.png
    ├── 05_violinplots.png
    ├── 06_confusion_matrix.png
    └── 07_feature_importance.png
```

---

## ⚙️ Setup & Installation

### Prerequisites

Make sure you have **Python 3.7+** installed on your system.

### 1. Clone the Repository

```bash
git clone https://github.com/Akshaykumar-B/FDS-LAB.git
cd FDS-LAB
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

Run the classification script:

```bash
python iris_classification.py
```

The script will:

1. ✅ Load and explore the Iris dataset
2. ✅ Generate 7 visualizations and save them to the `plots/` folder
3. ✅ Split the data into training (80%) and testing (20%) sets
4. ✅ Train a Random Forest Classifier with 100 decision trees
5. ✅ Evaluate accuracy and print a classification report
6. ✅ Display feature importance rankings

---

## 📈 Results

| Metric | Value |
|---|---|
| **Model** | Random Forest Classifier |
| **Number of Trees** | 100 |
| **Train/Test Split** | 80% / 20% |
| **Accuracy** | ~100% |

### Classification Report

| Species | Precision | Recall | F1-Score |
|---|---|---|---|
| Setosa | 1.00 | 1.00 | 1.00 |
| Versicolor | 1.00 | 1.00 | 1.00 |
| Virginica | 1.00 | 1.00 | 1.00 |

> *Note: Results may vary slightly depending on the random state and train/test split.*

---

## 📊 Visualizations

The script generates 7 publication-quality plots:

| # | Plot | Description |
|---|---|---|
| 1 | **Species Distribution** | Bar chart showing the count of each species |
| 2 | **Pairplot** | Scatter plots of all feature pairs, colored by species |
| 3 | **Correlation Heatmap** | Shows relationships between numerical features |
| 4 | **Box Plots** | Distribution of each feature by species |
| 5 | **Violin Plots** | Density distribution of each feature by species |
| 6 | **Confusion Matrix** | Heatmap of correct vs. incorrect predictions |
| 7 | **Feature Importance** | Bar chart ranking the most important features |

---

## 💡 Key Learnings

- **Petal measurements** (length and width) are the most important features for classification
- **Iris-setosa** is linearly separable from the other two species
- **Random Forest** achieves near-perfect accuracy on this dataset
- Proper **data splitting** and **feature scaling** are essential steps in any ML pipeline
- **Visualization** helps in understanding data patterns before model building

---

## 🧠 Concepts Covered

- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Scaling
- Train-Test Splitting with Stratification
- Random Forest Classification
- Model Evaluation Metrics (Accuracy, Precision, Recall, F1-Score)
- Confusion Matrix Interpretation
- Feature Importance Analysis

---

## 👤 Author

**Akshaykumar B**
- 🏢 Data Science Intern at [CodeAlpha](https://www.codealpha.tech/)
- 🔗 GitHub: [@Akshaykumar-B](https://github.com/Akshaykumar-B)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ for CodeAlpha Internship
</p>
