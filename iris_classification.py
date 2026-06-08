# ============================================================
# TASK 1: Iris Flower Classification
# Intern: Akshaykumar B | Company: CodeAlpha
# Description: Train a machine learning model to classify
#              Iris flower species (setosa, versicolor, virginica)
#              based on sepal and petal measurements.
# ============================================================

# ----- STEP 0: Import all required libraries -----

import pandas as pd                          # For data manipulation and analysis
import numpy as np                           # For numerical operations
import matplotlib.pyplot as plt              # For creating static plots
import seaborn as sns                        # For beautiful statistical visualizations
import os                                    # For file and directory operations

from sklearn.datasets import load_iris       # To load the built-in Iris dataset
from sklearn.model_selection import train_test_split  # To split data into train/test sets
from sklearn.preprocessing import StandardScaler      # To standardize (scale) features
from sklearn.ensemble import RandomForestClassifier   # Our classification algorithm
from sklearn.metrics import (                # Metrics to evaluate the model
    accuracy_score,                          # Overall accuracy percentage
    classification_report,                   # Precision, recall, F1-score per class
    confusion_matrix,                        # Table of correct vs incorrect predictions
    ConfusionMatrixDisplay                   # Helper to plot the confusion matrix
)

# ----- STEP 1: Create a folder to save all plot images -----

PLOT_DIR = "plots"                           # Name of the folder to store plots
os.makedirs(PLOT_DIR, exist_ok=True)         # Create the folder if it doesn't exist

# ----- STEP 2: Load the Iris dataset from scikit-learn -----

iris = load_iris()                           # Load the dataset object from sklearn

# Convert the dataset into a pandas DataFrame for easier manipulation
df = pd.DataFrame(
    data=iris.data,                          # The feature values (measurements)
    columns=iris.feature_names               # Column names: sepal/petal length/width
)

df["species"] = iris.target                  # Add the target column (0, 1, or 2)

# Map numeric labels to actual species names for readability
species_map = {0: "setosa", 1: "versicolor", 2: "virginica"}  # Label mapping
df["species_name"] = df["species"].map(species_map)            # Create readable column

# ----- STEP 3: Explore the dataset (EDA) -----

print("=" * 60)                              # Print a separator line
print("IRIS FLOWER CLASSIFICATION - CodeAlpha Task 1")  # Title
print("=" * 60)                              # Print a separator line

print("\n--- First 5 rows of the dataset ---")  # Section header
print(df.head())                             # Display the first 5 rows

print("\n--- Dataset shape ---")             # Section header
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")  # Print (rows, cols)

print("\n--- Dataset info ---")              # Section header
print(df.info())                             # Print column types and non-null counts

print("\n--- Statistical summary ---")       # Section header
print(df.describe())                         # Print mean, std, min, max, etc.

print("\n--- Species distribution ---")      # Section header
print(df["species_name"].value_counts())     # Count how many samples per species

print("\n--- Missing values ---")            # Section header
print(df.isnull().sum())                     # Check for any missing values (should be 0)

# ----- STEP 4: Data Visualization -----

# Set a clean visual style for all plots
sns.set_style("whitegrid")                   # Use a white grid background
plt.rcParams["figure.dpi"] = 150            # Set high resolution for saved plots

# --- PLOT 1: Species Count Bar Chart ---
plt.figure(figsize=(8, 5))                   # Create a new figure (8 inches wide, 5 tall)
ax = sns.countplot(                          # Create a bar chart
    x="species_name",                        # X-axis: species names
    hue="species_name",                      # Color by species (required in seaborn 0.14+)
    data=df,                                 # Data source: our DataFrame
    palette="viridis",                       # Color scheme: viridis
    edgecolor="black",                       # Black edges on bars
    legend=False                             # Hide redundant legend
)
plt.title("Distribution of Iris Species",    # Chart title
          fontsize=16, fontweight="bold")     # Title formatting
plt.xlabel("Species", fontsize=13)           # X-axis label
plt.ylabel("Count", fontsize=13)             # Y-axis label

for p in ax.patches:                         # Loop through each bar in the chart
    ax.annotate(                             # Add text annotation on each bar
        f"{int(p.get_height())}",            # The count value as text
        (p.get_x() + p.get_width() / 2., p.get_height()),  # Position: center-top of bar
        ha="center", va="bottom",            # Alignment: centered horizontally
        fontsize=12, fontweight="bold"       # Font styling
    )

plt.tight_layout()                           # Adjust spacing to prevent clipping
plt.savefig(f"{PLOT_DIR}/01_species_distribution.png")  # Save the plot as PNG
plt.close()                                  # Close the figure to free memory
print("\n[OK] Saved: 01_species_distribution.png")  # Confirmation message

# --- PLOT 2: Pairplot (all feature combinations colored by species) ---
pairplot = sns.pairplot(                     # Create a grid of scatter plots
    df,                                      # Data source
    hue="species_name",                      # Color by species
    palette="viridis",                       # Color scheme
    diag_kind="kde",                         # Diagonal: kernel density estimate
    plot_kws={"alpha": 0.7, "s": 40},       # Scatter plot: semi-transparent, size 40
    height=2.5                               # Height of each subplot
)
pairplot.figure.suptitle(                    # Add a super-title above the grid
    "Pairwise Feature Relationships",        # Title text
    y=1.02, fontsize=16, fontweight="bold"   # Position and formatting
)
plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/02_pairplot.png",   # Save the plot
            bbox_inches="tight")             # Tight bounding box to include title
plt.close()                                  # Close figure
print("[OK] Saved: 02_pairplot.png")          # Confirmation message

# --- PLOT 3: Correlation Heatmap ---
plt.figure(figsize=(8, 6))                   # Create a new figure
numeric_df = df.drop(                        # Remove non-numeric columns
    columns=["species", "species_name"]      # Drop species columns
)
correlation = numeric_df.corr()              # Calculate the correlation matrix
sns.heatmap(                                 # Create a heatmap
    correlation,                             # Data: the correlation matrix
    annot=True,                              # Show numbers in each cell
    fmt=".2f",                               # Format numbers to 2 decimal places
    cmap="coolwarm",                         # Color scheme: blue-to-red
    linewidths=0.5,                          # Width of cell borders
    square=True,                             # Make cells square-shaped
    cbar_kws={"shrink": 0.8}                # Shrink the color bar slightly
)
plt.title("Feature Correlation Heatmap",     # Chart title
          fontsize=16, fontweight="bold")     # Title formatting
plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/03_correlation_heatmap.png")  # Save the plot
plt.close()                                  # Close figure
print("[OK] Saved: 03_correlation_heatmap.png")  # Confirmation message

# --- PLOT 4: Box Plots for each feature ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # Create a 2x2 grid of subplots
fig.suptitle("Feature Distributions by Species",  # Super-title
             fontsize=16, fontweight="bold")        # Title formatting

features = iris.feature_names                # List of all 4 feature names
for i, (feature, ax) in enumerate(           # Loop through features and subplots
    zip(features, axes.flatten())            # Pair each feature with its subplot
):
    sns.boxplot(                             # Create a box plot
        x="species_name",                   # X-axis: species
        y=feature,                           # Y-axis: measurement values
        hue="species_name",                 # Color by species (seaborn 0.14+)
        data=df,                             # Data source
        palette="viridis",                   # Color scheme
        legend=False,                        # Hide redundant legend
        ax=ax                                # Which subplot to draw on
    )
    ax.set_title(feature.replace(" (cm)", "").title(),  # Clean subplot title
                 fontsize=13, fontweight="bold")          # Title formatting
    ax.set_xlabel("")                        # Remove x-axis label (shown on bottom row)
    ax.set_ylabel("cm", fontsize=11)         # Y-axis label

plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/04_boxplots.png")   # Save the plot
plt.close()                                  # Close figure
print("[OK] Saved: 04_boxplots.png")          # Confirmation message

# --- PLOT 5: Violin Plots ---
fig, axes = plt.subplots(2, 2, figsize=(12, 9))  # Create another 2x2 grid
fig.suptitle("Violin Plots - Feature Density by Species",  # Super-title
             fontsize=16, fontweight="bold")                 # Title formatting

for i, (feature, ax) in enumerate(           # Loop through features and subplots
    zip(features, axes.flatten())            # Pair them up
):
    sns.violinplot(                          # Create a violin plot
        x="species_name",                   # X-axis: species
        y=feature,                           # Y-axis: measurement values
        hue="species_name",                 # Color by species (seaborn 0.14+)
        data=df,                             # Data source
        palette="muted",                     # Soft color scheme
        inner="quartile",                    # Show quartile lines inside violin
        legend=False,                        # Hide redundant legend
        ax=ax                                # Which subplot to draw on
    )
    ax.set_title(feature.replace(" (cm)", "").title(),  # Clean subplot title
                 fontsize=13, fontweight="bold")          # Title formatting
    ax.set_xlabel("")                        # Remove x-axis label
    ax.set_ylabel("cm", fontsize=11)         # Y-axis label

plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/05_violinplots.png")  # Save the plot
plt.close()                                  # Close figure
print("[OK] Saved: 05_violinplots.png")       # Confirmation message

# ----- STEP 5: Prepare data for machine learning -----

X = df[iris.feature_names]                   # Features: the 4 measurement columns
y = df["species"]                            # Target: the species label (0, 1, or 2)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X,                                       # Feature data
    y,                                       # Target labels
    test_size=0.2,                           # 20% of data goes to test set
    random_state=42,                         # Fixed seed for reproducibility
    stratify=y                               # Ensure equal class proportions in both sets
)

print(f"\n--- Data Split ---")               # Section header
print(f"Training samples: {X_train.shape[0]}")  # Number of training samples
print(f"Testing samples:  {X_test.shape[0]}")   # Number of testing samples

# Scale/standardize the features (mean=0, std=1) for better model performance
scaler = StandardScaler()                    # Create a scaler object
X_train_scaled = scaler.fit_transform(X_train)  # Fit on training data and transform it
X_test_scaled = scaler.transform(X_test)     # Transform test data using same parameters

# ----- STEP 6: Train the Random Forest Classifier -----

print("\n--- Training Random Forest Classifier ---")  # Section header

model = RandomForestClassifier(              # Create the model
    n_estimators=100,                        # Use 100 decision trees in the forest
    random_state=42,                         # Fixed seed for reproducibility
    max_depth=5,                             # Limit tree depth to prevent overfitting
    n_jobs=-1                                # Use all CPU cores for faster training
)

model.fit(X_train_scaled, y_train)           # Train the model on scaled training data
print("[OK] Model trained successfully!")     # Confirmation message

# ----- STEP 7: Make predictions on the test set -----

y_pred = model.predict(X_test_scaled)        # Predict species for the test data

# ----- STEP 8: Evaluate the model -----

accuracy = accuracy_score(y_test, y_pred)    # Calculate overall accuracy

print(f"\n--- Model Evaluation ---")         # Section header
print(f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")  # Print accuracy

print("\n--- Classification Report ---")     # Section header
print(classification_report(                 # Print detailed metrics per class
    y_test,                                  # Actual labels
    y_pred,                                  # Predicted labels
    target_names=iris.target_names           # Use species names instead of numbers
))

# ----- STEP 9: Confusion Matrix Visualization -----

# --- PLOT 6: Confusion Matrix Heatmap ---
cm = confusion_matrix(y_test, y_pred)        # Calculate the confusion matrix

plt.figure(figsize=(8, 6))                   # Create a new figure
sns.heatmap(                                 # Create a heatmap of the confusion matrix
    cm,                                      # Data: the confusion matrix
    annot=True,                              # Show numbers in each cell
    fmt="d",                                 # Format as integers
    cmap="Blues",                             # Blue color scheme
    xticklabels=iris.target_names,           # X-axis labels: species names
    yticklabels=iris.target_names,           # Y-axis labels: species names
    linewidths=1,                            # Cell border width
    linecolor="black",                       # Cell border color
    square=True                              # Square-shaped cells
)
plt.title("Confusion Matrix",               # Chart title
          fontsize=16, fontweight="bold")     # Title formatting
plt.xlabel("Predicted Species", fontsize=13) # X-axis label
plt.ylabel("Actual Species", fontsize=13)    # Y-axis label
plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/06_confusion_matrix.png")  # Save the plot
plt.close()                                  # Close figure
print("\n[OK] Saved: 06_confusion_matrix.png")  # Confirmation message

# ----- STEP 10: Feature Importance -----

importances = model.feature_importances_     # Get importance score for each feature
feature_importance_df = pd.DataFrame({       # Create a DataFrame for visualization
    "Feature": iris.feature_names,           # Feature names
    "Importance": importances                # Importance scores
}).sort_values("Importance", ascending=True)  # Sort ascending for horizontal bar chart

# --- PLOT 7: Feature Importance Bar Chart ---
plt.figure(figsize=(8, 5))                   # Create a new figure
bars = plt.barh(                             # Horizontal bar chart
    feature_importance_df["Feature"],        # Y-axis: feature names
    feature_importance_df["Importance"],     # X-axis: importance scores
    color=sns.color_palette("viridis", 4),   # Use 4 viridis colors
    edgecolor="black"                        # Black edges on bars
)
plt.title("Feature Importance (Random Forest)",  # Chart title
          fontsize=16, fontweight="bold")          # Title formatting
plt.xlabel("Importance Score", fontsize=13)  # X-axis label

for bar in bars:                             # Loop through each bar
    width = bar.get_width()                  # Get the bar width (importance value)
    plt.text(                                # Add text label at end of bar
        width + 0.005,                       # X position: slightly right of bar end
        bar.get_y() + bar.get_height() / 2,  # Y position: center of bar
        f"{width:.3f}",                      # Text: importance value to 3 decimals
        va="center", fontsize=11             # Vertical alignment and font size
    )

plt.tight_layout()                           # Adjust spacing
plt.savefig(f"{PLOT_DIR}/07_feature_importance.png")  # Save the plot
plt.close()                                  # Close figure
print("[OK] Saved: 07_feature_importance.png")  # Confirmation message

# ----- STEP 11: Print the Feature Importance Table -----

print("\n--- Feature Importance ---")        # Section header
for _, row in feature_importance_df.iloc[::-1].iterrows():  # Loop in descending order
    bar = "#" * int(row["Importance"] * 50)  # Create a text-based bar
    print(f"  {row['Feature']:30s} {row['Importance']:.4f}  {bar}")  # Print each row

# ----- STEP 12: Final Summary -----

print("\n" + "=" * 60)                       # Separator line
print("SUMMARY")                             # Section header
print("=" * 60)                              # Separator line
print(f"  Model Used       : Random Forest Classifier")  # Model name
print(f"  Number of Trees  : 100")           # Number of estimators
print(f"  Training Samples : {X_train.shape[0]}")  # Training set size
print(f"  Testing Samples  : {X_test.shape[0]}")   # Test set size
print(f"  Accuracy         : {accuracy * 100:.2f}%")  # Accuracy percentage
print(f"  Plots Saved In   : ./{PLOT_DIR}/")  # Plot directory
print("=" * 60)                              # Separator line
print("\n[OK] All tasks completed successfully!")  # Final confirmation
print(f"[OK] {len(os.listdir(PLOT_DIR))} plots saved to './{PLOT_DIR}/' folder")  # Plot count
