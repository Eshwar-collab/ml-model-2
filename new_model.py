# Almatti Dam Seepage Prediction using Deep Neural Network
# Monthly Data Analysis and Prediction (No TensorFlow Required)

# ===========================
# 1. LIBRARY IMPORTS
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

print("="*60)
print("ALMATTI DAM SEEPAGE PREDICTION - DNN MODEL")
print("="*60)

# ===========================
# 2. FILE READ
# ===========================
file_path = r"D:\Almatti dam\2020 to 2025.xlsx"
print(f"\nReading data from: {file_path}")

try:
    df = pd.read_excel(file_path)
    print(f"✓ Data loaded successfully!")
    print(f"Shape: {df.shape}")
except FileNotFoundError:
    print("⚠ File not found. Trying alternative path...")
    try:
        file_path = r"D:\Almatti dam file name: 2020 to 2025.xlsx"
        df = pd.read_excel(file_path)
        print(f"✓ Data loaded successfully from alternative path!")
        print(f"Shape: {df.shape}")
    except:
        print("ERROR: Could not find the file. Please check these paths:")
        print("  - D:\\Almatti dam\\2020 to 2025.xlsx")
        print("  - D:\\Almatti dam file name: 2020 to 2025.xlsx")
        exit()

# Display column names
print(f"\nColumns in dataset: {list(df.columns)}")

# Standardize column names (remove extra spaces)
df.columns = df.columns.str.strip()

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

print(f"\nDate Range: {df['DATE'].min()} to {df['DATE'].max()}")
print(f"Total Records: {len(df)}")

# ===========================
# 3. DATA MINING / BASIC STATISTICS
# ===========================
print("\n" + "="*60)
print("DATA MINING & BASIC STATISTICS")
print("="*60)

# Display first few rows
print("\nFirst 5 rows:")
print(df.head())

# Basic statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("No missing values")

# Handle missing values if any
df = df.fillna(df.median(numeric_only=True))

# Data types
print("\nData Types:")
print(df.dtypes)

# ===========================
# 4. CORRELATION MATRIX
# ===========================
print("\n" + "="*60)
print("CORRELATION ANALYSIS")
print("="*60)

# Select numerical columns (exclude DATE)
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"\nNumerical columns: {numerical_cols}")

# Calculate correlation matrix
corr_matrix = df[numerical_cols].corr()

# Display correlation with target (Seepage)
seepage_col = 'Seepage (l/day)'
if seepage_col in df.columns:
    print(f"\nCorrelation with {seepage_col}:")
    print(corr_matrix[seepage_col].sort_values(ascending=False))
else:
    # Try to find seepage column with different name
    possible_names = [col for col in df.columns if 'seepage' in col.lower()]
    if possible_names:
        seepage_col = possible_names[0]
        print(f"\nUsing column: {seepage_col}")
        print(f"\nCorrelation with {seepage_col}:")
        print(corr_matrix[seepage_col].sort_values(ascending=False))
    else:
        print("ERROR: Could not find seepage column!")
        exit()

# Plot correlation heatmap
plt.figure(figsize=(14, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix - Almatti Dam Parameters', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Correlation matrix saved")

# ===========================
# 5. FEATURE RANKING
# ===========================
print("\n" + "="*60)
print("FEATURE RANKING")
print("="*60)

# Prepare features and target
X = df[numerical_cols].drop(columns=[seepage_col])
y = df[seepage_col]

print(f"\nFeatures: {list(X.columns)}")
print(f"Target: {seepage_col}")

# Random Forest for feature importance
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance Ranking:")
print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color='steelblue')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Feature Importance Ranking', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Feature importance plot saved")

# ===========================
# 6. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ===========================
print("\n" + "="*60)
print("PRINCIPAL COMPONENT ANALYSIS")
print("="*60)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA()
pca_components = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\nExplained Variance by each PC:")
for i, var in enumerate(explained_variance, 1):
    print(f"PC{i}: {var:.4f} ({var*100:.2f}%)")

print(f"\nCumulative Variance:")
for i, var in enumerate(cumulative_variance, 1):
    print(f"PC1-PC{i}: {var:.4f} ({var*100:.2f}%)")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].bar(range(1, len(explained_variance)+1), explained_variance, color='steelblue', alpha=0.8)
axes[0].plot(range(1, len(explained_variance)+1), explained_variance, marker='o', color='red')
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('Scree Plot', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Cumulative variance
axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', color='green')
axes[1].axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Explained Variance', fontsize=12)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ PCA analysis saved")

# ===========================
# 7. DATA SPLITTING & MODEL TRAINING
# ===========================
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

# Split data: 70% train, 20% test, 10% predict
n = len(df)
train_size = int(0.7 * n)
test_size = int(0.2 * n)

train_df = df[:train_size].copy()
test_df = df[train_size:train_size+test_size].copy()
predict_df = df[train_size+test_size:].copy()

print(f"\nData Split:")
print(f"Training Set: {len(train_df)} samples ({len(train_df)/n*100:.1f}%)")
print(f"Testing Set: {len(test_df)} samples ({len(test_df)/n*100:.1f}%)")
print(f"Prediction Set: {len(predict_df)} samples ({len(predict_df)/n*100:.1f}%)")

# Prepare datasets
X_train = train_df[X.columns]
y_train = train_df[seepage_col]
X_test = test_df[X.columns]
y_test = test_df[seepage_col]
X_predict = predict_df[X.columns]
y_predict = predict_df[seepage_col]

# Scale features
scaler_model = StandardScaler()
X_train_scaled = scaler_model.fit_transform(X_train)
X_test_scaled = scaler_model.transform(X_test)
X_predict_scaled = scaler_model.transform(X_predict)

# Build Deep Neural Network Model using MLPRegressor
print("\nBuilding Multi-Layer Perceptron (Deep Neural Network)...")
print("Architecture: Input → 128 → 64 → 32 → 16 → Output")

model = MLPRegressor(
    hidden_layer_sizes=(128, 64, 32, 16),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=20,
    verbose=True
)

# Train model
print("\nTraining model...")
model.fit(X_train_scaled, y_train)

print("✓ Model training completed!")
print(f"Number of iterations: {model.n_iter_}")
print(f"Training loss: {model.loss_:.4f}")

# ===========================
# 8. MODEL EVALUATION
# ===========================
print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_predict_pred = model.predict(X_predict_scaled)

# Calculate metrics
def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    print(f"\n{dataset_name} Set Metrics:")
    print(f"  MSE:  {mse:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE:  {mae:.2f}")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}

train_metrics = evaluate_model(y_train, y_train_pred, "Training")
test_metrics = evaluate_model(y_test, y_test_pred, "Testing")
predict_metrics = evaluate_model(y_predict, y_predict_pred, "Prediction")

# ===========================
# 9. PLOTS
# ===========================
print("\n" + "="*60)
print("GENERATING PLOTS")
print("="*60)

# Plot 1: Training Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(model.loss_curve_, linewidth=2, color='blue')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Model Training Loss Curve', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_loss.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Training loss plot saved")

# Plot 2: Time Series - Actual vs Predicted (All Data)
plt.figure(figsize=(16, 6))

plt.plot(train_df['DATE'], y_train, 'o-', label='Train Actual', color='blue', alpha=0.6, markersize=4)
plt.plot(train_df['DATE'], y_train_pred, 's-', label='Train Predicted', color='lightblue', alpha=0.8, markersize=3)

plt.plot(test_df['DATE'], y_test, 'o-', label='Test Actual', color='green', alpha=0.6, markersize=4)
plt.plot(test_df['DATE'], y_test_pred, 's-', label='Test Predicted', color='lightgreen', alpha=0.8, markersize=3)

plt.plot(predict_df['DATE'], y_predict, 'o-', label='Predict Actual', color='red', alpha=0.6, markersize=4)
plt.plot(predict_df['DATE'], y_predict_pred, 's-', label='Predict Predicted', color='salmon', alpha=0.8, markersize=3)

plt.xlabel('Date', fontsize=12)
plt.ylabel('Seepage (l/day)', fontsize=12)
plt.title('Time Series: Actual vs Predicted Seepage', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('timeseries_all.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Time series plot saved")

# Plot 3: Scatter Plots - Actual vs Predicted
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

datasets = [
    (y_train, y_train_pred, 'Training', 'blue', train_metrics),
    (y_test, y_test_pred, 'Testing', 'green', test_metrics),
    (y_predict, y_predict_pred, 'Prediction', 'red', predict_metrics)
]

for idx, (y_true, y_pred, name, color, metrics) in enumerate(datasets):
    axes[idx].scatter(y_true, y_pred, alpha=0.6, color=color, edgecolors='k', s=50)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[idx].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    axes[idx].set_xlabel('Actual Seepage (l/day)', fontsize=12)
    axes[idx].set_ylabel('Predicted Seepage (l/day)', fontsize=12)
    axes[idx].set_title(f'{name} Set (R²={metrics["R2"]:.3f})', fontsize=14, fontweight='bold')
    axes[idx].legend()
    axes[idx].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('scatter_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Scatter plots saved")

# Plot 4: Residual Plots
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (dates, y_true, y_pred, name, color) in enumerate([
    (train_df['DATE'], y_train, y_train_pred, 'Training', 'blue'),
    (test_df['DATE'], y_test, y_test_pred, 'Testing', 'green'),
    (predict_df['DATE'], y_predict, y_predict_pred, 'Prediction', 'red')
]):
    residuals = y_true.values - y_pred
    axes[idx].plot(dates, residuals, 'o-', color=color, alpha=0.6, markersize=4)
    axes[idx].axhline(y=0, color='black', linestyle='--', linewidth=2)
    axes[idx].fill_between(dates, 0, residuals, alpha=0.2, color=color)
    axes[idx].set_xlabel('Date', fontsize=12)
    axes[idx].set_ylabel('Residuals', fontsize=12)
    axes[idx].set_title(f'{name} Residuals', fontsize=14, fontweight='bold')
    axes[idx].grid(alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Residual plots saved")

# Plot 5: Model Performance Comparison
metrics_df = pd.DataFrame({
    'Train': [train_metrics['R2'], train_metrics['RMSE'], train_metrics['MAE'], train_metrics['MAPE']],
    'Test': [test_metrics['R2'], test_metrics['RMSE'], test_metrics['MAE'], test_metrics['MAPE']],
    'Predict': [predict_metrics['R2'], predict_metrics['RMSE'], predict_metrics['MAE'], predict_metrics['MAPE']]
}, index=['R²', 'RMSE', 'MAE', 'MAPE'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, metric in enumerate(['R²', 'RMSE', 'MAE', 'MAPE']):
    ax = axes[idx]
    metrics_df.loc[metric].plot(kind='bar', ax=ax, color=['blue', 'green', 'red'], alpha=0.7)
    ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel('Dataset', fontsize=11)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=0)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Metrics comparison plot saved")

# Plot 6: Detailed Time Series by Dataset
fig, axes = plt.subplots(3, 1, figsize=(16, 12))

plot_data = [
    (train_df['DATE'], y_train, y_train_pred, 'Training', 'blue'),
    (test_df['DATE'], y_test, y_test_pred, 'Testing', 'green'),
    (predict_df['DATE'], y_predict, y_predict_pred, 'Prediction', 'red')
]

for idx, (dates, actual, predicted, title, color) in enumerate(plot_data):
    axes[idx].plot(dates, actual, 'o-', label='Actual', color=color, alpha=0.7, linewidth=2, markersize=5)
    axes[idx].plot(dates, predicted, 's--', label='Predicted', color='orange', alpha=0.7, linewidth=2, markersize=4)
    axes[idx].fill_between(dates, actual, predicted, alpha=0.2, color='gray')
    axes[idx].set_xlabel('Date', fontsize=11)
    axes[idx].set_ylabel('Seepage (l/day)', fontsize=11)
    axes[idx].set_title(f'{title} Set - Actual vs Predicted', fontsize=13, fontweight='bold')
    axes[idx].legend(loc='best')
    axes[idx].grid(alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('detailed_timeseries.png', dpi=300, bbox_inches='tight')
plt.show()
print("✓ Detailed time series plot saved")

print("\n" + "="*60)
print("ANALYSIS COMPLETED SUCCESSFULLY")
print("="*60)
print("\nOutput Files Generated:")
print("  1. correlation_matrix.png")
print("  2. feature_importance.png")
print("  3. pca_analysis.png")
print("  4. training_loss.png")
print("  5. timeseries_all.png")
print("  6. scatter_plots.png")
print("  7. residual_plots.png")
print("  8. metrics_comparison.png")
print("  9. detailed_timeseries.png")

# Save predictions to CSV
results_df = pd.DataFrame({
    'Date': predict_df['DATE'],
    'Actual_Seepage': y_predict.values,
    'Predicted_Seepage': y_predict_pred,
    'Absolute_Error': np.abs(y_predict.values - y_predict_pred),
    'Percentage_Error': np.abs((y_predict.values - y_predict_pred) / y_predict.values) * 100
})
results_df.to_csv('seepage_predictions.csv', index=False)
print(" 10. seepage_predictions.csv")

# Save all metrics to CSV
all_metrics = pd.DataFrame({
    'Dataset': ['Training', 'Testing', 'Prediction'],
    'MSE': [train_metrics['MSE'], test_metrics['MSE'], predict_metrics['MSE']],
    'RMSE': [train_metrics['RMSE'], test_metrics['RMSE'], predict_metrics['RMSE']],
    'MAE': [train_metrics['MAE'], test_metrics['MAE'], predict_metrics['MAE']],
    'R2': [train_metrics['R2'], test_metrics['R2'], predict_metrics['R2']],
    'MAPE': [train_metrics['MAPE'], test_metrics['MAPE'], predict_metrics['MAPE']]
})
all_metrics.to_csv('model_metrics.csv', index=False)
print(" 11. model_metrics.csv")

print("\n✓ All analysis complete! Check the generated files in your directory.")
print("\nModel Summary:")
print(f"  Best R² Score: {max(train_metrics['R2'], test_metrics['R2'], predict_metrics['R2']):.4f}")
print(f"  Best RMSE: {min(train_metrics['RMSE'], test_metrics['RMSE'], predict_metrics['RMSE']):.2f}")
print(f"  Architecture: 128 → 64 → 32 → 16 neurons")
print(f"  Training iterations: {model.n_iter_}")