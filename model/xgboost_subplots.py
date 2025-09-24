import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import plotly.graph_objects as go
import plotly.subplots as sp
from scipy.stats import pearsonr
import joblib
import time
import xgboost as xgb 
from tensorflow.keras.models import load_model

# Load the CSV files using the same data as the neural network
combined_df = pd.read_csv("/Github/DeepLforConcreteReuse/Data/combined_csv_file.csv")

# Specify targets and features same as the neural network code
targets = ['out:Leftover inventory[m2]', 'out:Displacement Average[cm]', 'out:Uncovered Wall Area[m2]', 'out:Average Mapping Tolerance']
features = ['in:Inventory', 'in:Wall Length', 'in:Wall Height', 'in:Door', 'in:Curve Frequency', 'in:Curve Amplitude'] 

# Check for NaN values in the input data and handle them
print(f"NaN values in features: {combined_df[features].isna().sum().sum()}")
print(f"NaN values in targets: {combined_df[targets].isna().sum().sum()}")

# Splitting data into training and testing sets - same random_state for consistency
X_train, X_test, Y_train, Y_test = train_test_split(
    combined_df[features], combined_df[targets], test_size=0.2, random_state=42
)

# Using the same preprocessing pipeline as the neural network
# Initialize the PowerTransformer
scaler_yeo = PowerTransformer(method='yeo-johnson', standardize=False) 

# Fit and transform the target data
Y_train_trans = scaler_yeo.fit_transform(Y_train)

# Initialize StandardScaler
scaler_features = StandardScaler()
scaler_targets = StandardScaler()

# Fit and transform features and targets
X_train_scaled = scaler_features.fit_transform(X_train)
Y_train_scaled = scaler_targets.fit_transform(Y_train_trans)
X_test_scaled = scaler_features.transform(X_test)

# Create and train XGBoost model
print("Training XGBoost model...")
start_time = time.time()

# Create base XGBoost regressor with parameters to minimize overfitting
base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=1,
    gamma=0,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42
)

# Wrap with MultiOutputRegressor
xgb_multi = MultiOutputRegressor(base_model)

# Train the model
xgb_multi.fit(X_train_scaled, Y_train_scaled)

training_time = time.time() - start_time
print(f"XGBoost training completed in {training_time:.2f} seconds")

# Save the model
joblib.dump(xgb_multi, '/Github/DeepLforConcreteReuse/Model/xgb_model.pkl')
print("XGBoost model saved.")

# Make predictions and time the inference
start_time = time.time()
xgb_scaled_pred = xgb_multi.predict(X_test_scaled)
inference_time = time.time() - start_time
print(f"XGBoost inference completed in {inference_time:.2f} seconds for {len(X_test)} samples")

# Convert predictions to DataFrame
xgb_scaled_pred_df = pd.DataFrame(xgb_scaled_pred, columns=Y_train.columns)

# Inverse transform predictions (same process as for neural network)
# First reverse the standard scaling
xgb_pred = scaler_targets.inverse_transform(xgb_scaled_pred_df)
xgb_pred_df = pd.DataFrame(xgb_pred, columns=Y_train.columns)

# Then reverse the Yeo-Johnson transformation
xgb_predictions = scaler_yeo.inverse_transform(xgb_pred_df)

# Calculate MSE and R² Score for XGBoost
xgb_mse = mean_squared_error(Y_test, xgb_predictions)
xgb_r2 = r2_score(Y_test, xgb_predictions)
print(f"XGBoost - MSE: {xgb_mse:.4f}, R²: {xgb_r2:.4f}")

# Load the neural network model and generate predictions
print("Loading Neural Network model and generating predictions...")
model = load_model('/Github/DeepLforConcreteReuse//Model/model.keras')

# Generate neural network predictions using the same test data
nn_scaled_pred = model.predict(X_test_scaled)
nn_scaled_pred_df = pd.DataFrame(nn_scaled_pred, columns=Y_train.columns)

# Inverse transform the prediction (same process as for XGBoost)
# First reverse the standard scaling
nn_pred = scaler_targets.inverse_transform(nn_scaled_pred_df)
nn_pred_df = pd.DataFrame(nn_pred, columns=Y_train.columns)

# Then reverse the Yeo-Johnson transformation
nn_predictions = scaler_yeo.inverse_transform(nn_pred_df)

# Calculate MSE and R² Score for Neural Network
nn_mse = mean_squared_error(Y_test, nn_predictions)
nn_r2 = r2_score(Y_test, nn_predictions)
print(f"Neural Network - MSE: {nn_mse:.4f}, R²: {nn_r2:.4f}")

# Save the neural network predictions for future use
np.save('/Github/DeepLforConcreteReuse/Model/nn_predictions.npy', nn_predictions)

def create_journal_landscape_comparison(nn_predictions, xgb_predictions, actuals):
    """
    Create journal-quality comparison plots in landscape format:
    - Top row: 4 DNN subplots (one for each target)
    - Bottom row: 4 XGBoost subplots (one for each target)
    """
    
    # Short names for each attribute
    short_names = {
        'out:Leftover inventory[m2]': 'Leftover Inventory',
        'out:Displacement Average[cm]': 'Displacement Avg.',
        'out:Uncovered Wall Area[m2]': 'Uncovered Area',
        'out:Average Mapping Tolerance': 'Mapping Tolerance Avg.'
    }

    # Four completely distinct colors for journal publication
    colors = {
        'out:Leftover inventory[m2]': 'rgba(178, 24, 43, 1)',  # Deep red
        'out:Displacement Average[cm]': 'rgba(214, 96, 77, 1)',  # Soft red
        'out:Uncovered Wall Area[m2]': 'rgba(33, 102, 172, 1)',  # Deep blue
        'out:Average Mapping Tolerance': 'rgba(67, 147, 195, 1)'  # Soft blue
    }
    
    # Calculate overall R² scores
    nn_r2_overall = r2_score(actuals, nn_predictions)
    xgb_r2_overall = r2_score(actuals, xgb_predictions)
    
    # Calculate R² scores for each target to include in titles
    individual_r2_scores = {}
    for i, col in enumerate(actuals.columns):
        target_idx = actuals.columns.get_loc(col)
        nn_r2_individual = r2_score(actuals[col], nn_predictions[:, target_idx])
        xgb_r2_individual = r2_score(actuals[col], xgb_predictions[:, target_idx])
        individual_r2_scores[col] = {'nn': nn_r2_individual, 'xgb': xgb_r2_individual}
    
    # Create subplot titles for 2x4 layout with R² scores
    dnn_titles = [f"DNN - {short_names[col]} (R² = {individual_r2_scores[col]['nn']:.3f})" for col in actuals.columns]
    xgb_titles = [f"XGBoost - {short_names[col]} (R² = {individual_r2_scores[col]['xgb']:.3f})" for col in actuals.columns]
    subplot_titles = dnn_titles + xgb_titles
    
    # Create 2x4 subplots (2 rows, 4 columns)
    fig = sp.make_subplots(
        rows=2, cols=4,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.15,
        row_titles=["Deep Neural Network", "XGBoost"]
    )
    
    # Plot each target variable
    for i, col in enumerate(actuals.columns):
        col_idx = i + 1  # Column index (1-4)
        target_idx = actuals.columns.get_loc(col)
        
        # DNN subplot (top row)
        fig.add_trace(
            go.Scatter(
                x=actuals[col],
                y=nn_predictions[:, target_idx],
                mode='markers',
                marker=dict(
                    color=colors[col],
                    size=5,
                    opacity=0.7,
                    line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                ),
                name=f"DNN - {short_names[col]}",
                showlegend=False
            ),
            row=1, col=col_idx
        )
        
        # XGBoost subplot (bottom row)  
        fig.add_trace(
            go.Scatter(
                x=actuals[col],
                y=xgb_predictions[:, target_idx],
                mode='markers',
                marker=dict(
                    color=colors[col],
                    size=5,
                    opacity=0.7,
                    line=dict(width=0.5, color='rgba(0,0,0,0.3)')
                ),
                name=f"XGBoost - {short_names[col]}",
                showlegend=False
            ),
            row=2, col=col_idx
        )
        
        # Perfect prediction lines for both models
        for row_idx in [1, 2]:  # DNN and XGBoost rows
            if row_idx == 1:
                preds = nn_predictions[:, target_idx]
            else:
                preds = xgb_predictions[:, target_idx]
                
            min_val = min(actuals[col].min(), preds.min())
            max_val = max(actuals[col].max(), preds.max())
            
            fig.add_trace(
                go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    line=dict(color='black', dash='dash', width=2),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row_idx, col=col_idx
            )
    
    # Update layout for journal quality landscape format
    fig.update_layout(
        template='plotly_white',
        height=800,   # Reduced height for landscape
        width=1600,   # Increased width for landscape
        title=dict(
            text="Model Performance Comparison: Deep Neural Network vs XGBoost",
            font=dict(size=18, color='black'),
            x=0.5,
            xanchor='center',
            y=0.98
        ),
        font=dict(family="Arial", size=12, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=60, r=60, t=120, b=80)  # Increased top margin for longer titles
    )
    
    # Update all x and y axes
    fig.update_xaxes(
        title_text="Ground Truth",
        showline=True,
        linewidth=2,
        linecolor='black',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        mirror=True,
        ticks="outside",
        tickcolor='black',
        ticklen=4,
        title_font=dict(size=12)
    )
    
    fig.update_yaxes(
        title_text="Predictions", 
        showline=True,
        linewidth=2,
        linecolor='black',
        gridcolor='rgba(128,128,128,0.2)',
        gridwidth=1,
        mirror=True,
        ticks="outside",
        tickcolor='black',
        ticklen=4,
        title_font=dict(size=12)
    )
    
    # Add overall performance annotations at the bottom
    fig.add_annotation(
        text=f"Overall DNN R² = {nn_r2_overall:.4f}",
        x=0.30, y=1.1,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color='black'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=2
    )
    
    fig.add_annotation(
        text=f"Overall XGBoost R² = {xgb_r2_overall:.4f}",
        x=0.70, y=1.1,
        xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color='black'),
        bgcolor='rgba(255,255,255,0.9)',
        bordercolor='black',
        borderwidth=2
    )
    
    return fig

def print_performance_summary(nn_predictions, xgb_predictions, Y_test):
    """Print detailed performance summary for manuscript"""
    
    short_names = {
        'out:Leftover inventory[m2]': 'Leftover Inventory',
        'out:Displacement Average[cm]': 'Displacement Avg.',
        'out:Uncovered Wall Area[m2]': 'Uncovered Area',
        'out:Average Mapping Tolerance': 'Mapping Tolerance Avg.'
    }
    
    print("=" * 80)
    print("DETAILED PERFORMANCE SUMMARY FOR MANUSCRIPT")
    print("=" * 80)

    print(f"Overall Performance:")
    print(f"  Deep Neural Network: R² = {r2_score(Y_test, nn_predictions):.4f}")
    print(f"  XGBoost:            R² = {r2_score(Y_test, xgb_predictions):.4f}")
    print(f"  Performance difference: {abs(r2_score(Y_test, nn_predictions) - r2_score(Y_test, xgb_predictions)):.4f}")

    print(f"\nIndividual Target Performance:")

    for i, col in enumerate(Y_test.columns):
        nn_r2_individual = r2_score(Y_test[col], nn_predictions[:, i])
        xgb_r2_individual = r2_score(Y_test[col], xgb_predictions[:, i])
        nn_r, _ = pearsonr(Y_test[col], nn_predictions[:, i])
        xgb_r, _ = pearsonr(Y_test[col], xgb_predictions[:, i])
        
        print(f"\n{short_names[col]}:")
        print(f"  DNN:     R² = {nn_r2_individual:.4f}, r = {nn_r:.3f}")
        print(f"  XGBoost: R² = {xgb_r2_individual:.4f}, r = {xgb_r:.3f}")
        print(f"  R² Difference: {abs(nn_r2_individual - xgb_r2_individual):.4f}")

    print(f"\nFiles Generated:")
    print(f"  - journal_landscape_comparison.png (high-resolution)")
    print(f"  - journal_landscape_comparison.pdf (vector graphics)")
    print("=" * 80)

print("=" * 60)
print("GENERATING JOURNAL-QUALITY LANDSCAPE SUBPLOT COMPARISON")
print("=" * 60)

# Create the journal landscape comparison
journal_fig = create_journal_landscape_comparison(nn_predictions, xgb_predictions, Y_test)

# Display the figure
journal_fig.show()

# Save high-resolution files for journal submission
print("Saving high-resolution files...")

# PNG for raster graphics
journal_fig.write_image(
    '/Github/DeepLforConcreteReuse/Model/journal_landscape_comparison.png',
    width=1600, 
    height=800, 
    scale=3  # High DPI for journal quality
)

# PDF for vector graphics
journal_fig.write_image(
    '/Github/DeepLforConcreteReuse/Model/journal_landscape_comparison.pdf',
    width=1600, 
    height=800
)

# Print detailed performance summary
print_performance_summary(nn_predictions, xgb_predictions, Y_test)