import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import plotly.graph_objects as go
import plotly.subplots as sp
import joblib
import time
import xgboost as xgb 
from tensorflow.keras.models import load_model

# Load the CSV files - using the same data as your neural network
combined_df = pd.read_csv("/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/Code/ACADIA24/combined_csv_file.csv")

# Specify targets and features - same as your neural network code
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
joblib.dump(xgb_multi, '/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/ACADIA24/TAD/xgb_model.pkl')
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

# Plot predictions using the same function from your neural network code
def plot_predictions(predictions, actuals, model_name):
    fig = go.Figure()

    # Short names for each attribute
    short_names = {
        'out:Leftover inventory[m2]': 'Leftover Inventory',
        'out:Displacement Average[cm]': 'Displacement Avg.',
        'out:Uncovered Wall Area[m2]': 'Uncovered Area',
        'out:Average Mapping Tolerance': 'Mapping Tolerance Avg.'
    }

    # Colors for each attribute
    colors = {
        'out:Leftover inventory[m2]': 'rgba(178, 24, 43, 1)',  # Deep red
        'out:Displacement Average[cm]': 'rgba(214, 96, 77, 1)',  # Soft red
        #'out:Uncovered Wall Area[m2]': 'rgba(33, 102, 172, 1)',  # Deep blue
        #'out:Average Mapping Tolerance': 'rgba(67, 147, 195, 1)'  # Soft blue
        'out:Uncovered Wall Area[m2]': 'rgba(67, 147, 195, 1)',  # Soft blue
        'out:Average Mapping Tolerance': 'rgba(6, 48, 98, 1)'  # Deep blue
    }

    # Create a scatter plot for each attribute
    for col in actuals.columns:
        fig.add_trace(go.Scatter(
            x=actuals[col],
            y=predictions[:, actuals.columns.get_loc(col)],
            mode='markers',
            marker=dict(color=colors[col], size=5, opacity=0.5),
            name=short_names[col]
        ))

        # Add line for perfect prediction
        min_val = min(actuals[col].min(), predictions[:, actuals.columns.get_loc(col)].min())
        max_val = max(actuals[col].max(), predictions[:, actuals.columns.get_loc(col)].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='black', dash='dash'),
            showlegend=False
        ))

    # Update the layout for the plot
    fig.update_layout(
        xaxis_title='Ground Truth',
        yaxis_title='Predictions',
        template='plotly_white',
        height=600,
        width=800,
        legend=dict(
            x=1,
            y=1,
            xanchor='right',
            yanchor='bottom',
            orientation='h',
        ),
        font=dict(color='black', size=18),
        xaxis=dict(showline=True, linewidth=2, linecolor='black'),
        yaxis=dict(showline=True, linewidth=2, linecolor='black')
    )
    
    # Show the figure
    fig.show()
    
    # Save the figure (optional)
    fig.write_image(f'/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/ACADIA24/TAD/{model_name}_predictions.png')

# Plot XGBoost predictions
plot_predictions(xgb_predictions, Y_test, "XGBoost")

# Create comparison plot with your neural network predictions
# You need to have access to the neural network predictions (nn_predictions)
# If they're available in memory, you can use them directly
# Otherwise, uncomment this and run your neural network code first:
# 
# # Load your neural network model
model = load_model('/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/ACADIA24/TAD/model.keras')
# 
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
np.save('/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/ACADIA24/TAD/Peer Review/Model/nn_predictions.npy', nn_predictions)

# Comparison function (if you have both predictions available)
def compare_models(nn_predictions, xgb_predictions, actuals):
    # Create subplots with 1 row and 2 columns
    fig = sp.make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Neural Network (R²: {:.4f})".format(r2_score(Y_test, nn_predictions)), 
                        "XGBoost (R²: {:.4f})".format(r2_score(Y_test, xgb_predictions))),
        horizontal_spacing=0.1
    )
    
    # Short names for each attribute
    short_names = {
        'out:Leftover inventory[m2]': 'Leftover Inventory',
        'out:Displacement Average[cm]': 'Displacement Avg.',
        'out:Uncovered Wall Area[m2]': 'Uncovered Area',
        'out:Average Mapping Tolerance': 'Mapping Tolerance Avg.'
    }

    # Colors for each attribute
    colors = {
        'out:Leftover inventory[m2]': 'rgba(178, 24, 43, 1)',  # Deep red
        'out:Displacement Average[cm]': 'rgba(214, 96, 77, 1)',  # Soft red
        #'out:Uncovered Wall Area[m2]': 'rgba(33, 102, 172, 1)',  # Deep blue
        #'out:Average Mapping Tolerance': 'rgba(67, 147, 195, 1)'  # Soft blue
        'out:Uncovered Wall Area[m2]': 'rgba(67, 147, 195, 1)',  # Soft blue
        'out:Average Mapping Tolerance': 'rgba(6, 48, 98, 1)'  # Deep blue
    }

    # First subplot (Neural Network)
    for col in actuals.columns:
        fig.add_trace(
            go.Scatter(
                x=actuals[col],
                y=nn_predictions[:, actuals.columns.get_loc(col)],
                mode='markers',
                marker=dict(color=colors[col], size=5, opacity=0.5),
                name=short_names[col],
                legendgroup=col,
                showlegend=True
            ),
            row=1, col=1
        )

        # Perfect prediction line
        min_val = min(actuals[col].min(), nn_predictions[:, actuals.columns.get_loc(col)].min())
        max_val = max(actuals[col].max(), nn_predictions[:, actuals.columns.get_loc(col)].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )

    # Second subplot (XGBoost)
    for col in actuals.columns:
        fig.add_trace(
            go.Scatter(
                x=actuals[col],
                y=xgb_predictions[:, actuals.columns.get_loc(col)],
                mode='markers',
                marker=dict(color=colors[col], size=5, opacity=0.5),
                name=short_names[col],
                legendgroup=col,
                showlegend=False
            ),
            row=1, col=2
        )

        # Perfect prediction line
        min_val = min(actuals[col].min(), xgb_predictions[:, actuals.columns.get_loc(col)].min())
        max_val = max(actuals[col].max(), xgb_predictions[:, actuals.columns.get_loc(col)].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='black', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )

    # Update layout to match the white background style
    fig.update_layout(
        template='plotly_white',
        height=600,
        width=1600,
        title_text="Model Comparison: Neural Network vs XGBoost",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial", size=18, color="black"),
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Update axes labels and add black borders to ALL axes
    fig.update_xaxes(
        title_text="Ground Truth", 
        showline=True, 
        linewidth=2, 
        linecolor='black',
        mirror=False  # This adds the opposite side border too
    )
    
    fig.update_yaxes(
        title_text="Predictions", 
        showline=True, 
        linewidth=2, 
        linecolor='black',
        mirror=False # This adds the opposite side border too
    )

    # Show the figure
    fig.show()
    
    # Save the figure (optional)
    fig.write_image('/Users/boenalan/Library/CloudStorage/GoogleDrive-boenalan@ethz.ch/My Drive/CEA/PhD/ACADIA24/TAD/model_comparison.png')

# If you have neural network predictions available, uncomment this line:
compare_models(nn_predictions, xgb_predictions, Y_test)