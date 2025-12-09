import os
import glob
import joblib
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    confusion_matrix,
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score
)

# Configuration paths
OHLCV_DATA_DIRECTORY = "./ohlcv_data"
MODEL_OUTPUT_DIRECTORY = "./models"

# Create models directory if it doesn't exist
os.makedirs(MODEL_OUTPUT_DIRECTORY, exist_ok=True)


def find_latest_csv_file(directory, file_pattern):
    """
    Find the most recently created CSV file matching the given pattern.
    
    Args:
        directory (str): Directory to search in
        file_pattern (str): Glob pattern for file names (e.g., "*_5m_*.csv")
    
    Returns:
        str: Full path to the latest matching file, or None if no matches found
    """
    matching_files = glob.glob(os.path.join(directory, file_pattern))
    
    if not matching_files:
        return None
    
    # Return the file with the most recent creation time
    return max(matching_files, key=os.path.getctime)


def _load_and_resample_5m_to_1h(csv_file_path):
    """
    Load 5-minute OHLCV data and resample it to 1-hour bars.
    
    Args:
        csv_file_path (str): Path to the 5-minute CSV file
    
    Returns:
        pd.DataFrame: Hourly OHLCV data
    """
    # Read CSV with datetime index
    dataframe = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    
    # Ensure the index is a proper DatetimeIndex
    try:
        dataframe.index = pd.to_datetime(dataframe.index)
    except Exception:
        # Fallback: coerce invalid dates to NaT and drop them
        dataframe.index = pd.to_datetime(dataframe.index, errors='coerce')
    
    # Remove rows with invalid (NaT) timestamps
    dataframe = dataframe[~dataframe.index.isna()]
    dataframe = dataframe.sort_index()
    
    # Convert OHLCV columns to numeric type (handle any string values)
    for column_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')
    
    # Resample to 1-hour bars using OHLC aggregation
    hourly_open = dataframe['Open'].resample('1H').first()
    hourly_high = dataframe['High'].resample('1H').max()
    hourly_low = dataframe['Low'].resample('1H').min()
    hourly_close = dataframe['Close'].resample('1H').last()
    hourly_volume = dataframe['Volume'].resample('1H').sum()
    
    # Combine into a single DataFrame
    hourly_dataframe = pd.DataFrame({
        'Open': hourly_open,
        'High': hourly_high,
        'Low': hourly_low,
        'Close': hourly_close,
        'Volume': hourly_volume
    })
    
    # Remove any rows with missing values
    hourly_dataframe = hourly_dataframe.dropna()
    
    file_name = os.path.basename(csv_file_path)
    print(f"✓ Loaded 5-minute data and resampled to 1-hour from: {file_name}")
    
    return hourly_dataframe


def _load_1h_data(csv_file_path):
    dataframe = pd.read_csv(csv_file_path, index_col=0, parse_dates=True)
    
    # Ensure the index is a proper DatetimeIndex
    try:
        dataframe.index = pd.to_datetime(dataframe.index)
    except Exception:
        dataframe.index = pd.to_datetime(dataframe.index, errors='coerce')
    
    # Remove rows with invalid timestamps and sort
    dataframe = dataframe[~dataframe.index.isna()]
    dataframe = dataframe.sort_index()
    
    # Convert OHLCV columns to numeric type
    for column_name in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if column_name in dataframe.columns:
            dataframe[column_name] = pd.to_numeric(dataframe[column_name], errors='coerce')
    
    # Remove any rows with missing values
    dataframe = dataframe.dropna()
    
    file_name = os.path.basename(csv_file_path)
    print(f"✓ Loaded 1-hour data from: {file_name}")
    
    return dataframe


def load_and_prepare_ohlcv_data():
    # Look for 5-minute and 1-hour CSV files
    csv_file_5m = find_latest_csv_file(OHLCV_DATA_DIRECTORY, "*_5m_*.csv")
    csv_file_1h = find_latest_csv_file(OHLCV_DATA_DIRECTORY, "*_1h_*.csv")

    # Prefer 5-minute data (resample to 1-hour for finer granularity)
    if csv_file_5m:
        return _load_and_resample_5m_to_1h(csv_file_5m)

    # Fall back to 1-hour data if 5-minute is unavailable
    if csv_file_1h:
        return _load_1h_data(csv_file_1h)

    # Error if neither file type is found
    error_message = "No OHLCV CSV files found in ./ohlcv_data. Please run data-iteration.py first."
    raise FileNotFoundError(error_message)


def create_labeled_features(
    hourly_data,
    spike_threshold_multiplier=2.0,
    real_move_price_threshold=0.01,
    historical_lookback_hours=24
):
    """
    Engineer features from overlapping 2-hour windows and label them as 
    'real' or 'pump' based on price movement.
    
    Algorithm:
    ----------
    1. For each overlapping 2-hour window [hour_i, hour_i+1]:
       - Identify which hour has higher volume (spike_hour)
       - Check if spike volume > baseline_mean + spike_threshold_multiplier * baseline_std
       - If spike detected:
         * Calculate 2-hour price movement
         * Label as 'real' (1) if |price_move| >= real_move_price_threshold
         * Otherwise label as 'pump' (0)
    
    Args:
        hourly_data (pd.DataFrame): Hourly OHLCV data
        spike_threshold_multiplier (float): Std deviations above mean to trigger spike detection
        real_move_price_threshold (float): Minimum price movement to classify as real (e.g., 0.01 = 1%)
        historical_lookback_hours (int): Hours to use for baseline volume statistics
    
    Returns:
        tuple: (X, y, metadata_df)
            - X (pd.DataFrame): Feature matrix with 7 engineered features
            - y (pd.Series): Binary labels (0 = pump, 1 = real)
            - metadata_df (pd.DataFrame): Debug info including timestamps and raw metrics
    """
    training_examples = []

    # Precompute rolling baseline statistics for volume
    volume_baseline_mean = hourly_data['Volume'].rolling(
        window=historical_lookback_hours, 
        min_periods=6
    ).mean()
    
    volume_baseline_std = hourly_data['Volume'].rolling(
        window=historical_lookback_hours, 
        min_periods=6
    ).std()

    # Iterate through overlapping 2-hour windows
    # Window structure: [hour_i, hour_i+1]
    for window_index in range(1, len(hourly_data)):
        two_hour_window = hourly_data.iloc[window_index - 1 : window_index + 1]
        
        if len(two_hour_window) < 2:
            continue

        # Extract individual hours
        first_hour = two_hour_window.iloc[0]
        second_hour = two_hour_window.iloc[1]
        
        # Identify which hour had the volume spike
        first_hour_volume = first_hour['Volume']
        second_hour_volume = second_hour['Volume']
        
        if first_hour_volume >= second_hour_volume:
            spike_hour_index = 0
            spike_hour_volume = first_hour_volume
        else:
            spike_hour_index = 1
            spike_hour_volume = second_hour_volume
        
        spike_hour_timestamp = two_hour_window.index[spike_hour_index]

        # Get baseline statistics for this spike hour
        if spike_hour_timestamp in volume_baseline_mean.index:
            baseline_mean_volume = volume_baseline_mean.loc[spike_hour_timestamp]
        else:
            baseline_mean_volume = np.nan

        if spike_hour_timestamp in volume_baseline_std.index:
            baseline_std_volume = volume_baseline_std.loc[spike_hour_timestamp]
        else:
            baseline_std_volume = np.nan

        # Detect if this hour's volume is a spike
        is_volume_spike = False
        if not np.isnan(baseline_mean_volume) and not np.isnan(baseline_std_volume):
            spike_threshold = baseline_mean_volume + (spike_threshold_multiplier * baseline_std_volume)
            is_volume_spike = spike_hour_volume > spike_threshold

        # Skip windows without detected volume spikes
        # (We only want to learn from fused-volume events)
        if not is_volume_spike:
            continue

        # Calculate 2-hour price movement
        opening_price = two_hour_window['Close'].iloc[0]
        closing_price = two_hour_window['Close'].iloc[-1]
        two_hour_return = (closing_price / opening_price) - 1.0
        absolute_two_hour_return = abs(two_hour_return)

        # Label the window: real move vs pump
        # Real (1): significant price movement with the volume spike
        # Pump (0): volume spike without meaningful price movement
        if absolute_two_hour_return >= real_move_price_threshold:
            movement_label = 1  # Real move
        else:
            movement_label = 0  # Pump (false volume)

        # Engineer features for this window
        total_volume_2h = two_hour_window['Volume'].sum()
        
        if not np.isnan(baseline_mean_volume):
            volume_spike_ratio = spike_hour_volume / (baseline_mean_volume + 1e-9)
        else:
            volume_spike_ratio = np.nan

        # Intra-hour returns (close/open - 1)
        first_hour_return = (first_hour['Close'] / first_hour['Open']) - 1.0
        second_hour_return = (second_hour['Close'] / second_hour['Open']) - 1.0

        # Recent volatility: standard deviation of close-to-close returns
        close_to_close_returns = hourly_data['Close'].pct_change()
        recent_volatility_series = close_to_close_returns.rolling(
            window=historical_lookback_hours,
            min_periods=6
        ).std()
        
        if spike_hour_timestamp in recent_volatility_series.index:
            recent_volatility = recent_volatility_series.loc[spike_hour_timestamp]
        else:
            recent_volatility = np.nan

        # Store the example
        training_examples.append({
            'timestamp': two_hour_window.index[-1],
            'spike_volume': spike_hour_volume,
            'total_volume_2h': total_volume_2h,
            'volume_spike_ratio': volume_spike_ratio,
            'first_hour_return': first_hour_return,
            'second_hour_return': second_hour_return,
            'baseline_volume_mean': baseline_mean_volume,
            'recent_volatility': recent_volatility,
            'absolute_price_return_2h': absolute_two_hour_return,
            'label': movement_label,
        })

    # Convert to DataFrame
    examples_dataframe = pd.DataFrame(training_examples)
    
    if examples_dataframe.empty:
        error_msg = (
            "No labeled windows found! Try lowering spike_threshold_multiplier "
            "or real_move_price_threshold, or ensure data has sufficient length."
        )
        raise ValueError(error_msg)

    # Select features for model training
    feature_columns = [
        'spike_volume',
        'total_volume_2h',
        'volume_spike_ratio',
        'first_hour_return',
        'second_hour_return',
        'baseline_volume_mean',
        'recent_volatility'
    ]
    
    feature_matrix = examples_dataframe[feature_columns]
    target_labels = examples_dataframe['label']
    
    # Fill any remaining NaN values with 0 (edge cases)
    feature_matrix = feature_matrix.fillna(0.0)
    
    return feature_matrix, target_labels, examples_dataframe


def train_and_evaluate_model(feature_matrix, target_labels):
    """
    Train a RandomForest classifier using 90/10 train/test split and 
    evaluate performance using multiple metrics.
    
    Args:
        feature_matrix (pd.DataFrame): Input features (7 columns)
        target_labels (pd.Series): Binary labels (0 or 1)
    
    Returns:
        sklearn.pipeline.Pipeline: Trained pipeline containing StandardScaler 
                                   and RandomForestClassifier
    """
    # Split data: 90% training, 10% testing
    print("\nSplitting data: 90% train, 10% test...")
    (
        training_features,
        testing_features,
        training_labels,
        testing_labels
    ) = train_test_split(
        feature_matrix,
        target_labels,
        test_size=0.1,
        shuffle=False,  # Preserve temporal order
        random_state=42
    )

    # Build pipeline with feature scaling and classification
    model_pipeline = Pipeline([
        ('feature_scaler', StandardScaler()),
        (
            'classifier',
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight='balanced',  # Handle class imbalance
                n_jobs=-1  # Use all CPU cores
            )
        )
    ])

    # Train the model
    print("Training RandomForest classifier...")
    model_pipeline.fit(training_features, training_labels)

    # Make predictions on test set
    test_predictions = model_pipeline.predict(testing_features)
    test_probabilities = model_pipeline.predict_proba(testing_features)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(testing_labels, test_predictions)
    precision = precision_score(testing_labels, test_predictions, zero_division=0)
    recall = recall_score(testing_labels, test_predictions, zero_division=0)
    f1 = f1_score(testing_labels, test_predictions, zero_division=0)

    # Print metrics
    print("\n" + "=" * 60)
    print("MODEL EVALUATION (Test Set - 10% of Data)")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("=" * 60 + "\n")

    # Detailed classification report
    print("Detailed Classification Report:")
    print("-" * 60)
    print(classification_report(
        testing_labels,
        test_predictions,
        zero_division=0
    ))

    # ROC AUC score (if binary classification and both classes present)
    if test_probabilities is not None and len(np.unique(testing_labels)) > 1:
        try:
            roc_auc = roc_auc_score(testing_labels, test_probabilities)
            print(f"ROC AUC Score: {roc_auc:.4f}")
        except Exception:
            pass

    # Confusion matrix
    print("\nConfusion Matrix:")
    print("-" * 60)
    conf_matrix = confusion_matrix(testing_labels, test_predictions)
    print(conf_matrix)
    
    # Only print interpretation if we have a proper 2x2 matrix
    if conf_matrix.shape == (2, 2):
        print("\nInterpretation:")
        print(f"  True Negatives (Correctly predicted pump) : {conf_matrix[0, 0]}")
        print(f"  False Positives (Pump predicted as real)  : {conf_matrix[0, 1]}")
        print(f"  False Negatives (Real predicted as pump)  : {conf_matrix[1, 0]}")
        print(f"  True Positives (Correctly predicted real) : {conf_matrix[1, 1]}")
    else:
        print("\nNote: Confusion matrix is not 2x2 (possible class imbalance in test set)")

    return model_pipeline


def save_trained_model(trained_pipeline):
    """
    Save the trained model pipeline to disk using joblib.
    
    Args:
        trained_pipeline (sklearn.pipeline.Pipeline): The trained model
    
    Returns:
        str: Path to the saved model file
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_file_path = os.path.join(
        MODEL_OUTPUT_DIRECTORY,
        f'model_rf_{timestamp}.joblib'
    )
    
    joblib.dump(trained_pipeline, model_file_path)
    print(f"\n✓ Model successfully saved to: {model_file_path}")
    
    return model_file_path


def main():
    """
    Main execution function: orchestrates the entire training pipeline.
    """
    print("\n" + "=" * 60)
    print("FADE MODEL TRAINING PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    print("\n[1/4] Loading OHLCV data...")
    hourly_ohlcv = load_and_prepare_ohlcv_data()
    print(f"      Loaded {len(hourly_ohlcv)} hourly candles")

    # Step 2: Engineer features
    print("\n[2/4] Engineering features and labels from 2-hour windows...")
    features, labels, metadata = create_labeled_features(
        hourly_ohlcv,
        spike_threshold_multiplier=2.0,
        real_move_price_threshold=0.01,
        historical_lookback_hours=24
    )
    
    positive_class_ratio = labels.mean()
    print(f"      Generated {len(features)} training examples")
    print(f"      Class distribution - Pump: {1 - positive_class_ratio:.1%}, Real: {positive_class_ratio:.1%}")

    # Step 3: Train model
    print("\n[3/4] Training model...")
    trained_model = train_and_evaluate_model(features, labels)

    # Step 4: Save model
    print("\n[4/4] Saving model...")
    model_path = save_trained_model(trained_model)

    print("\n" + "=" * 60)
    print("✓ Training pipeline complete!")
    print(f"  Model saved: {model_path}")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()

