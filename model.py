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
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


DATA_DIR = "./ohlcv_data"
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)


def get_latest_csv(directory, pattern):
	files = glob.glob(os.path.join(directory, pattern))
	if not files:
		return None
	return max(files, key=os.path.getctime)


def load_best_ohlcv():
	"""Load the best available OHLCV source.

	Prefer 5m file (resample to 1h), otherwise use 1h file directly.
	"""
	csv_5m = get_latest_csv(DATA_DIR, "*_5m_*.csv")
	csv_1h = get_latest_csv(DATA_DIR, "*_1h_*.csv")

	if csv_5m:
		df = pd.read_csv(csv_5m, index_col=0, parse_dates=True)
		# Ensure index is datetime (some CSVs may have string indexes)
		try:
			df.index = pd.to_datetime(df.index)
		except Exception:
			# fallback: coerce and drop invalid
			df.index = pd.to_datetime(df.index, errors='coerce')
		# drop rows with invalid timestamps
		df = df[~df.index.isna()]
		df = df.sort_index()
		# Ensure numeric
		for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
			if c in df.columns:
				df[c] = pd.to_numeric(df[c], errors='coerce')
		# Resample to 1H (index is now a DatetimeIndex)
		ohlc = df['Close'].resample('1H').ohlc()
		volume = df['Volume'].resample('1H').sum()
		df1h = pd.concat([ohlc, volume], axis=1)
		df1h.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
		df1h = df1h.dropna()
		print(f"Loaded 5m -> resampled to 1h from {os.path.basename(csv_5m)}")
		return df1h

	if csv_1h:
		df = pd.read_csv(csv_1h, index_col=0, parse_dates=True)
		# Ensure index is datetime
		try:
			df.index = pd.to_datetime(df.index)
		except Exception:
			df.index = pd.to_datetime(df.index, errors='coerce')
		# drop invalid timestamps and sort
		df = df[~df.index.isna()]
		df = df.sort_index()
		for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
			if c in df.columns:
				df[c] = pd.to_numeric(df[c], errors='coerce')
		df = df.dropna()
		print(f"Loaded 1h from {os.path.basename(csv_1h)}")
		return df

	raise FileNotFoundError("No OHLCV CSV files found in ./ohlcv_data")


def engineer_features(df1h, spike_k=2.0, price_move_threshold=0.01, lookback_hours=24):
	"""
	Build examples from overlapping 2-hour windows.

	Returns X (DataFrame of features) and y (0 pump, 1 real).
	"""
	rows = []

	# Precompute rolling stats for volume baseline
	vol_mean_24 = df1h['Volume'].rolling(lookback_hours, min_periods=6).mean()
	vol_std_24 = df1h['Volume'].rolling(lookback_hours, min_periods=6).std()

	# We'll iterate from t=1 to end-1 to get 2-hour windows [t-1, t]
	for i in range(1, len(df1h)):
		win = df1h.iloc[i-1:i+1]  # two rows
		if len(win) < 2:
			continue

		# Identify the 1-hour within the 2-hour window that has the higher volume
		hour0 = win.iloc[0]
		hour1 = win.iloc[1]
		spike_hour = 0 if hour0['Volume'] >= hour1['Volume'] else 1
		spike_vol = win['Volume'].iloc[spike_hour]
		spike_idx = win.index[spike_hour]

		# Compute baseline mean/std from vol_mean_24 at spike index
		baseline_mean = vol_mean_24.loc[spike_idx] if spike_idx in vol_mean_24.index else np.nan
		baseline_std = vol_std_24.loc[spike_idx] if spike_idx in vol_std_24.index else np.nan

		is_spike = False
		if not np.isnan(baseline_mean) and not np.isnan(baseline_std):
			is_spike = spike_vol > (baseline_mean + spike_k * baseline_std)

		# If no spike, skip (we only want windows with fused-volume events)
		if not is_spike:
			continue

		# Price move across the 2-hour window
		price_start = win['Close'].iloc[0]
		price_end = win['Close'].iloc[-1]
		ret2h = (price_end / price_start) - 1.0
		abs_ret2h = abs(ret2h)

		label = 1 if abs_ret2h >= price_move_threshold else 0

		# Features
		total_vol_2h = win['Volume'].sum()
		vol_ratio = spike_vol / (baseline_mean + 1e-9) if not np.isnan(baseline_mean) else np.nan
		hour0_ret = (hour0['Close'] / hour0['Open']) - 1.0
		hour1_ret = (hour1['Close'] / hour1['Open']) - 1.0
		recent_vol_mean = vol_mean_24.loc[spike_idx] if spike_idx in vol_mean_24.index else np.nan

		# volatility: std of returns over past lookback_hours (using close returns)
		close_returns = df1h['Close'].pct_change().rolling(lookback_hours, min_periods=6).std()
		recent_volatility = close_returns.loc[spike_idx] if spike_idx in close_returns.index else np.nan

		rows.append({
			'timestamp': win.index[-1],
			'spike_vol': spike_vol,
			'total_vol_2h': total_vol_2h,
			'vol_ratio': vol_ratio,
			'hour0_ret': hour0_ret,
			'hour1_ret': hour1_ret,
			'recent_vol_mean': recent_vol_mean,
			'recent_volatility': recent_volatility,
			'abs_ret2h': abs_ret2h,
			'label': label,
		})

	df = pd.DataFrame(rows)
	if df.empty:
		raise ValueError("No labeled windows found. Try lowering thresholds or ensure data length is sufficient.")

	X = df[['spike_vol', 'total_vol_2h', 'vol_ratio', 'hour0_ret', 'hour1_ret', 'recent_vol_mean', 'recent_volatility']]
	y = df['label']
	X = X.fillna(0.0)
	return X, y, df


def train_and_evaluate(X, y):
	# Use 90/10 train/test split as requested
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=False)

	pipe = Pipeline([
		('scaler', StandardScaler()),
		('clf', RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced'))
	])

	pipe.fit(X_train, y_train)

	y_pred = pipe.predict(X_test)
	y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, 'predict_proba') else None

	# Compute requested metrics
	acc = accuracy_score(y_test, y_pred)
	prec = precision_score(y_test, y_pred, zero_division=0)
	rec = recall_score(y_test, y_pred, zero_division=0)
	f1 = f1_score(y_test, y_pred, zero_division=0)

	print("\n=== Evaluation on test set (90/10 split) ===")
	print(f"Accuracy : {acc:.4f}")
	print(f"Precision: {prec:.4f}")
	print(f"Recall   : {rec:.4f}")
	print(f"F1 Score : {f1:.4f}\n")

	print("Detailed classification report:")
	print(classification_report(y_test, y_pred, zero_division=0))

	if y_proba is not None and len(np.unique(y_test)) > 1:
		try:
			auc = roc_auc_score(y_test, y_proba)
			print(f"ROC AUC: {auc:.4f}")
		except Exception:
			pass

	print("Confusion matrix:")
	print(confusion_matrix(y_test, y_pred))

	return pipe


def save_model(pipe):
	ts = datetime.now().strftime('%Y%m%d_%H%M%S')
	path = os.path.join(MODEL_DIR, f'model_rf_{ts}.joblib')
	joblib.dump(pipe, path)
	print(f"Model saved to {path}")
	return path


def main():
	print("Loading data...")
	df1h = load_best_ohlcv()

	print("Engineering features and labels...")
	X, y, meta = engineer_features(df1h, spike_k=2.0, price_move_threshold=0.01)

	print(f"Built dataset with {len(X)} examples. Positive class ratio: {y.mean():.3f}")

	print("Training model...")
	pipe = train_and_evaluate(X, y)

	save_model(pipe)


if __name__ == '__main__':
	main()

