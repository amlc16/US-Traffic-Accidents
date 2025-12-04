import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main(csv_path: Path):
	try:
		df = pd.read_csv(csv_path)
	except FileNotFoundError:
		print(f"Error: file not found: {csv_path}")
		sys.exit(2)

	# if 'Date' not in df.columns:
	# 	print(f"Error: 'Date' column not found in {csv_path}")
	# 	sys.exit(2)

	# df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
	# df.set_index('Date', inplace=True)

	# # Resample to daily frequency first, then fill missing values
	# df = df.resample('D').mean()
	# df.fillna(method='ffill', inplace=True)
	# df.fillna(method='bfill', inplace=True)

	# if 'PM2.5' not in df.columns:
	# 	print("Error: 'PM2.5' column not found after resample.")
	# 	sys.exit(2)

	# df['7_day_MA'] = df['PM2.5'].rolling(window=7, min_periods=1).mean()
	# df['30_day_MA'] = df['PM2.5'].rolling(window=30, min_periods=1).mean()
	# df['Anomaly'] = df['PM2.5'] - df['30_day_MA']
	# df['Anomaly'] = df['Anomaly'].apply(lambda x: x if abs(x) > 10 else 0)

	# plt.figure(figsize=(14, 7))
	# plt.plot(df.index, df['PM2.5'], label='Daily PM2.5', color='blue', alpha=0.5)
	# plt.plot(df.index, df['7_day_MA'], label='7-Day MA', color='orange')
	# plt.plot(df.index, df['30_day_MA'], label='30-Day MA', color='red')
	# plt.scatter(df.index, df['PM2.5'], c=df['Anomaly'].apply(lambda x: 'red' if x != 0 else 'blue'), label='Anomalies', alpha=0.6)
	# plt.title('Daily PM2.5 Levels with Moving Averages and Anomalies')
	# plt.xlabel('Date')
	# plt.ylabel('PM2.5 Levels')
	# plt.legend()
	# plt.grid(True)
	# plt.show()

	# print ("Data analysis and visualization complete.")
	# print all attributes of the dataframe
	print(df.info())	# Print information about the DataFrame
	print(df.head())	# Print the first few rows of the DataFrame
	# print number of records
	print(f"Number of records: {len(df)}")


if __name__ == '__main__':
	# Make path handling robust to spaces
	default = Path.cwd() / 'US_Accidents_March23.csv'
	csv_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else default
	main(csv_arg)