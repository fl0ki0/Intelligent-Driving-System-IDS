import csv
from collections import defaultdict
from datetime import datetime
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
import matplotlib.dates as mdates


def read_and_aggregate_data(csv_filename):
    aggregated_data = defaultdict(lambda: {'blink_count': 0, 'yawn_count': 0, 'drowsiness_count': 0, 'out_of_frame': 0, 'fatigue_status_counts': defaultdict(int)})
    
    with open(csv_filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            date = datetime.strptime(row['Time'].split(' ')[0], '%Y-%m-%d').date()
            aggregated_data[date]['blink_count'] += int(row['Blink Count'])
            aggregated_data[date]['yawn_count'] += int(row['Yawn Count'])
            aggregated_data[date]['drowsiness_count'] += int(row['Drowsiness Count'])
            aggregated_data[date]['out_of_frame'] += int(row['Out of Frame'])
            aggregated_data[date]['fatigue_status_counts'][row['Fatigue Status']] += 1
    
    for date, data in aggregated_data.items():
        data['fatigue_status'] = max(data['fatigue_status_counts'], key=data['fatigue_status_counts'].get)
        del data['fatigue_status_counts']  
    
    return aggregated_data

def generate_attentiveness_insights(date, data):
    print(f"Generating insights for {date}:")
    print(f"daily_summary: On {date}, you blinked {data['blink_count']} times, yawned {data['yawn_count']} times, had {data['drowsiness_count']} drowsiness incidents, and were out of frame {data['out_of_frame']} times.")
    if data['yawn_count'] > data['blink_count']:
        print("trend_insight: You tend to yawn more frequently than blink, which might indicate tiredness.")
    else:
        print("trend_insight: Your blinking rate is higher than yawn count, indicating general alertness.")
    if data['drowsiness_count'] > 5:
        print("recommendation: Consider taking more frequent breaks or adjusting your driving schedule to avoid drowsiness.")
    else:
        print("recommendation: Keep up the good work! Your attentiveness levels are commendable.")
    if data['fatigue_status'] == "Very Tired":
        print("fatigue_status_overview: You were very tired on this day. Ensure you get some rest before your next drive.")
    elif data['fatigue_status'] == "Tired":
        print("fatigue_status_overview: You showed signs of tiredness. Stay hydrated and take short breaks.")
    else:
        print("fatigue_status_overview: Your fatigue status is normal. Continue driving safely.")
    print("\n")  

csv_filename = 'attentiveness_data.csv'
aggregated_data = read_and_aggregate_data(csv_filename)

for date in sorted(aggregated_data.keys()):
    generate_attentiveness_insights(date, aggregated_data[date])

def convert_to_dataframe(aggregated_data):
    df = pd.DataFrame.from_dict(aggregated_data, orient='index')
    df.index = pd.to_datetime(df.index)
    df['day_of_week'] = df.index.day_name()
    df['month'] = df.index.month
    return df

def perform_correlation_analysis(df):
    correlation_matrix = df[['blink_count', 'yawn_count', 'drowsiness_count']].corr()
    sns.heatmap(correlation_matrix, annot=True)
    plt.show()

def train_predictive_model(df):
    X = df[['blink_count', 'yawn_count']]
    y = df['drowsiness_count']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def detect_anomalies(df):
    clf = IsolationForest(random_state=42)
    df['anomaly'] = clf.fit_predict(df[['drowsiness_count']])
    anomalies = df[df['anomaly'] == -1]
    return anomalies

def visualize_data(df):
    plt.figure(figsize=(10, 7))
    plt.scatter(df.index, df['blink_count'], color='blue', label='Blink Count')
    plt.scatter(df.index, df['yawn_count'], color='red', label='Yawn Count')
    plt.scatter(df.index, df['drowsiness_count'], color='green', label='Drowsiness Count')
    plt.gca().xaxis.set_major_locator(mdates.DayLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.title('Driver Behavior Metrics Over Time')
    plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()

df = convert_to_dataframe(aggregated_data)


model = train_predictive_model(df)
anomalies = detect_anomalies(df)

visualize_data(df)


