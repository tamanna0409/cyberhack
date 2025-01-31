import pandas as pd

# Sample data creation
data = {
    'user_id': [1, 2, 3, 4, 5],
    'account_age_days': [10, 5, 30, 2, 1],
    'friend_requests_sent': [100, 200, 5, 300, 400],
    'messages_sent': [50, 60, 10, 80, 90],
    'money_requests_sent': [5, 10, 0, 20, 25]
}

# Create DataFrame
df = pd.DataFrame(data)

from sklearn.ensemble import IsolationForest

# Feature selection
features = df[['account_age_days', 'friend_requests_sent', 'messages_sent', 'money_requests_sent']]

# Isolation Forest model for anomaly detection
model = IsolationForest(contamination=0.2)  # Adjust contamination based on expected fraud rate
df['anomaly'] = model.fit_predict(features)

def delete_fraudulent_accounts(fraudulent_accounts):
    for index in fraudulent_accounts.index:
        print(f"Deleting account with user_id: {fraudulent_accounts.loc[index, 'user_id']}")

# Identify fraudulent accounts
fraudulent_accounts = df[df['anomaly'] == -1]
delete_fraudulent_accounts(fraudulent_accounts)

def notify_user(user_id):
    print(f"Notification: Your account with user_id {user_id} has been flagged and deleted.")

# Notify users for deleted accounts
for index in fraudulent_accounts.index:
    notify_user(fraudulent_accounts.loc[index, 'user_id'])
