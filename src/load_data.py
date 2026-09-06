import pandas as pd

# Load features - no header row in this file, so we name columns ourselves
features = pd.read_csv('data/elliptic_txs_features.csv', header=None)
features.columns = ['txId', 'timestep'] + [f'feat_{i}' for i in range(165)]

# Load labels - force class as string since the CSV mixes '1'/'2'/'unknown'
classes = pd.read_csv('data/elliptic_txs_classes.csv', dtype={'class': str})

# Load the graph edges (which transaction sent money to which)
edges = pd.read_csv('data/elliptic_txs_edgelist.csv')

# Merge features with labels
df = features.merge(classes, on='txId', how='left')

# Drop transactions with unknown label - we can only train on labeled ones
df = df[df['class'] != 'unknown'].copy()
df['class'] = df['class'].map({'1': 1, '2': 0})  # 1 = illicit, 0 = licit

print("Total labeled transactions:", len(df))
print("Illicit count:", (df['class'] == 1).sum())
print("Licit count:", (df['class'] == 0).sum())
print(df.head())
