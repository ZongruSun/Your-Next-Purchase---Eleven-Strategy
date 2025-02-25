import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Charger les données
df1 = pd.read_csv('merged_data_with_features.csv')

# Vérifier que la colonne existe
if 'CategoryFamily' not in df1.columns:
    raise ValueError("La colonne 'CategoryFamily' est manquante dans df1")

# 🔹 1. Encoder les catégories directement
category_encoder = LabelEncoder()
df1['CategoryFamily_encoded'] = category_encoder.fit_transform(df1['CategoryFamily'])

# Sauvegarder l'encodeur pour l'utiliser plus tard
import pickle
with open("category_encoder.pkl", "wb") as f:
    pickle.dump(category_encoder, f)

# 🔹 2. Construction des séquences
sequences, next_categories = [], []
max_seq_length = 100

for _, group in tqdm(df1.groupby('ClientID'), desc="Traitement des clients"):
    category_sequence = group['CategoryFamily_encoded'].tolist()
    for i in range(1, len(category_sequence)):
        sequences.append(category_sequence[:i])
        next_categories.append(category_sequence[i])

# Padding des séquences
X_padded = [
    seq + [0] * (max_seq_length - len(seq)) if len(seq) < max_seq_length else seq[-max_seq_length:]
    for seq in sequences
]
y = next_categories

X = torch.tensor(X_padded, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# 🔹 3. Division des données
train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# 🔹 4. Création du Dataset PyTorch
class ProductRecommendationDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = ProductRecommendationDataset(X_train, y_train)
val_dataset = ProductRecommendationDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 🔹 5. Définition du modèle LSTM
class LSTMRecommender(nn.Module):
    def __init__(self, num_categories, embed_size=50, hidden_size=64):
        super(LSTMRecommender, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_categories)

    def forward(self, x):
        embedded = self.category_embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output

num_categories = len(category_encoder.classes_)

# Initialiser le modèle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRecommender(num_categories).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 🔹 6. Entraînement du modèle
num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_loader)}")

# 🔹 7. Sauvegarder le modèle
torch.save(model.state_dict(), "model.pth")
print("✅ Modèle entraîné et sauvegardé sous 'model.pth' !")
