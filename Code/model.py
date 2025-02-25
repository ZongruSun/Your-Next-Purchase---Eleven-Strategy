import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ProductRecommendationDataset(Dataset):
    """
    Dataset personnalisé pour la recommandation de produits.
    """

    def __init__(self, X, y):
        """
        Initialise le dataset avec les séquences (X) et les catégories suivantes (y).

        Args:
            X (torch.Tensor): Séquences de catégories encodées.
            y (torch.Tensor): Cibles des catégories suivantes.
        """
        self.X = X
        self.y = y

    def __len__(self):
        """
        Retourne la taille du dataset.

        Returns:
            int: Taille du dataset.
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Retourne un élément du dataset.

        Args:
            idx (int): Index de l'élément à retourner.

        Returns:
            tuple: Séquence d'entrée et catégorie cible.
        """
        return self.X[idx], self.y[idx]


class LSTMRecommender(nn.Module):
    """
    Modèle de recommandation basé sur LSTM.
    """

    def __init__(self, num_categories, embed_size=50, hidden_size=64):
        """
        Initialise le modèle avec une couche d'embedding, une LSTM et une couche de sortie.

        Args:
            num_categories (int): Nombre de catégories différentes.
            embed_size (int): Taille de l'embedding pour chaque catégorie.
            hidden_size (int): Taille du vecteur caché de la LSTM.
        """
        super(LSTMRecommender, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_categories)

    def forward(self, x):
        """
        Définit le passage avant du modèle.

        Args:
            x (torch.Tensor): Séquences d'entrée (Catégories encodées).

        Returns:
            torch.Tensor: Sortie du modèle (probabilités des catégories).
        """
        embedded = self.category_embedding(x)
        lstm_out, _ = self.lstm(embedded)
        output = self.fc(lstm_out[:, -1, :])
        return output


def recall_at_k(preds, targets, k=3):
    """
    Calcul de la métrique Recall@k.

    Args:
        preds (torch.Tensor): Prédictions du modèle.
        targets (torch.Tensor): Véritables catégories cibles.
        k (int): Nombre de meilleures prédictions à prendre en compte.

    Returns:
        float: Recall@k.
    """
    top_k_preds = torch.topk(preds, k, dim=1).indices
    matches = torch.any(top_k_preds == targets.unsqueeze(1), dim=1)
    return matches.float().mean().item()


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3):
    """
    Entraîne le modèle.

    Args:
        model (nn.Module): Le modèle à entraîner.
        train_loader (DataLoader): DataLoader pour les données d'entraînement.
        val_loader (DataLoader): DataLoader pour les données de validation.
        criterion (nn.Module): La fonction de perte.
        optimizer (optim.Optimizer): L'optimiseur.
        num_epochs (int): Le nombre d'époques.

    Returns:
        list, list: Liste des pertes d'entraînement et des pertes de validation.
    """
    train_losses, val_losses, val_recalls = [], [], []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1} - Training Loss: {running_loss / len(train_loader)}")

        model.eval()
        val_loss, val_recall = 0.0, 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, targets).item()
                val_recall += recall_at_k(outputs, targets, k=3)

        val_losses.append(val_loss / len(val_loader))
        val_recalls.append(val_recall / len(val_loader))
        print(f"Epoch {epoch+1} - Validation Loss: {val_losses[-1]} - Recall@3: {val_recalls[-1]}")

    return train_losses, val_losses, val_recalls


def recommend(client_id, model, df1, df_stocks, category_family_encoder, max_seq_length=100, k=3):
    """
    Fonction de recommandation de produits pour un client.

    Args:
        client_id (int): ID du client pour lequel faire la recommandation.
        model (nn.Module): Le modèle de recommandation.
        df1 (pd.DataFrame): Dataframe avec les informations des clients et produits.
        df_stocks (pd.DataFrame): Dataframe avec les informations de stock.
        category_family_encoder (LabelEncoder): Encodeur pour inverser les catégories.
        max_seq_length (int): Longueur maximale des séquences.
        k (int): Nombre de recommandations à faire.

    Returns:
        tuple: Top k catégories recommandées, nombre de catégories bien prédites,
               catégories réellement achetées, nombre de produits disponibles et produits recommandés.
    """
    model.eval()
    client_data = df1[df1['ClientID'] == client_id]
    if client_data.empty:
        return []

    sequence = client_data['CategoryFamily_encoded'].tolist()
    sequence = sequence[-max_seq_length:]
    input_seq = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_seq)

    top_k_categories = output.squeeze().topk(k).indices.cpu().numpy()
    top_k_names = category_family_encoder.inverse_transform(top_k_categories)

    actual_categories = set(client_data['CategoryFamily'])
    correctly_predicted = sum(1 for cat in top_k_names if cat in actual_categories)

    client_country = client_data['ClientCountry'].iloc[0]
    products_bought = set(client_data['ProductID'])
    recommended_products = set(df1[df1['CategoryFamily'].isin(top_k_names)]['ProductID'].unique()) - products_bought
    available_products = df_stocks[(df_stocks['ProductID'].isin(recommended_products)) & 
                                   (df_stocks['Quantity'] > 0) & 
                                   (df_stocks['StoreCountry'] == client_country)]

    return top_k_names, correctly_predicted, actual_categories, len(available_products), available_products['ProductID'].unique()[:3]


# Example of loading data and using the model

def load_data():
    # Load data as per your requirement (e.g., df1, df_stocks, category_family_encoder)
    df1 = pd.read_csv('merged_data_with_features.csv')
    df_stocks = pd.read_csv('stocks.csv')
    
    # Assuming 'CategoryFamily_encoded' exists in your data
    category_family_encoder = LabelEncoder()
    df1['CategoryFamily_encoded'] = category_family_encoder.fit_transform(df1['CategoryFamily'])
    
    return df1, df_stocks, category_family_encoder


if __name__ == "__main__":
    # Example of loading data
    df1, df_stocks, category_family_encoder = load_data()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Example usage of the model
    num_categories = len(category_family_encoder.classes_)
    model = LSTMRecommender(num_categories)
    model.to(device)
    
    # Create datasets
    train_loader, val_loader = create_data_loaders(df1)
    
    # Train the model
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=3)
    
    # Recommend products for a specific client
    client_id_test = 39370740138294
    top_categories, correct_count, actual_categories, total_available, recommended_products = recommend(
        client_id_test, model, df1, df_stocks, category_family_encoder, k=3
    )
    
    print(f"Top 3 catégories recommandées : {top_categories}")
    print(f"Top 3 produits recommandés : {recommended_products}")
