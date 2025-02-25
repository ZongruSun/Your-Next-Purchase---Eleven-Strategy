# Your-Next-Purchase---Eleven-Strategy

## 📌 Project Overview
This project is an **LSTM-based product recommendation system** designed to predict the product categories a customer is likely to purchase next, based on their purchase history. The system uses **Streamlit** as a front-end interface and PyTorch for deep learning inference.

## 🚀 Key Features
- **Predict future purchase categories** using an LSTM model based on historical data.
- **Recommend relevant products** by filtering available stock.
- **User-friendly web interface** where users input a client ID to get recommendations.

---

## 📂 File Structure
```plaintext
📦 Project Structure
│── app.py                # Streamlit UI, loads model, calls recommend()
│── data_loader.py        # Loads CSV data
│── model.py              # LSTM model definition & recommendation logic
│── train_model.py        # Trains LSTM model, generates model.pth
│── data/
│   ├── merged_data_with_features.csv  # Transaction & user history data
│   ├── stocks.csv                     # Stock information
│── model.pth            # Trained model file
│── category_encoder.pkl # Saved label encoder from training
│── requirements.txt     # Python dependencies
```

---

## 🛠️ Installation & Setup
### **1️⃣ Clone the Repository & Navigate to Directory**
```bash
git clone https://github.com/ZongruSun/Your-Next-Purchase---Eleven-Strategy.git
cd Your-Next-Purchase---Eleven-Strategy
```

### **2️⃣ Install Python Dependencies**
Ensure you have Python 3.8+ installed, then run:
```bash
pip install -r requirements.txt
```

---

## 📌 **Usage Guide**

### **1️⃣ Train the LSTM Model (Run Once)**
```bash
python train_model.py
```
- Reads `merged_data_with_features.csv`
- Trains `LSTMRecommender` and optimizes `CrossEntropyLoss`
- Saves the trained model to `model.pth`


### **2️⃣ Run the Streamlit Recommendation System**
```bash
streamlit run app.py
```
- Loads `model.pth`
- Accepts user input (Client ID)
- **Predicts recommended categories & products**
- **Displays results on the web interface**

---

## 📊 **Model Architecture**
This project uses an **LSTM (Long Short-Term Memory) network** for sequence modeling.
- **Input**: Encoded purchase category history.
- **LSTM Processing**: Captures temporal purchase patterns.
- **Output**: Predicts next likely purchase category.
- **Product Recommendation**: Filters available products based on stock data.

```python
class LSTMRecommender(nn.Module):
    def __init__(self, num_categories, embed_size=50, hidden_size=64):
        super(LSTMRecommender, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_categories)
```

---

## 📌 **API & Code Example**

### **1️⃣ Calling the Recommendation Function in Code**
```python
from model import recommend, load_data
import torch

# Load Data
df1, df_stocks, category_family_encoder = load_data()

# Load Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRecommender(len(category_family_encoder.classes_)).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Get Recommendations
top_categories, correct_count, actual_categories, total_available, recommended_products = recommend(
    client_id=39370740138294, model=model, df1=df1, df_stocks=df_stocks, category_family_encoder=category_family_encoder, k=3
)

print(f"📌 Recommended Categories: {top_categories}")
print(f"🏷️ Recommended Products: {recommended_products}")
```

---

## 📜 License
This project is open-sourced under the **MIT License**, and contributions are welcome!

## 🤝 Contributing
We welcome **Issues** and **Pull Requests**!
1. Fork this repository
2. Create a new branch (`git checkout -b feature-new-feature`)
3. Commit your changes (`git commit -m "Add new feature"`)
4. Push the branch (`git push origin feature-new-feature`)
5. Submit a Pull Request

---

## 📧 Contact
For any questions or suggestions, feel free to contact [ZongruSun](https://github.com/ZongruSun) 🚀

---

## 📸 **Application Screenshot**
Here is a screenshot of the working **Product Recommendation System**:

![App Screenshot](./path-to-your-screenshot.png)

### **🔹 How It Works**
1. **Enter a Client ID** in the input box.
2. **Click "Get Recommendations"**.
3. **The system will automatically generate recommendations**, displaying:
   - **Recommended Categories** (Top predicted purchase categories)
   - **Recommended Products** (Filtered based on stock availability)
   - **Performance Metrics** (Recall@3 score, correctly predicted categories)

---

