import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Step 1: Prepare DataFrames from CSV Files
def create_dataframe_dict(directory: str) -> dict:
    """
    Reads all CSV files in the given directory and creates a dictionary of DataFrames.
    """
    dataframe_dict = {}
    for filename in os.listdir(directory):
        if filename.endswith(".csv"):
            filepath = os.path.join(directory, filename)
            symbol = os.path.splitext(filename)[0]
            dataframe_dict[symbol] = pd.read_csv(filepath)
    return dataframe_dict

# Step 2: Create Input Sequences for LSTM
def create_sequences(data, window_size, feature_columns, target_column):
    """
    Create input sequences and target labels for LSTM training.
    """
    X, Y = [], []
    for i in range(len(data) - window_size):
        X.append(data[feature_columns].iloc[i:i + window_size].values)
        Y.append(data[target_column].iloc[i + window_size])
    return np.array(X), np.array(Y)

# Step 3: Build and Train the LSTM Model
def build_and_train_model(X_train, Y_train, input_shape):
    """
    Build and train the LSTM model.
    """
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: Hold (0), Buy (1), Sell (2)
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, Y_train, validation_split=0.1, epochs=50, batch_size=32)
    return model

# Step 4: Mock Broker Class
class MockBroker:
    def __init__(self, initial_balance: float, price_dict: dict, model, feature_columns, window_size):
        """
        Initialize MockBroker with starting balance, price dictionary, and LSTM model.
        """
        self.balance = initial_balance
        self.price_dict = price_dict
        self.model = model
        self.feature_columns = feature_columns
        self.window_size = window_size
        self.holdings = {}  # Symbol -> quantity
        self.current_day = 0
    
    def get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol based on current_day."""
        return self.price_dict[symbol]['close'].iloc[self.current_day]
    
    def buy(self, symbol: str, quantity: int) -> bool:
        """Buy specified quantity of a symbol if sufficient balance exists."""
        cost = self.get_current_price(symbol) * quantity
        if cost > self.balance:
            return False
        self.balance -= cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
        return True
    
    def sell(self, symbol: str, quantity: int) -> bool:
        """Sell specified quantity of a symbol if sufficient holdings exist."""
        if symbol not in self.holdings or self.holdings[symbol] < quantity:
            return False
        proceeds = self.get_current_price(symbol) * quantity
        self.balance += proceeds
        self.holdings[symbol] -= quantity
        if self.holdings[symbol] == 0:
            del self.holdings[symbol]
        return True
    
    def get_net_worth(self) -> float:
        """Calculate total net worth (cash + value of holdings)."""
        holdings_value = sum(
            self.get_current_price(symbol) * quantity 
            for symbol, quantity in self.holdings.items()
        )
        return self.balance + holdings_value
    
    def next_day(self):
        """Advance to the next trading day."""
        self.current_day += 1
    
    def decide_action(self, symbol):
        """
        Use the LSTM model to decide the action (Buy, Sell, Hold).
        """
        if self.current_day < self.window_size:
            return 0  # Not enough data for prediction
        df = self.price_dict[symbol]
        recent_data = df[self.feature_columns].iloc[self.current_day - self.window_size:self.current_day]
        recent_data = np.expand_dims(recent_data.values, axis=0)  # Reshape for LSTM
        prediction = self.model.predict(recent_data)
        return np.argmax(prediction) - 1  # Convert to -1, 0, 1 (Sell, Hold, Buy)
    
    def simulate_trading(self):
        """
        Simulate trading for all symbols over the entire dataset.
        """
        while self.current_day < len(next(iter(self.price_dict.values()))):  # Use the length of any DataFrame
            for symbol in self.price_dict.keys():
                action = self.decide_action(symbol)
                if action == 1:  # Buy
                    self.buy(symbol, 10)  # Example quantity
                elif action == -1:  # Sell
                    self.sell(symbol, 10)
            self.next_day()
        return self.get_net_worth()

# Step 5: Main Code
directory = "./"  # Directory containing your CSV files
dataframe_dict = create_dataframe_dict(directory)

# Training the LSTM model
window_size = 14
feature_columns = ['MACD', 'RSI', 'close']
target_column = 'positions'
X_all, Y_all = [], []

for symbol, df in dataframe_dict.items():
    X, Y = create_sequences(df, window_size, feature_columns, target_column)
    X_all.append(X)
    Y_all.append(Y)

X_all = np.concatenate(X_all, axis=0)
Y_all = np.concatenate(Y_all, axis=0)

X_train, X_test, Y_train, Y_test = train_test_split(X_all, Y_all, test_size=0.1, random_state=42)
model = build_and_train_model(X_train, Y_train, input_shape=(X_train.shape[1], X_train.shape[2]))

# Simulate trading
mock_broker = MockBroker(10000.0, dataframe_dict, model, feature_columns, window_size)
final_net_worth = mock_broker.simulate_trading()

print(f"Final Net Worth: ${final_net_worth:.2f}")
