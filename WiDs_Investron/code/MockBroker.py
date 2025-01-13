class MockBroker:
    def __init__(self, initial_balance: float, price_dict: dict):
        """
        Initialize MockBroker with starting balance and price dictionary
        
        Args:
            initial_balance: Starting cash balance
            price_dict: Dictionary with stock symbols as keys and list of daily prices as values
        """
        self.balance = initial_balance
        self.price_dict = price_dict
        self.holdings = {}  # Symbol -> quantity
        self.current_day = 0
    
    def get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol based on current_day"""
        return self.price_dict[symbol][self.current_day]
    
    def buy(self, symbol: str, quantity: int) -> bool:
        """
        Buy specified quantity of a symbol if sufficient balance exists
        
        Returns:
            bool: True if purchase successful, False otherwise
        """
        if symbol not in self.price_dict:
            return False
            
        cost = self.get_current_price(symbol) * quantity
        if cost > self.balance:
            return False
            
        self.balance -= cost
        self.holdings[symbol] = self.holdings.get(symbol, 0) + quantity
        return True
    
    def sell(self, symbol: str, quantity: int) -> bool:
        """
        Sell specified quantity of a symbol if sufficient holdings exist
        
        Returns:
            bool: True if sale successful, False otherwise
        """
        if symbol not in self.holdings or self.holdings[symbol] < quantity:
            return False
            
        proceeds = self.get_current_price(symbol) * quantity
        self.balance += proceeds
        self.holdings[symbol] -= quantity
        
        if self.holdings[symbol] == 0:
            del self.holdings[symbol]
        return True
    
    def get_net_worth(self) -> float:
        """Calculate total net worth (cash + value of holdings)"""
        holdings_value = sum(
            self.get_current_price(symbol) * quantity 
            for symbol, quantity in self.holdings.items()
        )
        return self.balance + holdings_value
    
    def __str__(self) -> str:
        """Return string representation showing balance, holdings and net worth"""
        result = f"Balance: ${self.balance:.2f}\n"
        result += "Holdings:\n"
        for symbol, quantity in self.holdings.items():
            price = self.get_current_price(symbol)
            value = price * quantity
            result += f"  {symbol}: {quantity} shares @ ${price:.2f} = ${value:.2f}\n"
        result += f"Net Worth: ${self.get_net_worth():.2f}"
        return result

    def next_day(self):
        """Advance to the next trading day"""
        self.current_day += 1



prices = {
    'AAPL': [150.0, 152.0, 151.0, 153.0],
    'GOOGL': [2500.0, 2520.0, 2480.0, 2510.0]
}

# Create broker instance
broker = MockBroker(10000.0, prices)

# Make trades
broker.buy('AAPL', 10)
print(broker)  # Shows current state

broker.next_day()  # Move to next day
broker.sell('AAPL', 5)
print(broker)