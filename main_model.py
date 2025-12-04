import main_model

from growwapi import GrowwAPI
 
# Groww API Credentials (Replace with your actual credentials)
API_AUTH_TOKEN = "your_token"
 
# Initialize Groww API
groww = GrowwAPI(API_AUTH_TOKEN)
 
get_instrument_by_groww_symbol_response = groww.get_instrument_by_groww_symbol(
    groww_symbol="NSE-RELIANCE"
)
print(get_instrument_by_groww_symbol_response)

from growwapi import GrowwAPI
 
# Groww API Credentials (Replace with your actual credentials)
API_AUTH_TOKEN = "your_token"
 
# Initialize Groww API
groww = GrowwAPI(API_AUTH_TOKEN)
 
place_order_response = groww.place_order(
    trading_symbol="WIPRO",
    transaction_type=groww.TRANSACTION_TYPE_BUY,
    price=250,               # Optional: Price of the stock (for Limit orders)
    trigger_price=245,       # Optional: Trigger price (if applicable)
    order_reference_id="Ab-654321234-1628190"  # Optional: User provided 8 to 20 length alphanumeric reference ID to track the order
)
print(place_order_response)