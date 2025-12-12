import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker
fake = Faker('id_ID')  # Indonesian locale

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_rows = 2000

# Define lists and weights
cities = ['Jakarta', 'Surabaya', 'Bandung', 'Medan', 'Makassar', 'Jayapura']
city_weights = [0.3, 0.25, 0.15, 0.1, 0.1, 0.1]  # Weighted toward Jakarta & Surabaya

product_categories = ['Electronics', 'Clothing', 'Home & Garden', 'Automotive']

vendors = [
    "PT Logistik Cepat",
    "CV Maju Jaya", 
    "Global Trans",
    "Mitra Abadi",
    "CV Kargo Lambat"  # Problematic vendor
]

shipping_modes = ['Air', 'Truck', 'Sea']

# Generate base dates for 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
# Convert to list for easier sampling
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

# Function to generate SKU based on category
def generate_sku(category):
    prefixes = {
        'Electronics': 'ELEC',
        'Clothing': 'CLTH',
        'Home & Garden': 'HMGN',
        'Automotive': 'AUTO'
    }
    suffix = ''.join(random.choices('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))
    return f"{prefixes[category]}-{suffix}"

# Function to generate unit price based on category
def generate_unit_price(category):
    if category == 'Electronics':
        return random.randint(5000000, 20000000)  # 5jt-20jt
    elif category == 'Clothing':
        return random.randint(100000, 500000)     # 100rb-500rb
    elif category == 'Home & Garden':
        return random.randint(200000, 3000000)    # 200rb-3jt
    elif category == 'Automotive':
        return random.randint(500000, 8000000)    # 500rb-8jt

# Function to calculate shipping cost
def calculate_shipping_cost(city, weight, mode):
    # Base costs by mode
    base_costs = {'Air': 50000, 'Truck': 20000, 'Sea': 15000}
    base_cost = base_costs[mode]
    
    # Distance multipliers (simplified)
    distance_multiplier = 1.0
    if city == 'Jayapura':
        distance_multiplier = random.uniform(3, 5)  # 3x-5x more expensive
    elif city in ['Medan', 'Makassar']:
        distance_multiplier = 1.5
    
    # Weight factor (assuming weight correlates with quantity)
    return base_cost + (weight * 1000 * distance_multiplier)

# Function to calculate planned lead time
def calculate_planned_lead_time(city):
    if city in ['Jakarta', 'Bandung']:
        return random.randint(1, 3)
    elif city == 'Jayapura':
        return random.randint(10, 14)
    else:  # Other cities (outside Java)
        return random.randint(5, 7)

# Generate data
data = []

for i in range(num_rows):
    # Order ID
    order_id = f"ORD-2024-{str(i+1).zfill(4)}"
    
    # Order Date
    order_date = random.choice(date_list)
    
    # Customer Location
    customer_location = np.random.choice(cities, p=city_weights)
    
    # Product Category
    product_category = random.choice(product_categories)
    
    # SKU
    sku = generate_sku(product_category)
    
    # Quantity (normal distribution, mean=20)
    quantity = max(1, int(np.random.normal(20, 10)))
    quantity = min(quantity, 100)  # Cap at 100
    
    # Unit Price
    unit_price = generate_unit_price(product_category)
    
    # Vendor Name
    vendor_name = random.choice(vendors)
    
    # Shipping Mode
    shipping_mode = random.choice(shipping_modes)
    
    # Weight estimation (for shipping cost calculation)
    weight = quantity * 0.5  # Simplified assumption
    
    # Shipping Cost
    shipping_cost = calculate_shipping_cost(customer_location, weight, shipping_mode)
    
    # Lead Time Planned
    lead_time_planned = calculate_planned_lead_time(customer_location)
    
    # Actual Shipping Days
    actual_shipping_days = lead_time_planned
    
    # Apply delays based on business rules
    # Rule 1: CV Kargo Lambat delay (60% chance of 3-7 extra days)
    if vendor_name == "CV Kargo Lambat" and random.random() < 0.6:
        actual_shipping_days += random.randint(3, 7)
    
    # Rule 2: December peak season delay (2-4 extra days for all vendors)
    if order_date.month == 12:
        actual_shipping_days += random.randint(2, 4)
    
    # Delivery Status
    # 5% chance of being "Damaged/Returned"
    if random.random() < 0.05:
        delivery_status = "Damaged/Returned"
    elif actual_shipping_days > lead_time_planned:
        delivery_status = "Late"
    else:
        delivery_status = "On Time"
    
    # Risk Score (0-100)
    # Based on delivery performance and location risk
    risk_score = 0
    
    # Base score from delivery performance
    if delivery_status == "Late":
        # More delay = higher risk
        delay_ratio = max(0, (actual_shipping_days - lead_time_planned) / lead_time_planned)
        risk_score += min(50, delay_ratio * 30)  # Cap at 50
    
    # Location risk factor
    if customer_location == 'Jayapura':
        risk_score += 20  # Remote location
    elif customer_location in ['Medan', 'Makassar']:
        risk_score += 10  # Regional challenges
    
    # Vendor risk factor
    if vendor_name == "CV Kargo Lambat":
        risk_score += 25
    
    # December peak season risk
    if order_date.month == 12:
        risk_score += 5
    
    # Ensure risk score is between 0-100
    risk_score = min(100, max(0, risk_score))
    
    # Add row to data
    data.append({
        'Order_ID': order_id,
        'Order_Date': order_date.strftime('%Y-%m-%d'),
        'Customer_Location': customer_location,
        'Product_Category': product_category,
        'SKU': sku,
        'Quantity': quantity,
        'Unit_Price': unit_price,
        'Vendor_Name': vendor_name,
        'Shipping_Mode': shipping_mode,
        'Shipping_Cost': round(shipping_cost, 2),
        'Lead_Time_Planned': lead_time_planned,
        'Actual_Shipping_Days': actual_shipping_days,
        'Delivery_Status': delivery_status,
        'Risk_Score': round(risk_score, 2)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('chainsense_synthetic_data.csv', index=False)

print(f"Generated {len(df)} rows of synthetic supply chain data")
print(f"Data saved to chainsense_synthetic_data.csv")
print("\nSample of generated data:")
print(df.head())