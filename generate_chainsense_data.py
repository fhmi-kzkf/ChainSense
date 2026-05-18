import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Parameters
num_rows = 2500

# Define Pro-Realism Indonesian Cities & Coordinates
city_data = {
    'Jakarta': (-6.2088, 106.8456),
    'Surabaya': (-7.2575, 112.7521),
    'Bandung': (-6.9175, 107.6191),
    'Medan': (3.5952, 98.6722),
    'Makassar': (-5.1476, 119.4327),
    'Jayapura': (-2.5330, 140.7181),
    'Banjarmasin': (-3.3194, 114.5908),
    'Denpasar': (-8.6705, 115.2126),
    'Palembang': (-2.9761, 104.7754),
    'Morotai': (2.3271, 128.4900) # Structural Fragility Node
}

cities = list(city_data.keys())
city_weights = [0.25, 0.2, 0.1, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.02] 

product_categories = ['Industrial Parts', 'Consumer Goods', 'Medical Supplies', 'Automotive']

# Pro-Realism Vendor Names
vendors = [
    "PT Samudera Logistik Nusantara", # The Black Swan Candidate
    "PT Khatulistiwa Trans Express",
    "CV Bahari Persada",
    "PT Jaya Nusantara Logistik",
    "CV Persada Cargo",              # Zombie Node Candidate
    "PT Trans Khatulistiwa",
    "CV Samudera Jaya"
]

shipping_modes = ['Air', 'Truck', 'Sea']

# Generate base dates for 2024
start_date = datetime(2024, 1, 1)
end_date = datetime(2024, 12, 31)
date_list = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]

def calculate_planned_lead_time(city):
    if city in ['Jakarta', 'Bandung', 'Surabaya']:
        return random.randint(1, 3)
    elif city in ['Jayapura', 'Morotai']:
        return random.randint(10, 15)
    else:
        return random.randint(4, 7)

# Generate data
data = []

for i in range(num_rows):
    order_id = f"CS-2024-{str(i+1).zfill(4)}"
    order_date = random.choice(date_list)
    customer_location = np.random.choice(cities, p=city_weights)
    lat, lon = city_data[customer_location]
    
    product_category = random.choice(product_categories)
    quantity = max(1, int(np.random.normal(30, 15)))
    unit_price = random.randint(500000, 50000000)
    
    # Structural Fragility Rule: Morotai only served by CV Bahari Persada
    if customer_location == 'Morotai':
        vendor_name = "CV Bahari Persada"
    else:
        vendor_name = random.choice(vendors)
    
    shipping_mode = random.choice(shipping_modes)
    lead_time_planned = calculate_planned_lead_time(customer_location)
    actual_shipping_days = lead_time_planned
    
    # --- ANOMALY INJECTION ---
    
    # 1. The Black Swan: PT Samudera Logistik Nusantara risk spike after Q3
    black_swan_active = False
    if vendor_name == "PT Samudera Logistik Nusantara" and order_date.month >= 9:
        actual_shipping_days += random.randint(5, 12)
        black_swan_active = True
        
    # 2. Zombie Nodes: CV Persada Cargo (High volume, always late)
    if vendor_name == "CV Persada Cargo":
        quantity = random.randint(80, 150) # High Volume
        actual_shipping_days += random.randint(2, 5) # Consistently Late
        
    # 3. Peak Season Delay
    if order_date.month == 12:
        actual_shipping_days += random.randint(2, 4)
    
    # Delivery Status
    if actual_shipping_days > lead_time_planned:
        delivery_status = "Late"
    elif random.random() < 0.03:
        delivery_status = "Damaged/Returned"
    else:
        delivery_status = "On Time"
        
    # Risk Score Calculation
    risk_score = 0
    if delivery_status == "Late":
        risk_score += 40
    if black_swan_active:
        risk_score += 50 # Massive spike
    if customer_location in ['Jayapura', 'Morotai']:
        risk_score += 15
    if vendor_name == "CV Persada Cargo":
        risk_score += 20
        
    risk_score = min(100, max(0, risk_score + random.randint(-5, 5)))
    
    # Shipping Cost Calculation
    base_costs = {'Air': 500000, 'Truck': 150000, 'Sea': 100000}
    shipping_cost = base_costs[shipping_mode] + (quantity * 5000)
    
    data.append({
        'Order_ID': order_id,
        'Order_Date': order_date.strftime('%Y-%m-%d'),
        'Customer_Location': customer_location,
        'Latitude': lat,
        'Longitude': lon,
        'Product_Category': product_category,
        'Quantity': quantity,
        'Unit_Price': unit_price,
        'Vendor_Name': vendor_name,
        'Shipping_Mode': shipping_mode,
        'Shipping_Cost': shipping_cost,
        'Actual_Shipping_Days': actual_shipping_days,
        'Planned_Lead_Time': lead_time_planned,
        'Delivery_Status': delivery_status,
        'Risk_Score': risk_score
    })

df = pd.DataFrame(data)
df.to_csv('chainsense_synthetic_data.csv', index=False)
print(f"✅ Pro-Realism Dataset 2.0 Generated: {len(df)} rows")