import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta

fake = Faker('id_ID')

def generate_pharma_data(num_rows=2000):
    print("Generating Pharmaceutical Cold Chain Data...")
    
    # 1. Setup Master Data
    locations = {
        'Jakarta': {'base_temp': 30, 'dist_factor': 1.0},
        'Bandung': {'base_temp': 24, 'dist_factor': 1.2},
        'Surabaya': {'base_temp': 32, 'dist_factor': 1.5},
        'Medan': {'base_temp': 28, 'dist_factor': 2.0},
        'Makassar': {'base_temp': 31, 'dist_factor': 2.5},
        'Jayapura': {'base_temp': 29, 'dist_factor': 4.0}
    }
    
    products = [
        {'name': 'Vaksin COVID-19', 'type': 'Cold Chain', 'ideal_temp': '2-8°C', 'price': 150000},
        {'name': 'Insulin', 'type': 'Cold Chain', 'ideal_temp': '2-8°C', 'price': 250000},
        {'name': 'Antibiotik', 'type': 'Cool', 'ideal_temp': '15-25°C', 'price': 50000},
        {'name': 'Vitamin C', 'type': 'Room', 'ideal_temp': 'Any', 'price': 25000}
    ]
    
    vendors = [
        'BioLogistics Pro',      # Vendor Premium (Mahal, Bagus)
        'ColdChain Indo',        # Spesialis Pendingin (Bagus)
        'Kimia Farma Trading',   # Standar BUMN
        'Cepat Kirim Express',   # Vendor Umum (Bukan spesialis dingin -> BAHAYA)
        'CV Truk Biasa'          # Vendor Murah (Sangat Bahaya buat Vaksin)
    ]

    data = []

    for i in range(num_rows):
        # Generate Order Info
        order_id = f"MED-{2024}-{str(i+1).zfill(5)}"
        date_order = fake.date_between(start_date='-1y', end_date='today')
        
        # Pick Random Attributes
        loc_name = random.choice(list(locations.keys()))
        prod = random.choice(products)
        vendor = random.choice(vendors)
        qty = random.randint(10, 500)
        
        # --- LOGIKA SKENARIO (INJEKSI MASALAH) ---
        
        # 1. Tentukan Suhu Pengiriman (Actual Temp)
        # Skenario: Vendor "CV Truk Biasa" & "Cepat Kirim" sering gagal jaga suhu
        if vendor in ['CV Truk Biasa', 'Cepat Kirim Express']:
            if prod['type'] == 'Cold Chain':
                # Bahaya! Vaksin dikirim pakai truk biasa -> Suhu naik
                actual_temp = random.uniform(10, 30) 
            else:
                actual_temp = random.uniform(20, 30)
        else:
            # Vendor bagus menjaga suhu sesuai standar
            if prod['type'] == 'Cold Chain':
                actual_temp = random.uniform(2, 8) # Aman
            else:
                actual_temp = random.uniform(18, 25)

        # 2. Status Kualitas (Spoilage)
        # Jika barang Cold Chain tapi suhunya > 8 derajat, maka RUSAK
        status = "On Time" # Default time status
        quality_status = "Good"
        
        if prod['type'] == 'Cold Chain' and actual_temp > 8.5:
            delivery_status = "Spoiled / Damaged" # Rusak karena suhu
            risk_score = random.randint(80, 100) # Risiko Mentok
        elif vendor == 'CV Truk Biasa' and random.random() < 0.3:
             delivery_status = "Late"
             risk_score = random.randint(50, 75)
        else:
            delivery_status = "On Time"
            risk_score = random.randint(5, 30)

        # 3. Biaya (Cost)
        # Pengiriman Jayapura mahal, Cold Chain mahal
        base_cost = 50000
        if prod['type'] == 'Cold Chain': base_cost *= 2
        dist_cost = 10000 * locations[loc_name]['dist_factor']
        shipping_cost = int(base_cost + dist_cost + (qty * 100))

        data.append({
            'Order_ID': order_id,
            'Order_Date': date_order,
            'Customer_Location': loc_name,
            'Product_Name': prod['name'],
            'Product_Type': prod['type'],
            'Ideal_Temperature': prod['ideal_temp'],
            'Actual_Temperature': round(actual_temp, 1),
            'Quantity': qty,
            'Vendor_Name': vendor,
            'Shipping_Cost': shipping_cost,
            'Delivery_Status': delivery_status,
            'Risk_Score': risk_score
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save
    filename = 'chainsense_pharma_data.csv'
    df.to_csv(filename, index=False)
    print(f"✅ Success! File '{filename}' created with {num_rows} rows.")
    print("Scenario Injected: 'CV Truk Biasa' destroying Vaccines due to high temp.")

if __name__ == "__main__":
    generate_pharma_data()