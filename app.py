from flask import Flask, render_template, request, redirect, url_for, flash
import pickle
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key_here_change_in_production'

# =================================================
# Load trained model
# =================================================
try:
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    model = model_data['model']
    label_encoder = model_data['label_encoder']
    locations = model_data['locations']

    print("✓ Model loaded successfully")
    print("✓ Available locations:", locations)

except Exception as e:
    print("❌ Error loading model:", e)
    model = None
    label_encoder = None
    locations = []

# =================================================
# CSV file for contact enquiries
# =================================================
ENQUIRIES_FILE = 'enquiries.csv'

# =================================================
# Home Page
# =================================================
@app.route('/')
def index():
    return render_template('index.html', locations=locations)

# =================================================
# Prediction Route
# =================================================
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        flash('Model not loaded. Please run model.py first!', 'danger')
        return redirect(url_for('index'))

    try:
        # Read form data
        area = float(request.form.get('area'))
        bedrooms = int(request.form.get('bedrooms'))
        bathrooms = int(request.form.get('bathrooms'))
        location = request.form.get('location')
        age = int(request.form.get('age'))

        # Encode location
        location_encoded = label_encoder.transform([location])[0]

        # Prepare input for model
        input_data = [[area, bedrooms, bathrooms, location_encoded, age]]

        # Predict price
        base_price = model.predict(input_data)[0]

        # Extra charges
        gst = base_price * 0.05
        stamp_duty = base_price * 0.06
        registration = base_price * 0.01
        total_price = base_price + gst + stamp_duty + registration

        # ✅ RESULT STRUCTURE (FIXED)
        result = {
            'input': {
                'area': area,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'location': location,
                'age': age
            },
            'base_price': f"₹{base_price:,.2f}",
            'gst': f"₹{gst:,.2f}",
            'stamp_duty': f"₹{stamp_duty:,.2f}",
            'registration': f"₹{registration:,.2f}",
            'total_price': f"₹{total_price:,.2f}"
        }

        return render_template(
            'index.html',
            locations=locations,
            result=result
        )

    except Exception as e:
        flash(f'Error in prediction: {e}', 'danger')
        return redirect(url_for('index'))

# =================================================
# About Page
# =================================================
@app.route('/about')
def about():
    return render_template('about.html')

# =================================================
# Contact Page
# =================================================
@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        try:
            name = request.form.get('name')
            email = request.form.get('email')
            phone = request.form.get('phone')
            message = request.form.get('message')

            df = pd.DataFrame([{
                'Name': name,
                'Email': email,
                'Phone': phone,
                'Message': message
            }])

            if os.path.exists(ENQUIRIES_FILE):
                df.to_csv(ENQUIRIES_FILE, mode='a', header=False, index=False)
            else:
                df.to_csv(ENQUIRIES_FILE, index=False)

            flash('Thank you! We will contact you soon.', 'success')
            return redirect(url_for('contact'))

        except Exception as e:
            flash(f'Error submitting enquiry: {e}', 'danger')
            return redirect(url_for('contact'))

    return render_template('contact.html')

# =================================================
# Run App
# =================================================
if __name__ == '__main__':
    print("=" * 60)
    print("🏠 House Price Prediction App Running")
    print("➜ http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True)
