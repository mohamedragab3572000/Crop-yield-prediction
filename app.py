from flask import Flask, render_template, request
import os
import pandas as pd
import joblib
from werkzeug.utils import secure_filename
from predict_image import predict_disease

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# تحميل النماذج والـ encoders
print("Current working directory:", os.getcwd())
print("Files in current directory:", os.listdir())

if not os.path.exists("best_model.keras"):
    print("ERROR: Model file not found!")
else:
    print("Model file found successfully.")
model = joblib.load('xgboost_model.pkl')
region_encoder = joblib.load('region_encoder.pkl')
soil_encoder = joblib.load('soil_encoder.pkl')
crop_encoder = joblib.load('crop_encoder.pkl')
weather_encoder = joblib.load('weather_encoder.pkl')

# ========================== الصفحة الرئيسية ==========================
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

# =================== Crop Yield + Best Crop + Profit ====================
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_text = None
    best_crop = None
    expected_profit = None

    if request.method == 'POST':
        try:
            # قراءه المدخلات
            region = region_encoder.transform([request.form['region']])[0]
            soil = soil_encoder.transform([request.form['soil']])[0]
            crop = crop_encoder.transform([request.form['crop']])[0]
            weather = weather_encoder.transform([request.form['weather']])[0]

            rainfall = float(request.form['rainfall'])
            temp = float(request.form['temperature'])
            fertilizer = 1.0 if 'fertilizer' in request.form else 0.0
            irrigation = 1.0 if 'irrigation' in request.form else 0.0
            harvest_days = float(request.form['days_to_harvest'])
            price = float(request.form['price'])

            # الميزات المشتقة
            rainfall_fertilizer_sum = rainfall + fertilizer
            temp_fertilizer_rainfall_interaction = temp * rainfall_fertilizer_sum
            rainfall_temp_fertilizer_ratio = (rainfall + 1) / (temp + 1) / (fertilizer + 1)

            # التنبؤ بالمحصول المطلوب
            input_df = pd.DataFrame([[region, soil, crop, rainfall, temp, fertilizer,
                                      irrigation, weather, harvest_days,
                                      rainfall_fertilizer_sum,
                                      temp_fertilizer_rainfall_interaction,
                                      rainfall_temp_fertilizer_ratio]],
                                    columns=[
                                        'Region', 'Soil_Type', 'Crop', 'Rainfall_mm', 'Temperature_Celsius',
                                        'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition',
                                        'Days_to_Harvest', 'Rainfall_Fertilizer_Sum',
                                        'Temp_Fertilizer_Rainfall_Interaction',
                                        'Rainfall_Temperature_Fertilizer_Ratio'
                                    ])

            prediction = model.predict(input_df)[0]
            prediction_text = f"{round(prediction, 2)} tons/hectare"
            expected_profit = round(prediction * price, 2)

            # التنبؤ بأفضل محصول للظروف
            best_yield = 0
            best_crop_name = None
            for c in crop_encoder.classes_:
                crop_test = crop_encoder.transform([c])[0]
                test_df = input_df.copy()
                test_df['Crop'] = crop_test
                pred = model.predict(test_df)[0]
                if pred > best_yield:
                    best_yield = pred
                    best_crop_name = c

            best_crop = f"{best_crop_name} ({round(best_yield, 2)} tons/hectare)"

        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('predict.html',
                           prediction_text=prediction_text,
                           best_crop=best_crop,
                           expected_profit=f"${expected_profit}" if expected_profit else None)

# =================== تشخيص مرض النبات ==========================
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    disease_result = None
    image_filename = None

    if request.method == 'POST' and 'plant_image' in request.files:
        file = request.files['plant_image']
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_filename = filepath
            disease_result = predict_disease(filepath)

    return render_template('disease.html',
                           disease_result=disease_result,
                           image_filename=image_filename)

# =================== تشغيل التطبيق ==========================
if __name__ == '__main__':
    app.run(debug=True)
