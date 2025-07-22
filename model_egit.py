import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("🤖 Regresyon Modeli Eğitimi Başlıyor...")

# Veriyi yükle
print("📂 Veri yükleniyor...")
df = pd.read_csv('egitim_verisi_regresyon.csv')

# Özellikler ve hedef
X = df.drop('yalan_orani', axis=1).values
y = df['yalan_orani'].values

print(f"📊 Veri Şekli: {X.shape}")
print(f"📊 Hedef Dağılımı:")
print(f"   Min: {y.min():.3f}")
print(f"   Max: {y.max():.3f}")
print(f"   Ortalama: {y.mean():.3f}")
print(f"   Std: {y.std():.3f}")

# Veriyi eğitim/test olarak böl
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=None
)

print(f"\n📈 Eğitim Seti: {X_train.shape[0]} örnek")
print(f"📈 Test Seti: {X_test.shape[0]} örnek")

# Veriyi normalize et
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regresyon modeli oluştur
print("\n🏗️ Model oluşturuluyor...")
model = keras.Sequential([
    layers.Input(shape=(7,), name='input_layer'),
    layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense_1'),
    layers.Dropout(0.3, name='dropout_1'),
    layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='dense_2'),
    layers.Dropout(0.2, name='dropout_2'),
    layers.Dense(16, activation='relu', name='dense_3'),
    layers.Dense(1, activation='sigmoid', name='output_layer')  # 0-1 arası çıktı
])

# Model derleme
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',  # Mean Squared Error
    metrics=['mae']  # Mean Absolute Error
)

print("📋 Model Özeti:")
model.summary()

# Early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Model eğitimi
print("\n🚀 Model eğitimi başlıyor...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test_scaled, y_test),
    callbacks=[early_stopping],
    verbose=1
)

# Model değerlendirme
print("\n📊 Model Değerlendirmesi:")
y_pred = model.predict(X_test_scaled).flatten()

# Metrikler
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"   MSE: {mse:.4f}")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE: {mae:.4f}")
print(f"   R²: {r2:.4f}")

# Tahmin analizi
print(f"\n🎯 Tahmin Analizi:")
print(f"   Gerçek Ortalama: {y_test.mean():.3f}")
print(f"   Tahmin Ortalama: {y_pred.mean():.3f}")
print(f"   Gerçek Std: {y_test.std():.3f}")
print(f"   Tahmin Std: {y_pred.std():.3f}")

# Sınıflandırma performansı (0.5 eşik değeri ile)
y_pred_class = (y_pred > 0.5).astype(int)
y_test_class = (y_test > 0.5).astype(int)

from sklearn.metrics import accuracy_score, classification_report
accuracy = accuracy_score(y_test_class, y_pred_class)
print(f"\n🎯 Sınıflandırma Performansı (0.5 eşik):")
print(f"   Doğruluk: {accuracy:.4f}")
print(f"\n📋 Detaylı Rapor:")
print(classification_report(y_test_class, y_pred_class, target_names=['Doğru', 'Yalan']))

# Model kaydetme
try:
    # Modeli farklı formatlarda kaydet
    model.save('mikro_ifade_model_regresyon.keras', save_format='keras', include_optimizer=False)
    print(f"\n💾 Model kaydedildi: mikro_ifade_model_regresyon.keras")
    
    # Alternatif kaydetme yöntemi - JSON formatında
    model_json = model.to_json()
    with open('mikro_ifade_model_regresyon.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('mikro_ifade_model_weights.h5')
    print(f"💾 Model JSON ve ağırlıkları kaydedildi")
    
except Exception as e:
    print(f"Model kaydetme hatası: {e}")
    # Basit kaydetme yöntemi
    try:
        model.save('mikro_ifade_model_regresyon.h5', include_optimizer=False)
        print(f"💾 Model alternatif formatta kaydedildi: mikro_ifade_model_regresyon.h5")
    except Exception as e2:
        print(f"Alternatif kaydetme de başarısız: {e2}")

# Eğitim grafiği
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Eğitim Loss')
plt.plot(history.history['val_loss'], label='Validasyon Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Eğitim MAE')
plt.plot(history.history['val_mae'], label='Validasyon MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.savefig('model_egitim_grafigi.png', dpi=300, bbox_inches='tight')
print(f"📈 Eğitim grafiği kaydedildi: model_egitim_grafigi.png")

# Tahmin vs Gerçek grafiği
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([0, 1], [0, 1], 'r--', lw=2)
plt.xlabel('Gerçek Yalan Oranı')
plt.ylabel('Tahmin Edilen Yalan Oranı')
plt.title('Tahmin vs Gerçek')
plt.grid(True, alpha=0.3)
plt.savefig('tahmin_vs_gercek.png', dpi=300, bbox_inches='tight')
print(f"📈 Tahmin grafiği kaydedildi: tahmin_vs_gercek.png")

print(f"\n✅ Model eğitimi tamamlandı!")
print(f"🎯 R² Skoru: {r2:.4f}")
print(f"🎯 RMSE: {rmse:.4f}")
print(f"🎯 Sınıflandırma Doğruluğu: {accuracy:.4f}") 