import os
import numpy as np
import json
from deepface import DeepFace
import cv2
from collections import Counter

def referans_dagilimlari_hesapla():
    """Dataset'teki her soru için doğru ve yalan referans dağılımlarını hesaplar"""
    
    dataset_path = "dataset/1"
    referans_dagilimlari = {}
    
    print("Referans dağılımları hesaplanıyor...")
    
    # Her soru için
    for soru_no in range(1, 16):
        dogru_duygular = []
        yalan_duygular = []
        
        # Doğru klasörü
        dogru_path = os.path.join(dataset_path, f"soru{soru_no}_dogru")
        if os.path.exists(dogru_path):
            print(f"Soru {soru_no} doğru klasörü analiz ediliyor: {dogru_path}")
            for img_file in os.listdir(dogru_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(dogru_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            analiz = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                            if analiz and len(analiz) > 0:
                                dominant_emotion = analiz[0]['dominant_emotion']
                                dogru_duygular.append(dominant_emotion)
                    except Exception as e:
                        print(f"Hata (Soru {soru_no} - Doğru): {e}")
        
        # Yalan klasörü
        yalan_path = os.path.join(dataset_path, f"soru{soru_no}_yalan")
        if os.path.exists(yalan_path):
            print(f"Soru {soru_no} yalan klasörü analiz ediliyor: {yalan_path}")
            for img_file in os.listdir(yalan_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(yalan_path, img_file)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            analiz = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
                            if analiz and len(analiz) > 0:
                                dominant_emotion = analiz[0]['dominant_emotion']
                                yalan_duygular.append(dominant_emotion)
                    except Exception as e:
                        print(f"Hata (Soru {soru_no} - Yalan): {e}")
        
        # Duygu dağılımlarını hesapla
        duygu_etiketleri = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Doğru dağılımı
        dogru_counter = Counter(dogru_duygular)
        dogru_dagilim = []
        for duygu in duygu_etiketleri:
            oran = dogru_counter.get(duygu, 0) / len(dogru_duygular) if len(dogru_duygular) > 0 else 0
            dogru_dagilim.append(oran)
        
        # Yalan dağılımı
        yalan_counter = Counter(yalan_duygular)
        yalan_dagilim = []
        for duygu in duygu_etiketleri:
            oran = yalan_counter.get(duygu, 0) / len(yalan_duygular) if len(yalan_duygular) > 0 else 0
            yalan_dagilim.append(oran)
        
        # Sonuçları sakla
        referans_dagilimlari[str(soru_no)] = {
            'dogru_dagilim': dogru_dagilim,
            'yalan_dagilim': yalan_dagilim,
            'dogru_sayi': len(dogru_duygular),
            'yalan_sayi': len(yalan_duygular),
            'duygu_etiketleri': duygu_etiketleri
        }
        
        print(f"Soru {soru_no}: Doğru {len(dogru_duygular)} görsel, Yalan {len(yalan_duygular)} görsel")
    
    # Sonuçları JSON dosyasına kaydet
    with open('referans_dagilimlari.json', 'w', encoding='utf-8') as f:
        json.dump(referans_dagilimlari, f, indent=2, ensure_ascii=False)
    
    print(f"\nReferans dağılımları 'referans_dagilimlari.json' dosyasına kaydedildi.")
    
    return referans_dagilimlari

def dagilim_benzerligi_hesapla(test_dagilim, referans_dagilim):
    """İki dağılım arasındaki benzerliği hesaplar (cosine similarity)"""
    test_array = np.array(test_dagilim)
    ref_array = np.array(referans_dagilim)
    
    # Cosine similarity
    dot_product = np.dot(test_array, ref_array)
    norm_test = np.linalg.norm(test_array)
    norm_ref = np.linalg.norm(ref_array)
    
    if norm_test == 0 or norm_ref == 0:
        return 0
    
    similarity = dot_product / (norm_test * norm_ref)
    return similarity

if __name__ == "__main__":
    referans_dagilimlari_hesapla() 