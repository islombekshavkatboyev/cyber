# 🏥 Suniy Intellekt Yordamida Diabet Tashxisi

> **Kurs ishi — Amaliy dastur**  
> Mavzu: *Suniy intellektning tibbiyot sohasida qo'llanilishi*

---

## 📁 Loyiha Tuzilmasi

```
kurs_ishi_AI_tibbiyot/
│
├── kurs_ishi.md               # To'liq kurs ishi matni (12 bo'lim)
├── ai_tibbiyot_dasturi.py     # Asosiy Python dasturi
├── README.md                  # Ushbu fayl
└── natijalar_vizualizatsiya.png  # Dastur ishlagandan so'ng yaratiladi
```

---

## 🎯 Dastur Maqsadi

Ushbu dastur **Random Forest** va **Decision Tree** algoritmlaridan foydalanib, bemorning tibbiy ko'rsatkichlari asosida **qandli diabet (diabetes)** xavfini aniqlaydi.

---

## ⚙️ Texnik Talablar

### Kerakli kutubxonalar

```bash
pip install numpy pandas matplotlib scikit-learn
```

| Kutubxona      | Versiya  | Maqsad                          |
|----------------|----------|---------------------------------|
| `numpy`        | ≥ 1.21   | Sonli hisob-kitoblar            |
| `pandas`       | ≥ 1.3    | Ma'lumotlar bilan ishlash       |
| `matplotlib`   | ≥ 3.4    | Grafik va vizualizatsiya        |
| `scikit-learn` | ≥ 0.24   | Mashina o'rganish algoritmlari  |
| `Python`       | ≥ 3.8    | Dasturlash tili                 |

---

## 🚀 Ishga Tushirish

```bash
# 1. Papkaga o'ting
cd kurs_ishi_AI_tibbiyot

# 2. Kutubxonalarni o'rnating
pip install numpy pandas matplotlib scikit-learn

# 3. Dasturni ishga tushiring
python ai_tibbiyot_dasturi.py
```

---

## 📊 Ma'lumotlar (Dataset)

Dastur **Pima Indians Diabetes Dataset** asosida yaratilgan sintetik ma'lumotlar bilan ishlaydi.

### Belgilar (Features)

| # | Belgi                      | Ta'rif                              | Birlik     |
|---|----------------------------|-------------------------------------|------------|
| 1 | `Pregnancies`              | Homiladorlik soni                   | —          |
| 2 | `Glucose`                  | Qon shakari darajasi                | mg/dL      |
| 3 | `BloodPressure`            | Diastol arterial bosim              | mm Hg      |
| 4 | `SkinThickness`            | Triceps teri qalinligi              | mm         |
| 5 | `Insulin`                  | 2 soatlik insulin darajasi          | mu U/ml    |
| 6 | `BMI`                      | Tana massasi indeksi                | kg/m²      |
| 7 | `DiabetesPedigreeFunction` | Oilaviy diabet tarixi ko'rsatkichi  | —          |
| 8 | `Age`                      | Yosh                                | yil        |
| 9 | `Outcome`                  | **Natija** (0=Sog'lom, 1=Diabet)   | —          |

---

## 🤖 Algoritm Haqida

### Random Forest (Tasodifiy O'rmon)

```
                    [Bemor Ma'lumotlari]
                           │
            ┌──────────────┼──────────────┐
            ▼              ▼              ▼
        [Daraxt 1]     [Daraxt 2]     [Daraxt N]
            │              │              │
            ▼              ▼              ▼
        [Sog'lom]      [Diabet]      [Sog'lom]
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                   [Ovoz berish]
                    2 Sog'lom vs 1 Diabet
                           ▼
                   ✅ NATIJA: Sog'lom
```

**Afzalliklari:**
- Yuqori aniqlik (~77–82%)
- Overfitting ga chidamli
- Belgilar ahamiyatini ko'rsatadi
- Ehtimollik foizini beradi

---

## 📈 Kutilayotgan Natijalar

```
Model                          Aniqlik
─────────────────────────────────────
Random Forest                  ~79-82%
Decision Tree                  ~74-77%
─────────────────────────────────────

Cross-Validation (5-fold):     ~78% ± 3%
ROC-AUC:                       ~0.83-0.87
```

---

## 🖥️ Dastur Chiqishi (Namuna)

```
████████████████████████████████████████████████████████████
█                                                          █
█   🏥 SUNIY INTELLEKT YORDAMIDA DIABET TASHXISI          █
█      Kurs ishi — Amaliy Dastur                          █
█      Algoritm: Random Forest & Decision Tree            █
█                                                          █
████████████████████████████████████████████████████████████

✅ Dataset tayyor!

============================================================
  📊 MA'LUMOTLAR TAHLILI
============================================================

📌 Jami bemorlar soni     : 900
📌 Belgilar (xususiyatlar): 8
📌 Sog'lom bemorlar       : 585 (65.0%)
📌 Diabet bor bemorlar    : 315 (35.0%)

============================================================
  🤖 MODEL O'QITISH VA BAHOLASH
============================================================

📦 O'qitish namunalari : 720
📦 Test namunalari     : 180

  Model                          Aniqlik
  ────────────────────────────── ──────────
  Random Forest                      80.56%
  Decision Tree                      75.00%

🏆 Tanlangan model : Random Forest
🎯 Test aniqligi   : 80.56%
📊 Cross-val (5-fold): 79.31% ± 2.14%
📈 ROC-AUC         : 0.8612

>>> Bemor A (Yuqori xavf) uchun tashxis:
  🔴 TASHXIS: DIABET XAVFI BOR  [YUQORI XAVF]
     → Darhol endokrinolog shifokorga murojaat qiling

>>> Bemor B (Past xavf) uchun tashxis:
  🟢 TASHXIS: DIABET XAVFI YO'Q  [PAST XAVF]
     → Yiliga bir marta profilaktik tekshiruv o'ting

>>> Bemor C (O'rta xavf) uchun tashxis:
  🔴 TASHXIS: DIABET XAVFI BOR  [O'RTA XAVF]
```

---

## 🖼️ Vizualizatsiya (9 ta grafik)

Dastur ishlagandan so'ng `natijalar_vizualizatsiya.png` fayli yaratiladi:

| # | Grafik | Ta'rif |
|---|--------|--------|
| 1 | Bemorlar taqsimoti | Donut chart (sog'lom vs diabet) |
| 2 | Glyukoza taqsimoti | Histogram — diabet chegarasi bilan |
| 3 | BMI taqsimoti | Histogram — semizbel chegarasi bilan |
| 4 | Confusion Matrix | To'g'ri/noto'g'ri tashxislar |
| 5 | ROC Egri Chizig'i | Model sifati (AUC) |
| 6 | Belgilar ahamiyati | Qaysi belgi muhimroq |
| 7 | Yoshga ko'ra diabet | Yosh guruhlari bo'yicha tahlil |
| 8 | Glyukoza vs BMI | 2D scatter plot |
| 9 | Model natijalari | Precision, Recall, F1-Score |

---

## 🔬 O'z Bemori Ma'lumotlarini Kiritish

`ai_tibbiyot_dasturi.py` faylining oxiriga quyidagini qo'shing:

```python
# Yangi bemor tashxisi
mening_bemorim = {
    'Pregnancies': 2,
    'Glucose': 130,
    'BloodPressure': 74,
    'SkinThickness': 25,
    'Insulin': 90,
    'BMI': 29.5,
    'DiabetesPedigreeFunction': 0.320,
    'Age': 35
}

# main() ichida model va scaler tayyor bo'lgandan so'ng:
bemor_tashxisi(model, scaler, mening_bemorim)
```

---

## 📚 Kurs Ishi Tuzilmasi

`kurs_ishi.md` faylida quyidagi bo'limlar mavjud:

1. **Kirish** — maqsad va vazifalar
2. **SI haqida umumiy tushuncha** — ML, Deep Learning
3. **Tibbiyotdagi asosiy yo'nalishlar** — sxema
4. **Tashxislashda SI** — diabet, yurak, saraton
5. **Tibbiy tasvirlarni qayta ishlash** — CNN, U-Net
6. **Dori ishlab chiqishda SI** — AlphaFold, COVID-19
7. **Prognozlash va monitoring**
8. **Etik muammolar** — bias, maxfiylik, shaffoflik
9. **Jahon tajribasi va O'zbekiston**
10. **Amaliy qism** — dastur tavsifi
11. **Xulosa** — 6 ta asosiy xulosa
12. **Foydalanilgan adabiyotlar** — 10 ta manba

---

## ⚠️ Muhim Eslatma

> Bu dastur **faqat ta'lim va tadqiqot maqsadida** yaratilgan.  
> Haqiqiy tibbiy tashxis qo'yish uchun **malakali shifokorga** murojaat qiling.  
> Dastur natijasi tibbiy xulosani **ALMASHTIRA OLMAYDI**.

---

## 👨‍💻 Muallif

**Talaba:** [Ismingizni yozing]  
**Fan:** Axborot texnologiyalari / Sun'iy intellekt  
**Yil:** 2026  

---

*README yaratildi: 2026-yil | Python 3.x | scikit-learn*
