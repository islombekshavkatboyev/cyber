"""
========================================================
  SUNIY INTELLEKT YORDAMIDA DIABET TASHXISI
  Kurs ishi - Amaliy dastur
  
  Mavzu: Suniy intellektning tibbiyot sohasida qo'llanilishi
  Algoritm: Random Forest Classifier
  Ma'lumotlar: Pima Indians Diabetes Dataset
========================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc
)
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. MA'LUMOTLARNI YARATISH (Sinttetik dataset - real Pima dataset asosida)
# ============================================================

def dataset_yaratish(n_samples=768, random_state=42):
    """
    Pima Indians Diabetes Dataset ga o'xshash sintetik dataset yaratish.
    Haqiqiy dataset: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
    """
    np.random.seed(random_state)
    n = n_samples

    # Diabetsiz bemorlar (0) - 500 ta
    n0 = int(n * 0.65)
    # Diabet bor bemorlar (1) - 268 ta
    n1 = n - n0

    def qiymat_yaratish(mean0, std0, mean1, std1, n0, n1, low=0, high=None):
        v0 = np.random.normal(mean0, std0, n0).clip(low, high)
        v1 = np.random.normal(mean1, std1, n1).clip(low, high)
        return np.concatenate([v0, v1])

    pregnancies  = qiymat_yaratish(3.3, 3.0, 4.9, 3.5, n0, n1, 0, 17)
    glucose      = qiymat_yaratish(109, 26,  141, 31,  n0, n1, 44, 199)
    bloodpressure= qiymat_yaratish(68,  18,  70,  21,  n0, n1, 24, 122)
    skinthickness= qiymat_yaratish(19,  15,  22,  17,  n0, n1, 7,  99)
    insulin      = qiymat_yaratish(68,  98,  100, 138, n0, n1, 14, 846)
    bmi          = qiymat_yaratish(30,  7.5, 35,  7.0, n0, n1, 18, 67)
    dpf          = qiymat_yaratish(0.43,0.30,0.55,0.37,n0, n1, 0.08, 2.42)
    age          = qiymat_yaratish(31,  11,  37,  10,  n0, n1, 21, 81)

    labels = np.array([0]*n0 + [1]*n1)

    df = pd.DataFrame({
        'Pregnancies'             : pregnancies.astype(int),
        'Glucose'                 : glucose.astype(int),
        'BloodPressure'           : bloodpressure.astype(int),
        'SkinThickness'           : skinthickness.astype(int),
        'Insulin'                 : insulin.astype(int),
        'BMI'                     : bmi.round(1),
        'DiabetesPedigreeFunction': dpf.round(3),
        'Age'                     : age.astype(int),
        'Outcome'                 : labels
    })

    # Tasodifiy tartibda aralashtirish
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ============================================================
# 2. MA'LUMOTLARNI TAHLIL QILISH
# ============================================================

def malumot_tahlili(df):
    print("\n" + "="*60)
    print("  📊 MA'LUMOTLAR TAHLILI")
    print("="*60)

    print(f"\n📌 Jami bemorlar soni     : {len(df)}")
    print(f"📌 Belgilar (xususiyatlar): {df.shape[1]-1}")
    print(f"📌 Sog'lom bemorlar       : {(df['Outcome']==0).sum()} ({(df['Outcome']==0).mean()*100:.1f}%)")
    print(f"📌 Diabet bor bemorlar    : {(df['Outcome']==1).sum()} ({(df['Outcome']==1).mean()*100:.1f}%)")

    print("\n📋 Statistik ko'rsatkichlar:")
    print("-"*60)
    stats = df.describe().T[['mean','std','min','max']]
    stats.columns = ['O\'rtacha', 'Standart og\'ish', 'Min', 'Max']
    print(stats.to_string())


# ============================================================
# 3. VIZUALIZATSIYA
# ============================================================

def vizualizatsiya(df, y_test, y_pred, y_prob, feature_names, importances, fpr, tpr, roc_auc):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0d1117')
    plt.suptitle(
        "SUNIY INTELLEKT YORDAMIDA DIABET TASHXISI\nTahlil Natijalari",
        fontsize=16, fontweight='bold', color='white', y=0.98
    )

    ax_color = '#161b22'
    title_color = '#58a6ff'
    text_color = '#e6edf3'
    grid_color = '#30363d'

    def style_ax(ax, title):
        ax.set_facecolor(ax_color)
        ax.tick_params(colors=text_color, labelsize=9)
        ax.set_title(title, color=title_color, fontsize=11, fontweight='bold', pad=10)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.xaxis.label.set_color(text_color)
        ax.yaxis.label.set_color(text_color)

    # --- 1. Diabet taqsimoti (Donut chart) ---
    ax1 = fig.add_subplot(3, 3, 1)
    ax1.set_facecolor(ax_color)
    counts = df['Outcome'].value_counts()
    colors = ['#3fb950', '#f85149']
    wedges, texts, autotexts = ax1.pie(
        counts, labels=['Sog\'lom', 'Diabet'],
        colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(width=0.5, edgecolor='#0d1117', linewidth=2)
    )
    for t in texts + autotexts:
        t.set_color(text_color)
        t.set_fontsize(10)
    ax1.set_title("Bemorlar Taqsimoti", color=title_color, fontsize=11, fontweight='bold')

    # --- 2. Glyukoza taqsimoti ---
    ax2 = fig.add_subplot(3, 3, 2)
    style_ax(ax2, "Glyukoza Darajasi (Diabet vs Sog'lom)")
    ax2.hist(df[df['Outcome']==0]['Glucose'], bins=25, alpha=0.7,
             color='#3fb950', label="Sog'lom", edgecolor='#0d1117')
    ax2.hist(df[df['Outcome']==1]['Glucose'], bins=25, alpha=0.7,
             color='#f85149', label='Diabet', edgecolor='#0d1117')
    ax2.axvline(126, color='#d29922', linestyle='--', linewidth=1.5, label='Chegara (126)')
    ax2.set_xlabel('Glyukoza (mg/dL)')
    ax2.set_ylabel('Bemorlar soni')
    ax2.legend(facecolor=ax_color, labelcolor=text_color, edgecolor=grid_color, fontsize=8)
    ax2.grid(axis='y', color=grid_color, alpha=0.5)

    # --- 3. BMI taqsimoti ---
    ax3 = fig.add_subplot(3, 3, 3)
    style_ax(ax3, "Tana Massasi Indeksi (BMI)")
    ax3.hist(df[df['Outcome']==0]['BMI'], bins=25, alpha=0.7,
             color='#3fb950', label="Sog'lom", edgecolor='#0d1117')
    ax3.hist(df[df['Outcome']==1]['BMI'], bins=25, alpha=0.7,
             color='#f85149', label='Diabet', edgecolor='#0d1117')
    ax3.axvline(30, color='#d29922', linestyle='--', linewidth=1.5, label='Semizbel chegarasi')
    ax3.set_xlabel('BMI (kg/m²)')
    ax3.set_ylabel('Bemorlar soni')
    ax3.legend(facecolor=ax_color, labelcolor=text_color, edgecolor=grid_color, fontsize=8)
    ax3.grid(axis='y', color=grid_color, alpha=0.5)

    # --- 4. Confusion Matrix ---
    ax4 = fig.add_subplot(3, 3, 4)
    style_ax(ax4, "Chalkashlik Matritsasi (Confusion Matrix)")
    cm = confusion_matrix(y_test, y_pred)
    im = ax4.imshow(cm, cmap='Blues', aspect='auto')
    for i in range(2):
        for j in range(2):
            ax4.text(j, i, str(cm[i,j]), ha='center', va='center',
                     fontsize=18, fontweight='bold',
                     color='white' if cm[i,j] > cm.max()/2 else '#0d1117')
    ax4.set_xticks([0,1])
    ax4.set_yticks([0,1])
    ax4.set_xticklabels(["Sog'lom (Pred)", "Diabet (Pred)"], color=text_color, fontsize=8)
    ax4.set_yticklabels(["Sog'lom (Haq.)", "Diabet (Haq.)"], color=text_color, fontsize=8)
    ax4.set_xlabel("Bashorat qilingan")
    ax4.set_ylabel("Haqiqiy")

    # --- 5. ROC Egri Chizig'i ---
    ax5 = fig.add_subplot(3, 3, 5)
    style_ax(ax5, f"ROC Egri Chizig'i (AUC = {roc_auc:.3f})")
    ax5.plot(fpr, tpr, color='#58a6ff', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax5.plot([0,1],[0,1], color='#484f58', linestyle='--', lw=1.5, label='Tasodifiy')
    ax5.fill_between(fpr, tpr, alpha=0.15, color='#58a6ff')
    ax5.set_xlabel("False Positive Rate")
    ax5.set_ylabel("True Positive Rate")
    ax5.legend(facecolor=ax_color, labelcolor=text_color, edgecolor=grid_color, fontsize=9)
    ax5.grid(color=grid_color, alpha=0.4)

    # --- 6. Belgilarning Ahamiyati ---
    ax6 = fig.add_subplot(3, 3, 6)
    style_ax(ax6, "Belgilarning Ahamiyati (Feature Importance)")
    sorted_idx = np.argsort(importances)
    colors_bar = plt.cm.RdYlGn(importances[sorted_idx] / importances.max())
    bars = ax6.barh(np.array(feature_names)[sorted_idx], importances[sorted_idx],
                    color=colors_bar, edgecolor='#0d1117', linewidth=0.5)
    for bar, val in zip(bars, importances[sorted_idx]):
        ax6.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                 f'{val:.3f}', va='center', ha='left', color=text_color, fontsize=8)
    ax6.set_xlabel("Ahamiyat darajasi")
    ax6.grid(axis='x', color=grid_color, alpha=0.4)

    # --- 7. Yoshga qarab diabet ---
    ax7 = fig.add_subplot(3, 3, 7)
    style_ax(ax7, "Yoshga Ko'ra Diabet Ko'rsatkichi")
    age_bins = [20, 30, 40, 50, 60, 70, 81]
    labels_age = ['20-29', '30-39', '40-49', '50-59', '60-69', '70+']
    df['AgeGroup'] = pd.cut(df['Age'], bins=age_bins, labels=labels_age)
    diabet_rate = df.groupby('AgeGroup', observed=True)['Outcome'].mean() * 100
    bars2 = ax7.bar(diabet_rate.index, diabet_rate.values,
                    color=['#3fb950' if v < 40 else '#d29922' if v < 55 else '#f85149'
                           for v in diabet_rate.values],
                    edgecolor='#0d1117', linewidth=0.5)
    for bar, val in zip(bars2, diabet_rate.values):
        ax7.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                 f'{val:.1f}%', ha='center', va='bottom', color=text_color, fontsize=9)
    ax7.set_xlabel("Yosh guruhi")
    ax7.set_ylabel("Diabet ko'rsatkichi (%)")
    ax7.grid(axis='y', color=grid_color, alpha=0.4)

    # --- 8. Glyukoza vs BMI scatter ---
    ax8 = fig.add_subplot(3, 3, 8)
    style_ax(ax8, "Glyukoza vs BMI (Tasniflash)")
    scatter_colors = ['#3fb950' if o==0 else '#f85149' for o in df['Outcome']]
    ax8.scatter(df['Glucose'], df['BMI'], c=scatter_colors, alpha=0.4, s=15, edgecolors='none')
    ax8.axvline(126, color='#d29922', linestyle='--', linewidth=1.2, alpha=0.8)
    ax8.axhline(30, color='#d29922', linestyle='--', linewidth=1.2, alpha=0.8)
    p1 = mpatches.Patch(color='#3fb950', label="Sog'lom")
    p2 = mpatches.Patch(color='#f85149', label='Diabet')
    ax8.legend(handles=[p1, p2], facecolor=ax_color, labelcolor=text_color,
               edgecolor=grid_color, fontsize=8)
    ax8.set_xlabel("Glyukoza (mg/dL)")
    ax8.set_ylabel("BMI (kg/m²)")
    ax8.grid(color=grid_color, alpha=0.3)

    # --- 9. Model natijalari ---
    ax9 = fig.add_subplot(3, 3, 9)
    ax9.set_facecolor(ax_color)
    ax9.axis('off')
    ax9.set_title("Model Natijalari", color=title_color, fontsize=11, fontweight='bold', pad=10)
    report = classification_report(y_test, y_pred, target_names=["Sog'lom","Diabet"], output_dict=True)
    acc = accuracy_score(y_test, y_pred) * 100
    lines = [
        ("", ""),
        ("🎯 Umumiy aniqlik",    f"{acc:.1f}%"),
        ("📊 ROC-AUC",           f"{roc_auc:.3f}"),
        ("", ""),
        ("── Sog'lom ──", ""),
        ("  Precision",          "{:.3f}".format(report["Sog'lom"]['precision'])),
        ("  Recall",             "{:.3f}".format(report["Sog'lom"]['recall'])),
        ("  F1-Score",           "{:.3f}".format(report["Sog'lom"]['f1-score'])),
        ("", ""),
        ("── Diabet ──", ""),
        ("  Precision",          f"{report['Diabet']['precision']:.3f}"),
        ("  Recall",             f"{report['Diabet']['recall']:.3f}"),
        ("  F1-Score",           f"{report['Diabet']['f1-score']:.3f}"),
    ]
    y_pos = 0.95
    for label, value in lines:
        if not label and not value:
            y_pos -= 0.05
            continue
        color = '#58a6ff' if label.startswith('──') else text_color
        ax9.text(0.05, y_pos, label, transform=ax9.transAxes,
                 color=color, fontsize=9.5, verticalalignment='top')
        if value:
            c = '#3fb950' if float(value.replace('%','')) > 0.75 or (value.endswith('%') and float(value[:-1]) > 75) else '#d29922'
            ax9.text(0.75, y_pos, value, transform=ax9.transAxes,
                     color=c, fontsize=9.5, verticalalignment='top', fontweight='bold')
        y_pos -= 0.07

    plt.tight_layout(rect=[0,0,1,0.96])
    out_path = 'natijalar_vizualizatsiya.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117', edgecolor='none')
    print(f"\n✅ Grafik saqlandi: {out_path}")
    plt.show()


# ============================================================
# 4. MODELNI O'QITISH VA BAHOLASH
# ============================================================

def model_orgatish(df):
    print("\n" + "="*60)
    print("  🤖 MODEL O'QITISH VA BAHOLASH")
    print("="*60)

    feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']
    X = df[feature_names].values
    y = df['Outcome'].values

    # Ma'lumotlarni bo'lish: 80% train, 20% test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Normalizatsiya
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print(f"\n📦 O'qitish namunalari : {X_train.shape[0]}")
    print(f"📦 Test namunalari     : {X_test.shape[0]}")

    # ---- Random Forest ----
    print("\n⚙️  Random Forest Classifier o'qitilmoqda...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_sc, y_train)

    y_pred_rf = rf_model.predict(X_test_sc)
    y_prob_rf = rf_model.predict_proba(X_test_sc)[:, 1]
    acc_rf    = accuracy_score(y_test, y_pred_rf)
    cv_scores = cross_val_score(rf_model, X_train_sc, y_train, cv=5, scoring='accuracy')

    # ---- Decision Tree ----
    print("⚙️  Decision Tree Classifier o'qitilmoqda...")
    dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
    dt_model.fit(X_train_sc, y_train)
    y_pred_dt = dt_model.predict(X_test_sc)
    acc_dt    = accuracy_score(y_test, y_pred_dt)

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
    roc_auc     = auc(fpr, tpr)

    print("\n" + "─"*60)
    print("  📈 MODEL TAQQOSLASH NATIJALARI")
    print("─"*60)
    print(f"  {'Model':<30} {'Aniqlik':>10}")
    print(f"  {'─'*30} {'─'*10}")
    print(f"  {'Random Forest':<30} {acc_rf*100:>9.2f}%")
    print(f"  {'Decision Tree':<30} {acc_dt*100:>9.2f}%")
    print("─"*60)

    print(f"\n🏆 Tanlangan model : Random Forest")
    print(f"🎯 Test aniqligi   : {acc_rf*100:.2f}%")
    print(f"📊 Cross-val (5-fold): {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print(f"📈 ROC-AUC         : {roc_auc:.4f}")

    print("\n📋 Batafsil hisobot:")
    print(classification_report(y_test, y_pred_rf, target_names=["Sog'lom", "Diabet"]))

    return rf_model, scaler, y_test, y_pred_rf, y_prob_rf, feature_names, fpr, tpr, roc_auc


# ============================================================
# 5. YANGI BEMOR TASHXISI
# ============================================================

def bemor_tashxisi(model, scaler, bemor_malumotlari: dict):
    """
    Yangi bemor uchun diabet tashxisi qilish.

    Parametrlar:
        bemor_malumotlari (dict): {
            'Pregnancies': int,
            'Glucose': int,
            'BloodPressure': int,
            'SkinThickness': int,
            'Insulin': int,
            'BMI': float,
            'DiabetesPedigreeFunction': float,
            'Age': int
        }
    """
    feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness',
                     'Insulin','BMI','DiabetesPedigreeFunction','Age']
    X_new = np.array([[bemor_malumotlari[f] for f in feature_names]])
    X_new_sc = scaler.transform(X_new)

    prediction = model.predict(X_new_sc)[0]
    probability = model.predict_proba(X_new_sc)[0]

    print("\n" + "="*60)
    print("  🏥 BEMOR TASHXISI NATIJASI")
    print("="*60)
    print("\n📋 Bemor ma'lumotlari:")
    col1 = "Ko'rsatkich"
    col2 = "Qiymat"
    print(f"  {col1:<35} {col2:>10}")
    sep = "─"
    print(f"  {sep*35} {sep*10}")
    labels = {
        'Pregnancies'             : ('Homiladorlik soni',          ''),
        'Glucose'                 : ('Qon shakari',                'mg/dL'),
        'BloodPressure'           : ('Arterial bosim',             'mm Hg'),
        'SkinThickness'           : ('Teri qalinligi',             'mm'),
        'Insulin'                 : ('Insulin darajasi',           'mu U/ml'),
        'BMI'                     : ('Tana massasi indeksi (BMI)', 'kg/m²'),
        'DiabetesPedigreeFunction': ('Oilaviy diabet ko\'r.',      ''),
        'Age'                     : ('Yosh',                       'yil'),
    }
    for k, v in bemor_malumotlari.items():
        name, unit = labels[k]
        print(f"  {name:<35} {v:>7} {unit}")

    diabet_prob = probability[1] * 100
    soghlom_prob = probability[0] * 100

    print(f"\n{'─'*60}")
    print(f"  Sog'lom ehtimoli   : {soghlom_prob:>6.1f}%")
    print(f"  Diabet ehtimoli    : {diabet_prob:>6.1f}%")
    print(f"{'─'*60}")

    if prediction == 1:
        xavf = "YUQORI" if diabet_prob >= 70 else "O'RTA"
        print(f"\n  🔴 TASHXIS: DIABET XAVFI BOR  [{xavf} XAVF]")
        print("\n  ⚠️  Tavsiya:")
        print("     • Darhol endokrinolog shifokorga murojaat qiling")
        print("     • Qon shakarini muntazam tekshiring")
        print("     • Parhez va jismoniy mashqlarga e'tibor bering")
        print("     • Bu natija tibbiy tashxisni ALMASHTIRMAYDI")
    else:
        xavf = "PAST" if diabet_prob < 30 else "O'RTA-PAST"
        print(f"\n  🟢 TASHXIS: DIABET XAVFI YO'Q  [{xavf} XAVF]")
        print("\n  ✅ Tavsiya:")
        print("     • Yiliga bir marta profilaktik tekshiruv o'ting")
        print("     • Sog'lom turmush tarzini davom ettiring")
        print("     • Qon shakarini yiliga bir marta tekshiring")
    print("="*60)

    return prediction, probability


# ============================================================
# 6. INTERAKTIV DEMO
# ============================================================

def interaktiv_demo(model, scaler):
    """Foydalanuvchi ma'lumotlarini kiritib tashxis olishi uchun demo."""
    print("\n" + "="*60)
    print("  💻 INTERAKTIV TASHXIS DEMO")
    print("="*60)

    demo_bemorlar = [
        {
            "ism": "Bemor A (Yuqori xavf)",
            "data": {
                'Pregnancies': 6, 'Glucose': 168, 'BloodPressure': 88,
                'SkinThickness': 40, 'Insulin': 250, 'BMI': 38.5,
                'DiabetesPedigreeFunction': 0.921, 'Age': 52
            }
        },
        {
            "ism": "Bemor B (Past xavf)",
            "data": {
                'Pregnancies': 1, 'Glucose': 95, 'BloodPressure': 62,
                'SkinThickness': 18, 'Insulin': 45, 'BMI': 23.1,
                'DiabetesPedigreeFunction': 0.178, 'Age': 28
            }
        },
        {
            "ism": "Bemor C (O'rta xavf)",
            "data": {
                'Pregnancies': 3, 'Glucose': 128, 'BloodPressure': 76,
                'SkinThickness': 28, 'Insulin': 110, 'BMI': 31.2,
                'DiabetesPedigreeFunction': 0.450, 'Age': 41
            }
        }
    ]

    for bemor in demo_bemorlar:
        print(f"\n{'▶'*3} {bemor['ism']} uchun tashxis:")
        bemor_tashxisi(model, scaler, bemor['data'])
        print()


# ============================================================
# 7. ASOSIY FUNKSIYA
# ============================================================

def main():
    print("\n" + "█"*60)
    print("█" + " "*58 + "█")
    print("█   🏥 SUNIY INTELLEKT YORDAMIDA DIABET TASHXISI        █")
    print("█      Kurs ishi — Amaliy Dastur                        █")
    print("█      Algoritm: Random Forest & Decision Tree          █")
    print("█" + " "*58 + "█")
    print("█"*60)

    # 1. Ma'lumot yaratish
    print("\n⏳ Ma'lumotlar tayyorlanmoqda...")
    df = dataset_yaratish(n_samples=900, random_state=42)
    print("✅ Dataset tayyor!")

    # 2. Tahlil
    malumot_tahlili(df)

    # 3. Modelni o'qitish
    model, scaler, y_test, y_pred, y_prob, feature_names, fpr, tpr, roc_auc = model_orgatish(df)
    importances = model.feature_importances_

    # 4. Vizualizatsiya
    print("\n⏳ Grafiklar tayyorlanmoqda...")
    vizualizatsiya(df, y_test, y_pred, y_prob, feature_names, importances, fpr, tpr, roc_auc)

    # 5. Demo tashxislar
    interaktiv_demo(model, scaler)

    print("\n" + "█"*60)
    print("█   ✅ DASTUR MUVAFFAQIYATLI YAKUNLANDI                  █")
    print("█   ⚠️  Bu tizim faqat ta'lim maqsadida yaratilgan!      █")
    print("█   ⚠️  Tibbiy qaror qabul qilishda vrach maslahatiga    █")
    print("█      murojaat qiling.                                  █")
    print("█"*60 + "\n")


if __name__ == "__main__":
    main()
