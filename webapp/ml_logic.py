import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

COT_SO = [
    "Tuoi", "Glucose", "HbA1C", "Cholesterol", "HDL", "LDL",
    "Triglycerid", "Creatinine", "Ure", "AcidUric", "AST", "ALT",
]
COT_DAU_VAO = ["GioiTinh"] + COT_SO
COT_NHAN = "NguyCoTieuDuong"

NHAN_CAC_TRUONG = [
    ("GioiTinh", "Giới tính"),
    ("Tuoi", "Tuổi (năm)"),
    ("Glucose", "Glucose (mmol/L)"),
    ("HbA1C", "HbA1C (%)"),
    ("Cholesterol", "Cholesterol (mmol/L)"),
    ("HDL", "HDL (mmol/L)"),
    ("LDL", "LDL (mmol/L)"),
    ("Triglycerid", "Triglycerid (mmol/L)"),
    ("Creatinine", "Creatinine (umol/L)"),
    ("Ure", "Ure (mmol/L)"),
    ("AcidUric", "Acid Uric (umol/L)"),
    ("AST", "AST (U/L)"),
    ("ALT", "ALT (U/L)"),
]

KHOANG_THAM_CHIEU = {
    "Tuoi": ("0", "110", "năm"),
    "Glucose": ("3.9", "6.9", "mmol/L"),
    "HbA1C": ("4.0", "5.7", "%"),
    "Cholesterol": ("2.9", "5.2", "mmol/L"),
    "HDL": ("0.9", "1.6", "mmol/L"),
    "LDL": ("1.0", "3.4", "mmol/L"),
    "Triglycerid": ("0.5", "1.7", "mmol/L"),
    "Creatinine": ("53", "115", "umol/L"),
    "Ure": ("2.5", "7.1", "mmol/L"),
    "AcidUric": ("150", "420", "umol/L"),
    "AST": ("5", "40", "U/L"),
    "ALT": ("5", "41", "U/L"),
}


def doc_du_lieu(duong_dan):
    df = pd.read_csv(duong_dan)
    print(f"Da doc du lieu: {df.shape[0]} dong, {df.shape[1]} cot")
    return df


def dong_bo_du_lieu(df):
    df["GioiTinh"] = df["GioiTinh"].replace({"Male": "Nam", "Female": "Nữ"})

    df[COT_NHAN] = df[COT_NHAN].replace({
        "TRUE": 1, "FALSE": 0,
        "Có nguy cơ": 1, "Không có nguy cơ": 0,
        "1": 1, "0": 0,
    })
    df[COT_NHAN] = df[COT_NHAN].astype(int)
    return df


def xu_ly_du_lieu_loi(df):
    khoang_hop_ly = {
        "Glucose": (1, 30),
        "HbA1C": (3, 15),
        "Cholesterol": (2, 15),
        "HDL": (0.2, 5),
        "LDL": (0.2, 10),
        "Triglycerid": (0.2, 15),
        "Creatinine": (20, 200),
        "Ure": (1, 30),
        "AcidUric": (50, 800),
        "AST": (3, 300),
        "ALT": (3, 300),
        "Tuoi": (0, 110),
    }

    for cot, (min_hop_ly, max_hop_ly) in khoang_hop_ly.items():
        if cot not in df.columns:
            continue
        mask_bat_thuong = (df[cot] < min_hop_ly) | (df[cot] > max_hop_ly)
        so_bat_thuong = mask_bat_thuong.sum()
        if so_bat_thuong == 0:
            continue

        print(f"  Cot {cot}: phat hien {so_bat_thuong} gia tri bat thuong -> {df.loc[mask_bat_thuong, cot].tolist()}")

        for idx in df.loc[mask_bat_thuong].index:
            gia_tri = df.at[idx, cot]
            da_sua = False
            for he_so in (10, 100, 1000):
                gia_tri_moi = gia_tri / he_so
                if min_hop_ly <= gia_tri_moi <= max_hop_ly:
                    df.at[idx, cot] = gia_tri_moi
                    da_sua = True
                    break
            if not da_sua:
                df.at[idx, cot] = np.nan
    return df


def xu_ly_du_lieu_thieu(df):
    print("So o du lieu thieu TRUOC khi xu ly:")
    print(df.isna().sum().to_string())

    for cot in COT_SO:
        if df[cot].isna().sum() > 0:
            trung_vi = df[cot].median()
            df[cot] = df[cot].fillna(trung_vi)

    if df["GioiTinh"].isna().sum() > 0:
        gia_tri_pho_bien = df["GioiTinh"].mode()[0]
        df["GioiTinh"] = df["GioiTinh"].fillna(gia_tri_pho_bien)

    print("So o du lieu thieu SAU khi xu ly:")
    print(df.isna().sum().to_string())
    return df


def xu_ly_ngoai_lai(df):
    for cot in COT_SO:
        q1 = df[cot].quantile(0.25)
        q3 = df[cot].quantile(0.75)
        iqr = q3 - q1

        if iqr == 0:
            print(f"  Cot {cot}: IQR = 0 (du lieu qua tap trung) -> bo qua, khong xu ly ngoai lai")
            continue

        nguong_duoi = q1 - 1.5 * iqr
        nguong_tren = q3 + 1.5 * iqr

        so_ngoai_lai = ((df[cot] < nguong_duoi) | (df[cot] > nguong_tren)).sum()
        print(f"  Cot {cot}: {so_ngoai_lai} gia tri ngoai lai (nguong hop le: {nguong_duoi:.2f} - {nguong_tren:.2f})")

        df[cot] = df[cot].clip(lower=nguong_duoi, upper=nguong_tren)
    return df


def kiem_tra_trung_lap(df):
    so_dong_trung = df.duplicated().sum()
    print(f"So dong du lieu trung lap: {so_dong_trung}")
    if so_dong_trung > 0:
        df = df.drop_duplicates()
        print(f"Da loai bo dong trung lap, con lai {df.shape[0]} dong")
    return df


def lam_sach_du_lieu(df):
    print("\n--- 3.1.a Dong bo du lieu ---")
    df = dong_bo_du_lieu(df)

    print("\n--- 3.1.b Xu ly du lieu loi ---")
    df = xu_ly_du_lieu_loi(df)

    print("\n--- 3.1.c Xu ly du lieu thieu ---")
    df = xu_ly_du_lieu_thieu(df)

    print("\n--- 3.1.d Xu ly du lieu ngoai lai (IQR) ---")
    df = xu_ly_ngoai_lai(df)

    print("\n--- 3.1.e Kiem tra du lieu trung lap ---")
    df = kiem_tra_trung_lap(df)

    return df


def ma_hoa_du_lieu(df):
    df["GioiTinh"] = df["GioiTinh"].map({"Nam": 1, "Nữ": 0})
    return df


def chuan_hoa_du_lieu(X_train, X_val, X_test):
    bo_chuan_hoa = StandardScaler()
    X_train_scaled = bo_chuan_hoa.fit_transform(X_train)
    X_val_scaled = bo_chuan_hoa.transform(X_val)
    X_test_scaled = bo_chuan_hoa.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, bo_chuan_hoa


def chia_du_lieu(X, y):
    X_train, X_tam, y_train, y_tam = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tam, y_tam, test_size=0.50, stratify=y_tam, random_state=42
    )

    print(f"Tap Train:      {X_train.shape[0]} dong")
    print(f"Tap Validation: {X_val.shape[0]} dong")
    print(f"Tap Test:       {X_test.shape[0]} dong")

    return X_train, X_val, X_test, y_train, y_val, y_test


def huan_luyen_cac_mo_hinh(X_train, y_train):
    cac_mo_hinh = {
        "Logistic Regression": LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42),
        "SVM": SVC(class_weight="balanced", probability=True, random_state=42),
    }

    for ten_mo_hinh, mo_hinh in cac_mo_hinh.items():
        mo_hinh.fit(X_train, y_train)
        print(f"Da huan luyen xong: {ten_mo_hinh}")

    return cac_mo_hinh


def danh_gia_mo_hinh(mo_hinh, X, y):
    y_du_doan = mo_hinh.predict(X)
    accuracy = accuracy_score(y, y_du_doan)
    recall = recall_score(y, y_du_doan)
    f1 = f1_score(y, y_du_doan)
    return accuracy, recall, f1


def danh_gia_va_chon_mo_hinh(cac_mo_hinh, X_val, y_val):
    print(f"\n{'Mo hinh':<22}{'Accuracy':<12}{'Recall':<12}{'F1-score':<12}")
    print("-" * 58)

    ket_qua = {}
    for ten_mo_hinh, mo_hinh in cac_mo_hinh.items():
        accuracy, recall, f1 = danh_gia_mo_hinh(mo_hinh, X_val, y_val)
        ket_qua[ten_mo_hinh] = f1
        print(f"{ten_mo_hinh:<22}{accuracy:<12.4f}{recall:<12.4f}{f1:<12.4f}")

    ten_mo_hinh_tot_nhat = max(ket_qua, key=ket_qua.get)
    print(f"\n=> Mo hinh duoc chon (F1 cao nhat tren Validation): {ten_mo_hinh_tot_nhat}")
    return ten_mo_hinh_tot_nhat, cac_mo_hinh[ten_mo_hinh_tot_nhat]


def danh_gia_tren_tap_test(mo_hinh, ten_mo_hinh, X_test, y_test):
    accuracy, recall, f1 = danh_gia_mo_hinh(mo_hinh, X_test, y_test)
    print(f"\n--- Danh gia mo hinh '{ten_mo_hinh}' tren tap TEST ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Recall:   {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    y_du_doan = mo_hinh.predict(X_test)
    ma_tran_nham_lan = confusion_matrix(y_test, y_du_doan)
    print("Confusion matrix (hang = thuc te, cot = du doan):")
    print(ma_tran_nham_lan)


def lay_3_yeu_to_quan_trong_nhat(mo_hinh_rf, df_train_goc, y_train):
    do_quan_trong = pd.Series(mo_hinh_rf.feature_importances_, index=COT_DAU_VAO)
    top_3 = do_quan_trong.sort_values(ascending=False).head(3).index.tolist()

    trung_binh_nhom_khong_nguy_co = df_train_goc[y_train == 0][top_3].mean()
    return top_3, trung_binh_nhom_khong_nguy_co


def huan_luyen_toan_bo(duong_dan_csv):
    df = doc_du_lieu(duong_dan_csv)

    print("\n===== BUOC 3.1: LAM SACH DU LIEU =====")
    df = lam_sach_du_lieu(df)

    print("\n===== BUOC 3.2: MA HOA DU LIEU =====")
    df = ma_hoa_du_lieu(df)

    X = df[COT_DAU_VAO]
    y = df[COT_NHAN]

    print("\n===== BUOC 3.3: PHAN CHIA DU LIEU =====")
    X_train, X_val, X_test, y_train, y_val, y_test = chia_du_lieu(X, y)

    X_train_scaled, X_val_scaled, X_test_scaled, bo_chuan_hoa = chuan_hoa_du_lieu(X_train, X_val, X_test)

    print("\n===== BUOC 4 + 5: HUAN LUYEN CAC MO HINH =====")
    cac_mo_hinh = huan_luyen_cac_mo_hinh(X_train_scaled, y_train)

    print("\n===== BUOC 6: DANH GIA MO HINH (TAP VALIDATION) =====")
    bang_danh_gia = []
    for ten, mh in cac_mo_hinh.items():
        acc, rec, f1 = danh_gia_mo_hinh(mh, X_val_scaled, y_val)
        bang_danh_gia.append({"model": ten, "accuracy": acc, "recall": rec, "f1": f1})

    ten_tot_nhat, mo_hinh_tot_nhat = danh_gia_va_chon_mo_hinh(cac_mo_hinh, X_val_scaled, y_val)
    danh_gia_tren_tap_test(mo_hinh_tot_nhat, ten_tot_nhat, X_test_scaled, y_test)
    test_acc, test_rec, test_f1 = danh_gia_mo_hinh(mo_hinh_tot_nhat, X_test_scaled, y_test)

    mo_hinh_rf = cac_mo_hinh["Random Forest"]
    top3, trung_binh_khong_nguy_co = lay_3_yeu_to_quan_trong_nhat(mo_hinh_rf, X_train, y_train)

    return {
        "best_name": ten_tot_nhat,
        "best_model": mo_hinh_tot_nhat,
        "scaler": bo_chuan_hoa,
        "comparison": bang_danh_gia,
        "test_metrics": {"accuracy": test_acc, "recall": test_rec, "f1": test_f1},
        "top3": top3,
        "avg_no_risk": trung_binh_khong_nguy_co,
    }
