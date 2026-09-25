import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from flask import Flask, jsonify, render_template, request

import ml_logic as loi

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(
    __name__,
    template_folder=os.path.join(_APP_DIR, "templates"),
    static_folder=os.path.join(_APP_DIR, "static"),
)

_CSV_PATH = os.path.join(_APP_DIR, "CSDL_TieuDuong.csv")
_ket_qua = loi.huan_luyen_toan_bo(_CSV_PATH)

_best_name = _ket_qua["best_name"]
_best_model = _ket_qua["best_model"]
_scaler = _ket_qua["scaler"]
_comparison = _ket_qua["comparison"]
_test_metrics = _ket_qua["test_metrics"]
_top3 = _ket_qua["top3"]
_avg_no_risk = _ket_qua["avg_no_risk"]

print(f"\n[webapp] Mo hinh dang phuc vu: {_best_name}")
print(f"[webapp] Test Accuracy={_test_metrics['accuracy']:.4f} "
      f"Recall={_test_metrics['recall']:.4f} F1={_test_metrics['f1']:.4f}")


def _tach_nhan_va_don_vi(nhan):
    if "(" in nhan and nhan.endswith(")"):
        ten, don_vi = nhan[:-1].split("(", 1)
        return ten.strip(), don_vi.strip()
    return nhan, ""


@app.route("/")
def trang_chu():
    truong_xet_nghiem = []
    for ten_cot, nhan in loi.NHAN_CAC_TRUONG:
        if ten_cot == "GioiTinh":
            continue
        ten_ngan, don_vi = _tach_nhan_va_don_vi(nhan)
        thap, cao, _ = loi.KHOANG_THAM_CHIEU[ten_cot]
        truong_xet_nghiem.append((ten_cot, ten_ngan, don_vi, thap, cao))

    return render_template(
        "index.html",
        fields=truong_xet_nghiem,
        best_model=_best_name,
        metrics=_test_metrics,
        comparison=_comparison,
    )


@app.route("/api/predict", methods=["POST"])
def du_doan():
    du_lieu = request.get_json(force=True)

    gia_tri_nhap = {}
    for ten_cot, nhan in loi.NHAN_CAC_TRUONG:
        if ten_cot == "GioiTinh":
            gia_tri_nhap[ten_cot] = 1 if du_lieu.get("GioiTinh") == "Nam" else 0
            continue
        try:
            gia_tri_nhap[ten_cot] = float(du_lieu.get(ten_cot))
        except (TypeError, ValueError):
            return jsonify({"error": f"Giá trị '{nhan}' không hợp lệ"}), 400

    df_dau_vao = pd.DataFrame([gia_tri_nhap])[loi.COT_DAU_VAO]
    du_lieu_chuan_hoa = _scaler.transform(df_dau_vao)
    xac_suat = float(_best_model.predict_proba(du_lieu_chuan_hoa)[0][1] * 100)

    if xac_suat < 30:
        muc_do = "THAP"
    elif xac_suat <= 70:
        muc_do = "TRUNG_BINH"
    else:
        muc_do = "CAO"

    giai_thich = []
    for ten_cot in _top3:
        gia_tri_nguoi_dung = gia_tri_nhap[ten_cot]
        gia_tri_trung_binh = float(_avg_no_risk[ten_cot])
        so_sanh = "cao hơn" if gia_tri_nguoi_dung > gia_tri_trung_binh else "thấp hơn"
        giai_thich.append({
            "field": ten_cot,
            "user_value": gia_tri_nguoi_dung,
            "avg_value": gia_tri_trung_binh,
            "text": f"{ten_cot} của bạn là {gia_tri_nguoi_dung:.1f}, {so_sanh} mức trung bình nhóm bình thường ({gia_tri_trung_binh:.1f})",
        })

    return jsonify({
        "probability": round(xac_suat, 1),
        "level": muc_do,
        "explanations": giai_thich,
        "model": _best_name,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
