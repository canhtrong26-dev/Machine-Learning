const form = document.getElementById("predict-form");
const errorBox = document.getElementById("form-error");
const verdictEmpty = document.getElementById("verdict-empty");
const verdictContent = document.getElementById("verdict-content");
const stamp = document.getElementById("stamp");
const stampRing = stamp.querySelector(".stamp-ring circle");
const probValue = document.getElementById("prob-value");
const levelBadge = document.getElementById("level-badge");
const explainList = document.getElementById("explain-list");

const RING_CIRCUMFERENCE = 2 * Math.PI * 54;

const SAMPLES = {
  low: {
    GioiTinh: "Nữ", Tuoi: 28, Glucose: 5.1, HbA1C: 5.2, Cholesterol: 4.6, HDL: 1.5,
    LDL: 2.6, Triglycerid: 1.1, Creatinine: 65, Ure: 4.5, AcidUric: 300, AST: 18, ALT: 16,
  },
  high: {
    GioiTinh: "Nam", Tuoi: 58, Glucose: 9.8, HbA1C: 8.9, Cholesterol: 6.4, HDL: 0.9,
    LDL: 4.3, Triglycerid: 3.2, Creatinine: 110, Ure: 8.1, AcidUric: 470, AST: 42, ALT: 45,
  },
};

function fillSample(sample) {
  Object.entries(sample).forEach(([key, value]) => {
    const el = document.getElementById(`f-${key}`);
    if (el) el.value = value;
  });
  errorBox.textContent = "";
}

document.getElementById("btn-sample-low").addEventListener("click", () => fillSample(SAMPLES.low));
document.getElementById("btn-sample-high").addEventListener("click", () => fillSample(SAMPLES.high));

function levelKeyOf(level) {
  if (level === "THAP") return "low";
  if (level === "TRUNG_BINH") return "mid";
  return "high";
}

function levelLabelOf(level) {
  if (level === "THAP") return "Nguy cơ thấp";
  if (level === "TRUNG_BINH") return "Nguy cơ trung bình";
  return "Nguy cơ cao";
}

stampRing.style.strokeDasharray = `${RING_CIRCUMFERENCE}`;
stampRing.style.strokeDashoffset = `${RING_CIRCUMFERENCE}`;
stampRing.style.transition = "stroke-dashoffset .7s ease";

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.textContent = "";

  const formData = new FormData(form);
  const payload = Object.fromEntries(formData.entries());

  for (const [key, value] of Object.entries(payload)) {
    if (key === "GioiTinh") continue;
    if (value.trim() === "" || Number.isNaN(Number(value))) {
      errorBox.textContent = "Vui lòng điền đủ giá trị hợp lệ cho tất cả các chỉ số trước khi lập phiếu.";
      return;
    }
  }

  const submitBtn = form.querySelector(".submit-btn");
  submitBtn.disabled = true;
  submitBtn.textContent = "Đang đối chiếu…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    if (!res.ok) {
      errorBox.textContent = data.error || "Không lập được phiếu, vui lòng kiểm tra lại số liệu.";
      return;
    }

    renderVerdict(data);
  } catch (err) {
    errorBox.textContent = "Không kết nối được máy chủ cục bộ. Kiểm tra app.py có đang chạy không.";
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = "Đối chiếu & lập phiếu";
  }
});

const LEVEL_TINTS = {
  low: "#DCFCE7",
  mid: "#FEF3C7",
  high: "#FEE2E2",
};

function renderVerdict(data) {
  verdictEmpty.classList.add("hidden");
  verdictContent.classList.remove("hidden");

  const pct = data.probability;
  const levelKey = levelKeyOf(data.level);
  stamp.dataset.level = levelKey;
  document.querySelector(".stamp-row").style.setProperty("--level-tint", LEVEL_TINTS[levelKey]);

  const offset = RING_CIRCUMFERENCE - (RING_CIRCUMFERENCE * Math.min(pct, 100)) / 100;
  requestAnimationFrame(() => {
    stampRing.style.strokeDashoffset = `${offset}`;
  });

  probValue.textContent = `${pct}%`;
  levelBadge.textContent = levelLabelOf(data.level);

  explainList.innerHTML = "";
  data.explanations.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item.text;
    explainList.appendChild(li);
  });

  verdictContent.scrollIntoView({ behavior: "smooth", block: "nearest" });
}
