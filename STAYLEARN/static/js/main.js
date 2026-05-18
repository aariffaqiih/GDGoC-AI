// main.js - dengan CSRF protection dan tombol riwayat
(function () {
  "use strict";
  const GAUGE_FULL = 251.2;
  const $ = (id) => document.getElementById(id);
  let currentResult = null;
  let currentNim = null;
  let abortController = null;

  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
  }

  function setDisplay(id, show) {
    const el = $(id);
    if (!el) return;
    el.classList.toggle("hidden", !show);
  }
  function readSlider(id, displayId) {
    const el = $(id);
    if (!el) return;
    const update = () => {
      $(displayId).textContent = el.value;
    };
    el.addEventListener("input", update);
    if (el.value) update();
  }
  function readSelect(id, displayId) {
    const el = $(id);
    if (!el) return;
    const update = () => {
      const val = el.value;
      $(displayId).textContent = val !== "" ? val : "-";
    };
    el.addEventListener("change", update);
    el.addEventListener("input", update);
    if (el.value) update();
  }
  function validateField(el) {
    const group = el.closest(".form-group");
    if (!group) return true;
    const errEl = group.querySelector(".form-error");
    if (!el.value && el.value !== 0 && el.value !== "0") {
      el.classList.add("invalid");
      if (errEl) errEl.textContent = "Kolom ini wajib diisi.";
      return false;
    }
    const min = parseFloat(el.getAttribute("min"));
    const max = parseFloat(el.getAttribute("max"));
    const val = parseFloat(el.value);
    if (!isNaN(min) && !isNaN(max) && (val < min || val > max)) {
      el.classList.add("invalid");
      if (errEl) errEl.textContent = `Nilai harus antara ${min} dan ${max}.`;
      return false;
    }
    el.classList.remove("invalid");
    if (errEl) errEl.textContent = "";
    return true;
  }
  function validateAll() {
    const form = document.getElementById("form-predict");
    const controls = form.querySelectorAll("input[required], select[required]");
    let valid = true;
    controls.forEach((el) => {
      if (!validateField(el)) valid = false;
    });
    return valid;
  }
  function buildPayload() {
    const form = document.getElementById("form-predict");
    const fd = new FormData(form);
    const payload = {};
    fd.forEach((v, k) => {
      payload[k] = isNaN(Number(v)) ? v : Number(v);
    });
    payload["location_type"] = form.elements["location_type"].value;
    return payload;
  }
  function setLoading(loading) {
    const btn = $("btn-submit");
    const label = btn.querySelector(".btn-label");
    const loader = btn.querySelector(".btn-loader");
    btn.disabled = loading;
    label.classList.toggle("hidden", loading);
    loader.classList.toggle("hidden", !loading);
  }
  function animateGauge(pct) {
    const arc = $("gauge-arc");
    const text = $("gauge-pct");
    if (!arc || !text) return;
    const offset = GAUGE_FULL - (GAUGE_FULL * pct) / 100;
    arc.style.transition =
      "stroke-dashoffset 0.9s cubic-bezier(0.22,1,0.36,1), stroke 0.4s ease";
    arc.style.strokeDashoffset = offset;
    const color = pct >= 70 ? "#10B981" : pct >= 40 ? "#F59E0B" : "#EF4444";
    arc.style.stroke = color;
    let current = 0;
    const target = pct;
    const step = () => {
      current = Math.min(current + (target - current) * 0.12 + 0.5, target);
      text.textContent = Math.round(current) + "%";
      if (current < target - 0.5) requestAnimationFrame(step);
      else text.textContent = Math.round(target) + "%";
    };
    requestAnimationFrame(step);
  }
  function animateBar(id, pct) {
    const el = $(id);
    if (!el) return;
    requestAnimationFrame(() => {
      el.style.width = pct + "%";
    });
  }
  function renderFactors(concerns, strengths) {
    const grid = $("factor-grid");
    if (!grid) return;
    const buildBlock = (type, title, items) => {
      const itemsHtml = items.length
        ? items
            .map(
              (t) =>
                `<div class="factor-item"><span class="factor-dot"></span><span>${escapeHtml(t)}</span></div>`
            )
            .join("")
        : `<p class="factor-none">Tidak ditemukan</p>`;
      return `<div class="factor-block ${type}"><div class="factor-block-title">${title}</div><div class="factor-items">${itemsHtml}</div></div>`;
    };
    grid.innerHTML =
      buildBlock("concern", "Faktor Risiko", concerns) +
      buildBlock("strength", "Faktor Protektif", strengths);
  }
  function escapeHtml(str) {
    if (!str) return "";
    return str.replace(/[&<>]/g, function(m) {
      if (m === "&") return "&amp;";
      if (m === "<") return "&lt;";
      if (m === ">") return "&gt;";
      return m;
    });
  }
  function renderResult(result) {
    currentResult = result;
    setDisplay("result-empty", false);
    setDisplay("result-error", false);
    setDisplay("result-content", true);
    animateGauge(result.p_stay);
    animateBar("bar-stay", result.p_stay);
    animateBar("bar-dropout", result.p_dropout);
    $("pct-stay").textContent = result.p_stay + "%";
    $("pct-dropout").textContent = result.p_dropout + "%";
    const badge = $("risk-badge");
    badge.textContent = result.risk_label;
    badge.className = "risk-badge " + result.risk_level;
    renderFactors(result.concerns || [], result.strengths || []);
    const rec = $("recommendation-box");
    rec.innerHTML = `<strong>Rekomendasi Tindak Lanjut</strong>${escapeHtml(result.recommendation)}`;

    // --- Career advice (SDG 4.4) ---
    const careerSection = document.getElementById("career-advice-section");
    const careerContent = document.getElementById("career-advice-content");
    if (careerSection && careerContent) {
      if (result.career_advice) {
        careerContent.textContent = result.career_advice;
        careerSection.classList.remove("hidden");
      } else {
        careerSection.classList.add("hidden");
      }
    }

    // --- Tampilkan tombol riwayat jika nim diisi ---
    const nimInput = document.getElementById("nim");
    const btnRiwayat = document.getElementById("btn-riwayat");
    if (btnRiwayat && nimInput && nimInput.value.trim() !== "") {
      btnRiwayat.href = "/riwayat?nim=" + encodeURIComponent(nimInput.value.trim());
      btnRiwayat.style.display = "inline-flex";
    } else if (btnRiwayat) {
      btnRiwayat.style.display = "none";
    }
  }
  function renderError(msg) {
    setDisplay("result-empty", false);
    setDisplay("result-content", false);
    setDisplay("result-error", true);
    $("error-message").textContent =
      msg || "Terjadi kesalahan yang tidak diketahui.";
    // Sembunyikan tombol riwayat saat error
    const btnRiwayat = document.getElementById("btn-riwayat");
    if (btnRiwayat) btnRiwayat.style.display = "none";
  }
  async function handleSubmit(e) {
    e.preventDefault();
    if (!validateAll()) return;
    if (abortController) abortController.abort();
    abortController = new AbortController();
    const signal = abortController.signal;
    setLoading(true);
    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": getCsrfToken()
        },
        body: JSON.stringify(buildPayload()),
        signal,
      });
      const json = await res.json();
      if (!res.ok || json.status === "error") {
        renderError(json.message || "Respons server tidak valid.");
      } else {
        renderResult(json.result);
        document
          .getElementById("result-panel")
          .scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      console.error("Error:", err);
      renderError(
        "Tidak dapat terhubung ke server. Periksa koneksi internet Anda dan coba lagi."
      );
    } finally {
      setLoading(false);
      abortController = null;
    }
  }
  async function fillRandom() {
    const btn = $("btn-random");
    btn.disabled = true;
    btn.textContent = "Mengambil data...";
    try {
      const res = await fetch("/api/random");
      if (!res.ok) throw new Error("Random data fetch failed");
      const data = await res.json();
      const form = document.getElementById("form-predict");
      Object.entries(data).forEach(([k, v]) => {
        const el = form.elements[k];
        if (!el) return;
        el.value = v;
        el.dispatchEvent(new Event("input", { bubbles: true }));
        el.dispatchEvent(new Event("change", { bubbles: true }));
        el.classList.remove("invalid");
        const group = el.closest(".form-group");
        if (group) {
          const err = group.querySelector(".form-error");
          if (err) err.textContent = "";
        }
      });
    } catch (err) {
      console.error("Random data error:", err);
      alert("Gagal mengambil data contoh. Coba lagi.");
    } finally {
      btn.disabled = false;
      btn.textContent = "Isi Data Contoh";
    }
  }
  function resetForm() {
    const form = document.getElementById("form-predict");
    form.reset();
    form
      .querySelectorAll(".form-control")
      .forEach((el) => el.classList.remove("invalid"));
    form.querySelectorAll(".form-error").forEach((el) => (el.textContent = ""));
    [
      "motivation_display",
      "attendance_display",
      "scores_display",
      "teaching_display",
      "stress_display",
    ].forEach((id) => {
      const el = $(id);
      if (el) el.textContent = "-";
    });
    setDisplay("result-content", false);
    setDisplay("result-error", false);
    setDisplay("result-empty", true);
    currentResult = null;
    const careerSection = document.getElementById("career-advice-section");
    if (careerSection) careerSection.classList.add("hidden");
    const btnRiwayat = document.getElementById("btn-riwayat");
    if (btnRiwayat) btnRiwayat.style.display = "none";
  }
  function init() {
    readSlider("motivation_score", "motivation_display");
    readSlider("attendance_rate", "attendance_display");
    readSlider("test_scores_avg", "scores_display");
    readSlider("teaching_quality_rating", "teaching_display");
    readSelect("stress_levels", "stress_display");
    document
      .getElementById("form-predict")
      .addEventListener("submit", handleSubmit);
    $("btn-random").addEventListener("click", fillRandom);
    $("btn-reset").addEventListener("click", resetForm);
    document
      .getElementById("form-predict")
      .querySelectorAll("input[required], select[required]")
      .forEach((el) => {
        el.addEventListener("blur", () => validateField(el));
      });
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();