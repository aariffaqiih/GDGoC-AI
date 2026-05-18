(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let currentFile = null;
  let currentResults = null;
  let currentErrors = null;
  let currentPage = 1;
  const rowsPerPage = 20;
  let abortController = null;

  // Dapatkan CSRF token dari meta tag
  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute('content') : '';
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(2) + " MB";
  }

  function showError(msg, details = null) {
    const el = $("batch-error");
    el.innerHTML = "";
    el.textContent = msg;
    if (details) {
      const detailEl = document.createElement("div");
      detailEl.className = "error-details";
      detailEl.textContent = details;
      el.appendChild(detailEl);
    }
    el.classList.remove("hidden");
  }

  function clearError() {
    const el = $("batch-error");
    el.innerHTML = "";
    el.classList.add("hidden");
  }

  function setFileSelected(file) {
    currentFile = file;
    $("drop-inner").classList.add("hidden");
    $("file-selected-info").classList.remove("hidden");
    $("file-name").textContent = file.name;
    $("file-size").textContent = formatBytes(file.size);
    $("btn-batch-submit").disabled = false;
    clearError();
  }

  function clearFile() {
    currentFile = null;
    $("csv-input").value = "";
    $("drop-inner").classList.remove("hidden");
    $("file-selected-info").classList.add("hidden");
    $("btn-batch-submit").disabled = true;
    clearError();
  }

  function setLoading(loading) {
    const btn = $("btn-batch-submit");
    const label = btn.querySelector(".btn-label");
    const loader = btn.querySelector(".btn-loader");
    btn.disabled = loading;
    if (label) label.classList.toggle("hidden", loading);
    if (loader) loader.classList.toggle("hidden", !loading);
    const loadingText = $("loading-text");
    if (loadingText) loadingText.classList.toggle("hidden", !loading);
  }

  function riskClass(label) {
    if (!label) return "";
    const l = label.toLowerCase();
    if (l.includes("tinggi")) return "tinggi";
    if (l.includes("sedang")) return "sedang";
    return "rendah";
  }

  const ALWAYS_COLS = [
    "location_type",
    "family_income",
    "financial_aid_status",
    "distance_to_institute",
    "internet_connectivity_issues",
    "motivation_score",
    "career_alignment",
    "stress_levels",
    "family_support",
    "attendance_rate",
    "test_scores_avg",
    "backlogs",
    "teaching_quality_rating",
    "kemungkinan_bertahan_pct",
    "kemungkinan_dropout_pct",
    "tingkat_risiko",
  ];

  const COL_LABELS = {
    nim: "NIM",
    nama: "Nama",
    location_type: "Lokasi",
    family_income: "Pendapatan (Rp)",
    financial_aid_status: "Bantuan",
    distance_to_institute: "Jarak (km)",
    internet_connectivity_issues: "Koneksi",
    motivation_score: "Motivasi",
    career_alignment: "Karir",
    stress_levels: "Stres",
    family_support: "Dukungan Kel.",
    attendance_rate: "Kehadiran (%)",
    test_scores_avg: "Nilai Rata-rata",
    backlogs: "Tunggakan",
    teaching_quality_rating: "Kualitas Pengajar",
    kemungkinan_bertahan_pct: "Bertahan (%)",
    kemungkinan_dropout_pct: "Dropout (%)",
    tingkat_risiko: "Tingkat Risiko",
  };

  const NUMERIC_COLS = new Set([
    "family_income",
    "distance_to_institute",
    "attendance_rate",
    "test_scores_avg",
    "backlogs",
    "teaching_quality_rating",
    "kemungkinan_bertahan_pct",
    "kemungkinan_dropout_pct",
  ]);

  function formatNumber(val, key) {
    if (val === undefined || val === null) return "-";
    if (key === "family_income") {
      const num = Number(val);
      if (isNaN(num)) return val;
      return new Intl.NumberFormat("id-ID").format(Math.round(num));
    }
    if (
      key === "kemungkinan_bertahan_pct" ||
      key === "kemungkinan_dropout_pct"
    ) {
      const num = Number(val);
      if (isNaN(num)) return val;
      return num.toFixed(1) + "%";
    }
    if (typeof val === "number") {
      if (key === "attendance_rate" || key === "test_scores_avg") {
        return val.toFixed(1);
      }
      return val;
    }
    return val;
  }

  function escapeHtml(str) {
    if (str == null) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function buildSummary(results, errors) {
    let tinggi = 0,
      sedang = 0,
      rendah = 0;
    results.forEach((r) => {
      const rc = riskClass(r.tingkat_risiko);
      if (rc === "tinggi") tinggi++;
      else if (rc === "sedang") sedang++;
      else rendah++;
    });
    const errorCount = errors ? errors.length : 0;
    const el = $("batch-summary");
    el.innerHTML = ` <span class="summary-pill total">${
      results.length
    } Mahasiswa</span> <span class="summary-pill risk-tinggi">${tinggi} Risiko Tinggi</span> <span class="summary-pill risk-sedang">${sedang} Risiko Sedang</span> <span class="summary-pill risk-rendah">${rendah} Risiko Rendah</span> ${
      errorCount > 0
        ? `<span class="summary-pill error">${errorCount} baris error</span>`
        : ""
    } `;
  }

  function renderTfoot(tfoot, results) {
    if (!results || results.length === 0) {
      tfoot.style.display = "none";
      return;
    }
    let totalStay = 0,
      totalDropout = 0;
    results.forEach((r) => {
      const stay = parseFloat(r.kemungkinan_bertahan_pct);
      const dropout = parseFloat(r.kemungkinan_dropout_pct);
      if (!isNaN(stay)) totalStay += stay;
      if (!isNaN(dropout)) totalDropout += dropout;
    });
    const avgStay = (totalStay / results.length).toFixed(1);
    const avgDropout = (totalDropout / results.length).toFixed(1);
    const allKeys = Object.keys(results[0]);
    const identCols = allKeys.filter((k) => !ALWAYS_COLS.includes(k));
    const displayCols = [...identCols, ...ALWAYS_COLS];
    const stayIdx = displayCols.indexOf("kemungkinan_bertahan_pct");
    const dropoutIdx = displayCols.indexOf("kemungkinan_dropout_pct");
    const riskIdx = displayCols.indexOf("tingkat_risiko");
    const totalCols = displayCols.length;
    const labelSpan = stayIdx;
    const trailSpan = totalCols - riskIdx;
    tfoot.style.display = "table-footer-group";
    tfoot.innerHTML = ` <tr class="tfoot-avg"> <td colspan="${labelSpan}" class="tfoot-label">Rata-rata keseluruhan</td> <td class="numeric tfoot-stay">${avgStay}%</td> <td class="numeric tfoot-dropout">${avgDropout}%</td> <td colspan="${trailSpan}"></td> </tr> `;
  }

  function renderErrorDetails(errors) {
    const existing = document.getElementById("batch-error-detail");
    if (existing) existing.remove();
    if (!errors || errors.length === 0) return;
    const container = document.createElement("div");
    container.id = "batch-error-detail";
    container.className = "card";
    container.style.marginTop = "1rem";
    const header = document.createElement("div");
    header.className = "card-header";
    header.innerHTML = `<h2 class="card-title" style="color:var(--color-danger,#f0384a)">Baris Tidak Dapat Diproses (${errors.length})</h2>`;
    container.appendChild(header);
    const tableWrap = document.createElement("div");
    tableWrap.className = "table-wrap";
    const table = document.createElement("table");
    table.className = "result-table";
    const thead = document.createElement("thead");
    thead.innerHTML = `<tr><th>Baris (index)</th><th>Alasan Error</th></tr>`;
    table.appendChild(thead);
    const tbody = document.createElement("tbody");
    errors.forEach((e) => {
      const tr = document.createElement("tr");
      const tdRow = document.createElement("td");
      tdRow.className = "numeric";
      tdRow.textContent = e.row + 2; // baris 1-indexed
      const tdErr = document.createElement("td");
      tdErr.style.color = "var(--color-danger,#f0384a)";
      tdErr.textContent = escapeHtml(String(e.error));
      tr.appendChild(tdRow);
      tr.appendChild(tdErr);
      tbody.appendChild(tr);
    });
    table.appendChild(tbody);
    tableWrap.appendChild(table);
    container.appendChild(tableWrap);
    const resultCard = $("batch-result-card");
    resultCard.parentNode.insertBefore(container, resultCard.nextSibling);
  }

  function renderTablePage() {
    if (!currentResults || currentResults.length === 0) return;
    const totalPages = Math.ceil(currentResults.length / rowsPerPage);
    if (currentPage < 1) currentPage = 1;
    if (currentPage > totalPages) currentPage = totalPages;
    const start = (currentPage - 1) * rowsPerPage;
    const pageResults = currentResults.slice(start, start + rowsPerPage);
    const allKeys = Object.keys(currentResults[0]);
    const identCols = allKeys.filter((k) => !ALWAYS_COLS.includes(k));
    const displayCols = [...identCols, ...ALWAYS_COLS];

    // Header
    const thead = $("batch-thead");
    thead.innerHTML = "";
    const headerRow = document.createElement("tr");
    displayCols.forEach((k) => {
      const th = document.createElement("th");
      th.setAttribute("data-col", k);
      th.textContent = COL_LABELS[k] || k;
      headerRow.appendChild(th);
    });
    thead.appendChild(headerRow);

    // Body
    const tbody = $("batch-tbody");
    tbody.innerHTML = "";
    pageResults.forEach((row) => {
      const tr = document.createElement("tr");
      displayCols.forEach((k) => {
        const td = document.createElement("td");
        if (k === "tingkat_risiko") {
          const rc = riskClass(row[k]);
          const span = document.createElement("span");
          span.className = `risk-tag ${rc}`;
          span.textContent = escapeHtml(row[k]);
          td.appendChild(span);
        } else {
          const raw = row[k] !== undefined && row[k] !== null ? row[k] : "-";
          const formatted = formatNumber(raw, k);
          td.textContent = escapeHtml(formatted);
          if (NUMERIC_COLS.has(k)) td.className = "numeric";
        }
        tr.appendChild(td);
      });
      tbody.appendChild(tr);
    });

    renderTfoot($("batch-tfoot"), currentResults);
    const paginationDiv = $("pagination-controls");
    if (totalPages > 1) {
      paginationDiv.style.display = "flex";
      $(
        "pagination-info"
      ).textContent = `Halaman ${currentPage} dari ${totalPages}`;
      $("pagination-prev").disabled = currentPage === 1;
      $("pagination-next").disabled = currentPage === totalPages;
    } else {
      paginationDiv.style.display = "none";
    }

    buildSummary(currentResults, currentErrors);
    $("batch-result-card").classList.remove("hidden");
    $("batch-result-card").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
    renderErrorDetails(currentErrors);
  }

  function exportCSV() {
    if (!currentResults || !currentResults.length) return;
    const allKeys = Object.keys(currentResults[0]);
    const identCols = allKeys.filter((k) => !ALWAYS_COLS.includes(k));
    const displayCols = [...identCols, ...ALWAYS_COLS];
    const header = displayCols.join(",");
    const rows = currentResults.map((row) =>
      displayCols
        .map((k) => {
          let v = row[k] !== undefined && row[k] !== null ? row[k] : "";
          let s = String(v);
          // Cegah CSV injection dengan menambahkan apostrof jika diawali karakter berbahaya
          if (s && /^[=+\-@]/.test(s)) {
            s = "'" + s;
          }
          if (s.includes(",") || s.includes('"') || s.includes("\n")) {
            s = `"${s.replace(/"/g, '""')}"`;
          }
          return s;
        })
        .join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob(["\uFEFF" + csv], {
      type: "text/csv;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "hasil_analisis_dropout.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function handleBatchSubmit() {
    if (!currentFile) return;
    if (abortController) abortController.abort();
    abortController = new AbortController();
    const signal = abortController.signal;
    clearError();
    setLoading(true);
    $("batch-result-card").classList.add("hidden");
    const prev = document.getElementById("batch-error-detail");
    if (prev) prev.remove();
    currentPage = 1;
    currentResults = null;
    currentErrors = null;

    const fd = new FormData();
    fd.append("file", currentFile);
    try {
      const res = await fetch("/api/batch", {
        method: "POST",
        body: fd,
        signal,
        headers: {
          "X-CSRFToken": getCsrfToken()
        }
      });
      const json = await res.json();
      if (!res.ok || json.status === "error") {
        showError(
          json.message || "Terjadi kesalahan saat memproses file.",
          json.details || null
        );
      } else {
        currentResults = json.results || [];
        currentErrors = json.errors || [];
        renderTablePage();
        if (currentErrors.length > 0) {
          showError(
            `${currentErrors.length} baris tidak dapat diproses. Lihat detail di bawah tabel.`
          );
        }
      }
    } catch (err) {
      if (err.name === "AbortError") return;
      showError(
        "Tidak dapat terhubung ke server. Periksa koneksi internet Anda dan coba lagi."
      );
    } finally {
      setLoading(false);
      abortController = null;
    }
  }

  function prevPage() {
    if (currentPage > 1) {
      currentPage--;
      renderTablePage();
    }
  }

  function nextPage() {
    const totalPages = Math.ceil(currentResults.length / rowsPerPage);
    if (currentPage < totalPages) {
      currentPage++;
      renderTablePage();
    }
  }

  function initDropZone() {
    const zone = $("drop-zone");
    const input = $("csv-input");
    zone.addEventListener("click", (e) => {
      if (e.target.id === "btn-clear-file") return;
      if (!currentFile) input.click();
    });
    zone.addEventListener("keydown", (e) => {
      if ((e.key === "Enter" || e.key === " ") && !currentFile) input.click();
    });
    input.addEventListener("change", () => {
      const file = input.files[0];
      if (file) setFileSelected(file);
    });
    $("btn-clear-file").addEventListener("click", (e) => {
      e.stopPropagation();
      clearFile();
    });
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("drag-over");
    });
    zone.addEventListener("dragleave", () =>
      zone.classList.remove("drag-over")
    );
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("drag-over");
      const file = e.dataTransfer.files[0];
      if (!file) return;
      if (!file.name.toLowerCase().endsWith(".csv")) {
        showError("Hanya file berformat .csv yang diterima.");
        return;
      }
      setFileSelected(file);
    });
  }

  function init() {
    initDropZone();
    $("btn-batch-submit").addEventListener("click", handleBatchSubmit);
    $("btn-export").addEventListener("click", exportCSV);
    $("pagination-prev").addEventListener("click", prevPage);
    $("pagination-next").addEventListener("click", nextPage);
    if (!$("loading-text")) {
      const btn = $("btn-batch-submit");
      const span = document.createElement("span");
      span.id = "loading-text";
      span.className = "hidden";
      span.textContent = "Memproses file, harap tunggu...";
      btn.parentNode.insertBefore(span, btn.nextSibling);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();