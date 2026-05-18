(function () {
  "use strict";

  let activeCounselingId = null;

  function getCsrfToken() {
    const meta = document.querySelector('meta[name="csrf-token"]');
    return meta ? meta.getAttribute("content") : "";
  }

  // ─── Panel ────────────────────────────────────────────────────────────────
  function openPanel(counselingId, currentStatus) {
    activeCounselingId = counselingId;

    document.getElementById("panel-title").textContent =
      "Kelola Konseling #" + counselingId;
    document.getElementById("manage-panel").style.display = "block";
    document.getElementById("panel-placeholder").style.display = "none";

    document.getElementById("scheduled_at").value = "";
    document.getElementById("notes").value = "";
    hideFeedback();

    document.getElementById("section-schedule").style.display =
      currentStatus === "scheduled" ? "none" : "block";
  }

  function closePanel() {
    activeCounselingId = null;
    document.getElementById("manage-panel").style.display = "none";
    document.getElementById("panel-placeholder").style.display = "block";
  }

  // ─── API Call ─────────────────────────────────────────────────────────────
  async function patchCounseling(payload) {
    if (!activeCounselingId) return null;
    try {
      const res = await fetch(
        `/konselor/api/konseling/${activeCounselingId}`,
        {
          method: "PATCH",
          headers: {
            "Content-Type": "application/json",
            "X-CSRFToken": getCsrfToken(),
          },
          body: JSON.stringify(payload),
        }
      );
      if (!res.ok) {
        const errText = await res.text();
        throw new Error(`HTTP ${res.status}: ${errText}`);
      }
      return await res.json();
    } catch (err) {
      console.error("patchCounseling error:", err);
      return { status: "error", message: err.message };
    }
  }

  function showFeedback(msg, isError = false) {
    const el = document.getElementById("panel-feedback");
    el.style.display = "block";
    el.style.color = isError ? "#EF4444" : "#10B981";
    el.textContent = msg;
  }

  function hideFeedback() {
    const el = document.getElementById("panel-feedback");
    if (el) el.style.display = "none";
  }

  function updateStatusLabel(counselingId, record) {
    const el = document.getElementById("status-label-" + counselingId);
    if (!el) return;

    if (record.status === "scheduled") {
      el.style.color = "var(--color-primary)";
      const dt = record.scheduled_at ? record.scheduled_at.slice(0, 16) : "-";
      el.textContent = "Dijadwal: " + dt;
    } else if (record.status === "done") {
      el.style.color = "#10B981";
      const dt = record.completed_at ? record.completed_at.slice(0, 10) : "-";
      el.textContent = "Selesai: " + dt;
      const btn = document.getElementById("btn-" + counselingId);
      if (btn) btn.style.display = "none";
      closePanel();
    } else if (record.status === "skipped") {
      el.style.color = "#64748B";
      el.textContent = "Dilewati";
      const btn = document.getElementById("btn-" + counselingId);
      if (btn) btn.style.display = "none";
      closePanel();
    }
  }

  // ─── Actions ──────────────────────────────────────────────────────────────
  async function scheduleIt() {
    const dt = document.getElementById("scheduled_at").value;
    if (!dt) {
      showFeedback("Pilih tanggal dan waktu terlebih dahulu.", true);
      return;
    }
    const formatted = dt.replace("T", " ") + ":00";
    const json = await patchCounseling({ status: "scheduled", scheduled_at: formatted });
    if (!json) { showFeedback("Kesalahan internal.", true); return; }
    if (json.status === "success") {
      showFeedback("Jadwal berhasil disimpan.");
      updateStatusLabel(activeCounselingId, json.record);
      document.getElementById("section-schedule").style.display = "none";
    } else {
      showFeedback(json.message || "Gagal menyimpan jadwal.", true);
    }
  }

  async function saveNotes() {
    const notes = document.getElementById("notes").value.trim();
    if (!notes) { showFeedback("Catatan tidak boleh kosong.", true); return; }
    const json = await patchCounseling({ notes });
    if (!json) { showFeedback("Kesalahan internal.", true); return; }
    if (json.status === "success") {
      showFeedback("Catatan berhasil disimpan.");
    } else {
      showFeedback(json.message || "Gagal menyimpan catatan.", true);
    }
  }

  async function markDone() {
    if (!confirm("Tandai sesi konseling ini sebagai selesai?")) return;
    const json = await patchCounseling({ status: "done" });
    if (!json) { showFeedback("Kesalahan internal.", true); return; }
    if (json.status === "success") {
      updateStatusLabel(activeCounselingId, json.record);
    } else {
      showFeedback(json.message || "Gagal memperbarui status.", true);
    }
  }

  async function markSkipped() {
    if (!confirm("Lewati sesi konseling ini? Mahasiswa tidak akan muncul di antrian untuk prediksi ini.")) return;
    const json = await patchCounseling({ status: "skipped" });
    if (!json) { showFeedback("Kesalahan internal.", true); return; }
    if (json.status === "success") {
      updateStatusLabel(activeCounselingId, json.record);
    } else {
      showFeedback(json.message || "Gagal memperbarui status.", true);
    }
  }

  // ─── Init: event delegation (menggantikan onclick yang diblokir CSP) ──────
  //
  // CSP dengan nonce HANYA mengizinkan <script nonce="..."> bukan onclick/onXxx.
  // Event delegation di sini menangani semua tombol tanpa atribut onclick sama sekali.
  //
  function init() {
    // Delegation untuk tombol "Kelola" (data-open-panel) yang di-render dalam loop
    document.addEventListener("click", function (e) {
      const openBtn = e.target.closest("[data-open-panel]");
      if (openBtn) {
        openPanel(openBtn.dataset.counselingId, openBtn.dataset.counselingStatus);
      }
    });

    const closePanelBtn = document.getElementById("btn-close-panel");
    if (closePanelBtn) closePanelBtn.addEventListener("click", closePanel);

    const scheduleBtn = document.getElementById("btn-schedule-action");
    if (scheduleBtn) scheduleBtn.addEventListener("click", scheduleIt);

    const notesBtn = document.getElementById("btn-save-notes-action");
    if (notesBtn) notesBtn.addEventListener("click", saveNotes);

    const doneBtn = document.getElementById("btn-mark-done-action");
    if (doneBtn) doneBtn.addEventListener("click", markDone);

    const skipBtn = document.getElementById("btn-mark-skip-action");
    if (skipBtn) skipBtn.addEventListener("click", markSkipped);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
