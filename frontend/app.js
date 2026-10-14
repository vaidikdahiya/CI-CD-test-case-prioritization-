const API_BASE = window.API_BASE || "http://localhost:8000";

const els = {
  apiStatus: document.getElementById("apiStatus"),
  loadSampleBtn: document.getElementById("loadSampleBtn"),
  csvInput: document.getElementById("csvInput"),
  runBtn: document.getElementById("runBtn"),
  genSlider: document.getElementById("genSlider"),
  genVal: document.getElementById("genVal"),
  popSlider: document.getElementById("popSlider"),
  popVal: document.getElementById("popVal"),
  loadedInfo: document.getElementById("loadedInfo"),
  loadedCount: document.getElementById("loadedCount"),
  loadedTime: document.getElementById("loadedTime"),
  results: document.getElementById("results"),
  emptyState: document.getElementById("emptyState"),
  metricsGrid: document.getElementById("metricsGrid"),
  resultTableBody: document.querySelector("#resultTable tbody"),
  stages: document.querySelectorAll(".stage"),
};

let currentTests = [];
let timeChart = null;
let convChart = null;

function setStage(index, state) {
  els.stages.forEach((el, i) => {
    el.classList.remove("active", "done");
    if (state === "active" && i === index) el.classList.add("active");
    if (state === "done" && i <= index) el.classList.add("done");
  });
}

function resetStages() {
  els.stages.forEach((el) => el.classList.remove("active", "done"));
}

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    els.apiStatus.innerHTML = '<span class="pulse"></span>backend connected';
    els.apiStatus.classList.add("ok");
  } catch (e) {
    els.apiStatus.innerHTML = '<span class="pulse"></span>backend unreachable — start it with uvicorn';
    els.apiStatus.classList.add("err");
  }
}

function setLoadedTests(tests) {
  currentTests = tests;
  els.runBtn.disabled = tests.length === 0;
  els.loadedInfo.style.display = tests.length ? "block" : "none";
  els.loadedCount.textContent = tests.length;
  const total = tests.reduce((s, t) => s + Number(t.exec_time || 0), 0);
  els.loadedTime.textContent = total.toFixed(1);
  setStage(0, "done");
}

els.loadSampleBtn.addEventListener("click", async () => {
  try {
    setStage(0, "active");
    const res = await fetch(`${API_BASE}/api/sample?n=40`);
    const data = await res.json();
    setLoadedTests(data.tests);
  } catch (e) {
    alert("Could not load sample suite. Is the backend running?");
  }
});

els.csvInput.addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  try {
    setStage(0, "active");
    const res = await fetch(`${API_BASE}/api/upload`, { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Upload failed");
    setLoadedTests(data.tests);
  } catch (err) {
    alert(err.message);
  }
});

els.genSlider.addEventListener("input", () => (els.genVal.textContent = els.genSlider.value));
els.popSlider.addEventListener("input", () => (els.popVal.textContent = els.popSlider.value));

els.runBtn.addEventListener("click", async () => {
  if (currentTests.length === 0) return;
  els.runBtn.disabled = true;
  els.runBtn.textContent = "Running…";
  els.emptyState.style.display = "none";

  try {
    for (const [i] of ["Neural Risk", "Fuzzy Priority", "GA Ordering"].entries()) {
      setStage(i + 1, "active");
      await new Promise((r) => setTimeout(r, 220)); // brief staged reveal
    }

    const res = await fetch(`${API_BASE}/api/prioritize`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        tests: currentTests,
        generations: Number(els.genSlider.value),
        population_size: Number(els.popSlider.value),
      }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Pipeline failed");

    setStage(4, "done");
    renderResults(data);
  } catch (err) {
    alert(err.message);
    resetStages();
  } finally {
    els.runBtn.disabled = false;
    els.runBtn.textContent = "Run prioritization ▸";
  }
});

function renderResults(data) {
  els.results.style.display = "block";
  const m = data.metrics;

  const metricDefs = [
    { label: "CI time saved", value: `${m.ci_time_saved_pct}%`, cls: "" },
    { label: "APFD (optimized)", value: m.optimized_apfd, cls: "neutral" },
    { label: "APFD (baseline)", value: m.baseline_apfd, cls: "neutral" },
    { label: "Total tests", value: m.total_tests, cls: "neutral" },
    { label: "Suite time (s)", value: m.total_suite_time_sec, cls: "neutral" },
    { label: "Time to 80% value — before", value: `${m.time_to_80pct_value_before_sec}s`, cls: "" },
    { label: "Time to 80% value — after", value: `${m.time_to_80pct_value_after_sec}s`, cls: "" },
  ];

  els.metricsGrid.innerHTML = metricDefs
    .map(
      (d) => `<div class="metric"><div class="value ${d.cls}">${d.value}</div><div class="label">${d.label}</div></div>`
    )
    .join("");

  renderTimeChart(m);
  renderConvChart(m);
  renderTable(data.tests);
}

function renderTimeChart(m) {
  const ctx = document.getElementById("timeChart");
  if (timeChart) timeChart.destroy();
  timeChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Original order", "GA-optimized order"],
      datasets: [
        {
          label: "Seconds to reach 80% fault-detection value",
          data: [m.time_to_80pct_value_before_sec, m.time_to_80pct_value_after_sec],
          backgroundColor: ["#8992a8", "#4fd1a5"],
          borderRadius: 6,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8992a8" }, grid: { color: "#1c2130" } },
        y: { ticks: { color: "#8992a8" }, grid: { color: "#1c2130" } },
      },
    },
  });
}

function renderConvChart(m) {
  const ctx = document.getElementById("convChart");
  if (convChart) convChart.destroy();
  convChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: m.ga_convergence.map((_, i) => i + 1),
      datasets: [
        {
          label: "Best APFD so far",
          data: m.ga_convergence,
          borderColor: "#e0a63d",
          backgroundColor: "rgba(224,166,61,0.12)",
          fill: true,
          tension: 0.25,
          pointRadius: 0,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { ticks: { color: "#8992a8" }, grid: { display: false } },
        y: { ticks: { color: "#8992a8" }, grid: { color: "#1c2130" } },
      },
    },
  });
}

function badgeClass(label) {
  return { high: "badge-high", medium: "badge-medium", low: "badge-low" }[label] || "badge-medium";
}

function renderTable(tests) {
  els.resultTableBody.innerHTML = tests
    .map((t) => {
      const rankDelta = t.original_rank - t.new_rank;
      const rankClass = rankDelta > 0 ? "rank-up" : "rank-down";
      const arrow = rankDelta > 0 ? "▲" : rankDelta < 0 ? "▼" : "–";
      return `<tr>
        <td>${t.new_rank}</td>
        <td>${t.name}</td>
        <td class="${rankClass}">${t.original_rank} ${arrow}</td>
        <td>${t.exec_time.toFixed(2)}</td>
        <td>${t.risk_score.toFixed(3)}</td>
        <td><span class="badge ${badgeClass(t.priority_label)}">${t.priority_label} · ${t.priority_score.toFixed(0)}</span></td>
      </tr>`;
    })
    .join("");
}

checkHealth();
