// TENETDrill dashboard -- plain JS, no framework/build step.
// Talks to the FastAPI backend at window.TENETDRILL_API_URL (see env.js).

const API_BASE = window.TENETDRILL_API_URL;

const COLOR_RISK_LINE = "#3987e5";
const COLOR_STUCK_AREA = "#e66767";

const STATUS_COLORS = {
  low: "#0ca30c",
  "mild/moderate": "#fab219",
  elevated: "#ec835a",
  high: "#e66767",
  unmonitored: "#6b6a64",
};

let depthMin = 0;
let depthMax = 0;
let riskChart = null;
let lastRiskSeries = null;

async function fetchJSON(path, opts) {
  const res = await fetch(API_BASE + path, opts);
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${path} -> HTTP ${res.status}: ${body}`);
  }
  return res.json();
}

function fmt(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return Number(v).toFixed(digits);
}

function levelClass(level) {
  return "level-" + String(level).replace(/[\s/]+/g, "-").toLowerCase();
}

function levelLabel(level) {
  if (level === "mild/moderate") return "Mild / Moderate";
  if (!level) return "Unknown";
  return level.charAt(0).toUpperCase() + level.slice(1);
}

// ---------------------------------------------------------------------
// Overview cards
// ---------------------------------------------------------------------

async function loadOverview() {
  const ov = await fetchJSON("/api/overview");
  depthMin = ov.depth_range_m[0];
  depthMax = ov.depth_range_m[1];

  document.getElementById("wellId").textContent = ov.well;

  document.getElementById("cardRiskValue").textContent = fmt(ov.stuck_pipe_risk.score);
  const levelEl = document.getElementById("cardRiskLevel");
  const lvl = ov.stuck_pipe_risk.level;
  levelEl.innerHTML = `<span class="status-pill" style="background:${STATUS_COLORS[lvl] || STATUS_COLORS.unmonitored}22; color:${STATUS_COLORS[lvl] || STATUS_COLORS.unmonitored}">${levelLabel(lvl)}</span>`;

  document.getElementById("cardIntegrityValue").textContent =
    ov.well_integrity_pct === null ? "—" : fmt(ov.well_integrity_pct, 1) + "%";
  document.getElementById("cardIntegrityNote").textContent = ov.well_integrity_note;

  document.getElementById("cardNptValue").textContent = ov.npt_events;
  document.getElementById("cardNptNote").textContent = ov.npt_note;

  document.getElementById("cardDepthValue").textContent = fmt(ov.total_depth_m, 1) + " m";

  document.getElementById("depthRangeHint").textContent =
    `Logged range: ${fmt(depthMin, 1)}–${fmt(depthMax, 1)} m`;

  const depthInput = document.getElementById("depthInput");
  depthInput.min = depthMin;
  depthInput.max = depthMax;
  if (!depthInput.value) depthInput.value = ov.current_depth_m.toFixed(1);

  return ov;
}

// ---------------------------------------------------------------------
// Risk-over-depth chart (single axis: risk 0-1, stuck_rt normalized /3)
// ---------------------------------------------------------------------

async function loadRiskChart() {
  const data = await fetchJSON("/api/risk-over-depth?max_points=600");
  lastRiskSeries = data;

  const riskPoints = data.depth_m.map((d, i) => ({ x: d, y: data.ewma_risk[i] }));
  const stuckPoints = data.depth_m.map((d, i) => ({
    x: d,
    y: data.stuck_rt[i] === null ? null : data.stuck_rt[i] / 3,
  }));

  const ctx = document.getElementById("riskChart").getContext("2d");
  riskChart = new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        {
          label: "Actual stuck event (normalized)",
          data: stuckPoints,
          borderWidth: 0,
          backgroundColor: COLOR_STUCK_AREA + "40",
          fill: "origin",
          stepped: true,
          pointRadius: 0,
          spanGaps: false,
          order: 2,
        },
        {
          label: "TENETDrill risk score (EWMA)",
          data: riskPoints,
          borderColor: COLOR_RISK_LINE,
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
          tension: 0.15,
          spanGaps: true,
          order: 1,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "index", intersect: false },
      onClick: (evt) => {
        const points = riskChart.getElementsAtEventForMode(evt, "index", { intersect: false }, true);
        if (points.length) {
          const idx = points[0].index;
          const depth = riskChart.data.datasets[1].data[idx].x;
          setDepth(depth);
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#232322",
          borderColor: "#383835",
          borderWidth: 1,
          titleColor: "#ffffff",
          bodyColor: "#c3c2b7",
          callbacks: {
            title: (items) => `Depth ${fmt(items[0].parsed.x, 1)} m`,
            label: (item) => {
              if (item.datasetIndex === 1) return `Risk score: ${fmt(item.parsed.y)}`;
              const raw = item.raw.y;
              return raw === null ? "No STUCK_RT label here" : `Actual STUCK_RT level: ${Math.round(raw * 3)}`;
            },
          },
        },
      },
      scales: {
        x: {
          type: "linear",
          title: { display: true, text: "Measured Depth (m)", color: "#898781" },
          grid: { color: "#2c2c2a" },
          ticks: { color: "#898781" },
        },
        y: {
          min: 0,
          max: 1,
          title: { display: true, text: "Risk score (0–1)", color: "#898781" },
          grid: { color: "#2c2c2a" },
          ticks: { color: "#898781" },
        },
      },
    },
  });
}

function renderRiskTable() {
  const wrap = document.getElementById("riskTableWrap");
  if (!lastRiskSeries) return;
  const rows = lastRiskSeries.depth_m
    .map((d, i) => {
      const risk = lastRiskSeries.ewma_risk[i];
      const stuck = lastRiskSeries.stuck_rt[i];
      return `<tr><td>${fmt(d, 1)}</td><td>${risk === null ? "—" : fmt(risk)}</td><td>${stuck === null ? "—" : stuck}</td></tr>`;
    })
    .join("");
  wrap.innerHTML = `<table class="data-table"><thead><tr><th>Depth (m)</th><th>Risk score (EWMA)</th><th>Actual STUCK_RT</th></tr></thead><tbody>${rows}</tbody></table>`;
}

document.getElementById("toggleTableBtn").addEventListener("click", () => {
  const canvasWrap = document.querySelector(".chart-wrap");
  const tableWrap = document.getElementById("riskTableWrap");
  const showingTable = tableWrap.style.display !== "none";
  if (showingTable) {
    tableWrap.style.display = "none";
    canvasWrap.style.display = "block";
    document.getElementById("toggleTableBtn").textContent = "View as table";
  } else {
    renderRiskTable();
    tableWrap.style.display = "block";
    canvasWrap.style.display = "none";
    document.getElementById("toggleTableBtn").textContent = "View as chart";
  }
});

// ---------------------------------------------------------------------
// Depth inspection: alert box + telemetry table
// ---------------------------------------------------------------------

const RAW_TELEMETRY_LABELS = {
  "Weight on Bit kkgf": "Weight on Bit (kkgf)",
  "Average Surface Torque kN.m": "Surface Torque (kN·m)",
  "Rate of Penetration m/h": "Rate of Penetration (m/h)",
  "Average Rotary Speed rpm": "Rotary Speed (rpm)",
  "Mud Density In g/cm3": "Mud Density In (g/cm³)",
  "Mud Density Out g/cm3": "Mud Density Out (g/cm³)",
  "Average Standpipe Pressure kPa": "Standpipe Pressure (kPa)",
  "Corrected Total Hookload kkgf": "Total Hookload (kkgf)",
  "Flow Pumps L/min": "Flow Pumps (L/min)",
  "MWD Shock Risk unitless": "MWD Shock Risk",
};

async function loadDepth(depth) {
  const [alert, telemetry] = await Promise.all([
    fetchJSON(`/api/alert?depth=${depth}`),
    fetchJSON(`/api/telemetry?depth=${depth}`),
  ]);
  renderAlert(alert);
  renderTelemetry(telemetry);
}

function renderAlert(decision) {
  const box = document.getElementById("alertBox");
  box.className = "alert-box " + levelClass(decision.risk_level);

  document.getElementById("alertTitle").textContent =
    `Supervisor Agent — ${fmt(decision.depth_m, 1)}m — ${levelLabel(decision.risk_level)}`;

  const freshnessText = {
    fresh: "live sensor data",
    stale: "carried-forward trend, no fresh data",
    unmonitored: "no data at this depth",
  }[decision.data_freshness] || decision.data_freshness;
  document.getElementById("alertFreshness").textContent = freshnessText;

  document.getElementById("alertExplain").textContent = decision.explanation;

  const actionsEl = document.getElementById("alertActions");
  actionsEl.innerHTML = decision.recommended_actions.map((a) => `<li>${a}</li>`).join("");

  const cov = decision.sensor_coverage;
  document.getElementById("alertMeta").textContent =
    `Sensor coverage: ${cov.fusion_rules_active}/${cov.fusion_rules_total} fusion rules active ` +
    (decision.primary_driver ? `· Primary driver: ${decision.primary_driver}` : "") +
    ` · ${decision.generated_by}`;
}

function renderTelemetry(t) {
  const tbody = document.querySelector("#telemetryTable tbody");
  const rows = Object.entries(RAW_TELEMETRY_LABELS).map(([key, label]) => {
    const val = t.telemetry[key];
    return `<tr><td>${label}</td><td>${val === null ? "—" : fmt(val, 2)}</td></tr>`;
  });
  rows.push(
    `<tr><td>Fused risk (instantaneous)</td><td>${t.risk.fused_risk === null ? "—" : fmt(t.risk.fused_risk)}</td></tr>`,
    `<tr><td>EWMA risk (trend)</td><td>${t.risk.ewma_risk === null ? "—" : fmt(t.risk.ewma_risk)}</td></tr>`,
    `<tr><td>Rules fired</td><td>${t.risk.rules_fired || "—"}</td></tr>`,
    `<tr><td>Historical STUCK_RT (ground truth, retrospective only)</td><td>${t.stuck_rt_ground_truth === null ? "—" : t.stuck_rt_ground_truth}</td></tr>`
  );
  tbody.innerHTML = rows.join("");
}

function setDepth(depth) {
  const clamped = Math.max(depthMin, Math.min(depth, depthMax));
  document.getElementById("depthInput").value = clamped.toFixed(1);
  loadDepth(clamped).catch((err) => console.error(err));
}

document.getElementById("depthGoBtn").addEventListener("click", () => {
  const v = parseFloat(document.getElementById("depthInput").value);
  if (!Number.isNaN(v)) setDepth(v);
});
document.getElementById("depthInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") document.getElementById("depthGoBtn").click();
});

// ---------------------------------------------------------------------
// Copilot chat
// ---------------------------------------------------------------------

function appendChatMessage(text, who) {
  const log = document.getElementById("chatLog");
  const el = document.createElement("div");
  el.className = "chat-msg " + who;
  el.textContent = text;
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
}

async function sendCopilotMessage() {
  const input = document.getElementById("chatInput");
  const message = input.value.trim();
  if (!message) return;
  appendChatMessage(message, "user");
  input.value = "";

  try {
    const res = await fetchJSON("/api/copilot", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });
    appendChatMessage(res.reply, "agent");
    if (res.depth_m !== null && res.depth_m !== undefined) {
      setDepth(res.depth_m);
    }
  } catch (err) {
    appendChatMessage("Sorry, the copilot request failed: " + err.message, "agent");
  }
}

document.getElementById("chatSendBtn").addEventListener("click", sendCopilotMessage);
document.getElementById("chatInput").addEventListener("keydown", (e) => {
  if (e.key === "Enter") sendCopilotMessage();
});

// ---------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------

async function boot() {
  try {
    const ov = await loadOverview();
    await loadRiskChart();
    await loadDepth(ov.current_depth_m);
  } catch (err) {
    console.error(err);
    document.getElementById("alertExplain").textContent =
      "Could not reach the TENETDrill API at " + API_BASE + " -- " + err.message;
  }
}

boot();
