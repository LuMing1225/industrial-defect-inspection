/* ============================================================
   工业缺陷干扰项过滤系统 - 前端逻辑
   纯原生 JS，无外部依赖
   ============================================================ */

// 后端地址（前端由 http://localhost:8522 提供，跨域访问 8562）
const API = "http://localhost:8562";

// 全局缓存当前选中的文件
let currentFile = null;

// -------------------- 工具函数 --------------------
function $(id) { return document.getElementById(id); }

function setStatus(elem, ok, text) {
  elem.className = "dot " + (ok ? "online" : "offline");
  if (text) elem.parentElement.querySelector("span:last-child").textContent = text;
}

function log(msg) {
  $("footer-msg").textContent = msg;
  console.log("[frontend]", msg);
}

// -------------------- 1. 状态轮询（每 5 秒） --------------------
async function pollHealth() {
  try {
    const r = await fetch(API + "/api/health", { timeout: 3000 });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    setStatus($("dot-backend"), true, "后端连接");
    setStatus($("dot-ollama"), data.ollama_available, data.ollama_available ? "Ollama 在线" : "Ollama 离线");
    $("gpu-text").textContent = "GPU: " + data.gpu_info;
    $("dot-gpu").className = "dot " + (data.gpu_info.includes("CPU") ? "warn" : "online");
    log("后端健康检查正常");
  } catch (e) {
    setStatus($("dot-backend"), false, "后端离线");
    setStatus($("dot-ollama"), false, "Ollama 未知");
    $("gpu-text").textContent = "GPU: 后端未连接";
    $("dot-gpu").className = "dot offline";
    log("无法连接后端，请先运行 run_backend.bat");
  }
}

// -------------------- 2. 上传 --------------------
const dropZone = $("drop-zone");
const fileInput = $("file-input");

dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragover", "dragenter"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.add("dragover"); })
);
["dragleave", "drop"].forEach(ev =>
  dropZone.addEventListener(ev, (e) => { e.preventDefault(); dropZone.classList.remove("dragover"); })
);
dropZone.addEventListener("drop", (e) => handleFile(e.dataTransfer.files[0]));

function handleFile(file) {
  if (!file || !file.type.startsWith("image/")) {
    alert("请选择图片文件");
    return;
  }
  currentFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    $("preview-img").src = e.target.result;
    $("preview-img").style.display = "block";
    $("result-original").src = e.target.result;
    $("btn-detect").disabled = false;
    log("已选择：" + file.name);
  };
  reader.readAsDataURL(file);
}

// -------------------- 3. 检测 --------------------
$("btn-detect").addEventListener("click", async () => {
  if (!currentFile) return;
  const btn = $("btn-detect");
  btn.disabled = true; btn.textContent = "检测中…";
  log("正在上传并检测…");

  const fd = new FormData();
  fd.append("file", currentFile);

  try {
    const r = await fetch(API + "/api/detect", { method: "POST", body: fd });
    if (!r.ok) throw new Error("HTTP " + r.status);
    const data = await r.json();
    renderResult(data);
    log(`检测完成：候选${data.total_candidates} / 过滤${data.filtered_by_feature} / 缺陷${data.final_defects.filter(d=>d.final_label==="缺陷").length}`);
  } catch (e) {
    alert("检测失败：" + e.message);
    log("检测失败：" + e.message);
  } finally {
    btn.disabled = false; btn.textContent = "开始检测";
  }
});

function renderResult(data) {
  // 标注图
  if (data.visualization) {
    $("result-annotated").src = data.visualization;
    $("empty-hint").style.display = "none";
  }
  // 缺陷卡片
  const list = $("defect-list");
  $("defect-count").textContent = data.final_defects.length;
  if (data.final_defects.length === 0) {
    list.innerHTML = '<p class="muted">未检测到候选区域</p>';
  } else {
    list.innerHTML = data.final_defects.map((d, i) => {
      const cls = d.final_label === "缺陷" ? "defect" : "interference";
      const f = d.features || {};
      const v = d.vlm_result || {};
      return `<div class="defect-card ${cls}">
        <div class="dc-title">#${i+1} ${d.final_label} (score=${d.score})</div>
        <div class="dc-row"><span>bbox</span><b>[${d.bbox.join(", ")}]</b></div>
        <div class="dc-row"><span>圆度</span><b>${(f.roundness||0).toFixed(3)}</b></div>
        <div class="dc-row"><span>长宽比</span><b>${(f.aspect_ratio||0).toFixed(2)}</b></div>
        <div class="dc-row"><span>灰度方差</span><b>${(f.var_gray||0).toFixed(3)}</b></div>
        <div class="dc-row"><span>LBP对比</span><b>${(f.contrast_lbp||0).toFixed(3)}</b></div>
        <div class="dc-row"><span>SAM连通度</span><b>${(d.sam_mask_connectivity||0).toFixed(3)}</b></div>
        <div class="dc-vlm">
          <b>VLM</b>：${v.label || "-"} (${(v.confidence||0).toFixed(2)})<br/>
          ${v.reason || ""}
        </div>
      </div>`;
    }).join("");
  }

  // 时序图（以总时长为100%）
  const t = data.timing_ms;
  const total = t.total || 1;
  setBar("bar-feature", "val-feature", t.feature_filter, total);
  setBar("bar-sam", "val-sam", t.sam_combine, total);
  setBar("bar-vlm", "val-vlm", t.vlm_validate, total);

  // 性能指标
  $("perf-total").textContent = t.total + " ms";
  const filterOk = t.feature_filter + t.sam_combine <= 80;
  const vlmOk = t.vlm_validate <= 450;
  $("perf-filter").textContent = (t.feature_filter + t.sam_combine).toFixed(1) + " ms " + (filterOk ? "✓" : "✗");
  $("perf-vlm").textContent = t.vlm_validate.toFixed(1) + " ms " + (vlmOk ? "✓" : "✗");
  $("perf-status").textContent = (filterOk && vlmOk) ? "达标" : "超标";
  $("perf-status").style.color = (filterOk && vlmOk) ? "var(--green)" : "var(--red)";

  // 统计
  refreshStats();
}

function setBar(barId, valId, ms, total) {
  $(barId).style.width = Math.min(100, (ms / total) * 100) + "%";
  $(valId).textContent = ms.toFixed(1) + " ms";
}

// -------------------- 4. 原图/标注切换 --------------------
$("tab-annotated").addEventListener("click", () => switchView(true));
$("tab-original").addEventListener("click", () => switchView(false));
function switchView(showAnnotated) {
  $("tab-annotated").classList.toggle("active", showAnnotated);
  $("tab-original").classList.toggle("active", !showAnnotated);
  $("result-annotated").style.display = showAnnotated ? "block" : "none";
  $("result-original").style.display = showAnnotated ? "none" : "block";
}

// -------------------- 5. 参数面板 --------------------
const sliders = [
  ["s-roundness", "v-roundness", "roundness_threshold", "filter"],
  ["s-vargray", "v-vargray", "var_gray_threshold", "filter"],
  ["s-lbp", "v-lbp", "lbp_contrast_threshold", "filter"],
  ["s-score", "v-score", "score_threshold", "filter"],
  ["s-w1", "v-w1", "w1_roundness", "filter.weights"],
  ["s-w2", "v-w2", "w2_var_gray", "filter.weights"],
  ["s-w3", "v-w3", "w3_contrast_lbp", "filter.weights"],
];

sliders.forEach(([sid, vid, key, group]) => {
  $(sid).addEventListener("input", () => {
    $(vid).textContent = parseFloat($(sid).value).toFixed(2);
  });
  $(sid).addEventListener("change", async () => {
    const val = parseFloat($(sid).value);
    // 组装请求体
    let body;
    if (group === "filter.weights") {
      body = { filter: { weights: { [key]: val } } };
    } else {
      body = { filter: { [key]: val } };
    }
    try {
      await fetch(API + "/api/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      log("参数已更新：" + key + " = " + val);
    } catch (e) {
      log("参数更新失败：" + e.message);
    }
  });
});

// 启动时拉一次当前配置回显
async function loadConfig() {
  try {
    const r = await fetch(API + "/api/config");
    const cfg = await r.json();
    const f = cfg.filter || {};
    if (f.roundness_threshold !== undefined) setSlider("s-roundness", "v-roundness", f.roundness_threshold);
    if (f.var_gray_threshold !== undefined) setSlider("s-vargray", "v-vargray", f.var_gray_threshold);
    if (f.lbp_contrast_threshold !== undefined) setSlider("s-lbp", "v-lbp", f.lbp_contrast_threshold);
    if (f.score_threshold !== undefined) setSlider("s-score", "v-score", f.score_threshold);
    const w = f.weights || {};
    if (w.w1_roundness !== undefined) setSlider("s-w1", "v-w1", w.w1_roundness);
    if (w.w2_var_gray !== undefined) setSlider("s-w2", "v-w2", w.w2_var_gray);
    if (w.w3_contrast_lbp !== undefined) setSlider("s-w3", "v-w3", w.w3_contrast_lbp);
  } catch (e) { /* 后端未启动时静默 */ }
}
function setSlider(sid, vid, val) {
  $(sid).value = val;
  $(vid).textContent = parseFloat(val).toFixed(2);
}

// -------------------- 6. 统计面板 --------------------
async function refreshStats() {
  try {
    const r = await fetch(API + "/api/stats");
    const s = await r.json();
    $("st-candidates").textContent = s.total_candidates;
    $("st-filtered").textContent = s.filtered_by_feature;
    $("st-vlm").textContent = s.vlm_checked;
    $("st-defects").textContent = s.final_defects;
    $("st-interf").textContent = s.total_interference;
    $("st-rate").textContent = (s.interference_exclusion_rate * 100).toFixed(1) + "%";
  } catch (e) { /* 静默 */ }
}

// -------------------- 7. 压测 --------------------
$("btn-benchmark").addEventListener("click", async () => {
  log("开始压测…");
  try {
    const r = await fetch(API + "/api/benchmark", { method: "POST" });
    const b = await r.json();
    $("bench-result").textContent =
      `帧数: ${b.frames}  平均: ${b.avg_ms} ms  P95: ${b.p95_ms} ms\n` +
      `吞吐: ${b.throughput_fps} FPS  目标≤${b.target_latency_ms}ms  ${b.pass ? "✓ 达标" : "✗ 超标"}`;
    log("压测完成");
  } catch (e) {
    $("bench-result").textContent = "压测失败：" + e.message;
  }
});

// -------------------- 启动 --------------------
pollHealth();
refreshStats();
loadConfig();
setInterval(pollHealth, 5000);
