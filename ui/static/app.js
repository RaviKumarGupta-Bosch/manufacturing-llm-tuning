/**
 * app.js — Manufacturing LLM Comparison UI
 * Handles model loading, streaming SSE, markdown rendering, history, and tutorial.
 */

// ── State ─────────────────────────────────────────────────────────────────────
const state = {
  leftController:  null,
  rightController: null,
  isRunning: false,
  history: [],
  samplePrompts: []
};

// ── DOM refs ──────────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadModels();
  loadSamplePrompts();
  setupPromptInput();
  setupTemperature();
});

// ── Health Check ──────────────────────────────────────────────────────────────
async function checkHealth() {
  const badge = $('health-badge');
  try {
    const resp = await fetch('/api/health');
    const data = await resp.json();
    if (data.ollama === 'offline') {
      badge.textContent = '⛔ Ollama Offline';
      badge.className = 'badge badge-error';
      showSetupBanner(true);
    } else if (data.setup_needed) {
      badge.textContent = '⚠ Models Missing';
      badge.className = 'badge badge-warn';
      showSetupBanner(true);
    } else {
      badge.textContent = '✓ Ollama Online';
      badge.className = 'badge badge-ok';
      showSetupBanner(false);
    }
  } catch {
    badge.textContent = '⛔ Offline';
    badge.className = 'badge badge-error';
  }
}

function showSetupBanner(show) {
  $('setup-banner').classList.toggle('hidden', !show);
}

// ── Model Loading ─────────────────────────────────────────────────────────────
async function loadModels() {
  try {
    const resp = await fetch('/api/models');
    const data = await resp.json();
    const models = data.models || [];

    const leftSel  = $('model-left');
    const rightSel = $('model-right');
    leftSel.innerHTML  = '';
    rightSel.innerHTML = '';

    if (models.length === 0) {
      const opt = '<option value="">No models found</option>';
      leftSel.innerHTML = opt;
      rightSel.innerHTML = opt;
      return;
    }

    models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.name;
      leftSel.appendChild(opt.cloneNode(true));
      rightSel.appendChild(opt);
    });

    // Pre-select our models
    selectModelOption(leftSel,  ['mfg-base',   'llama3.2', models[0]?.name]);
    selectModelOption(rightSel, ['mfg-expert', 'llama3.1', models[0]?.name]);

    updatePanelNames();
    leftSel.addEventListener('change', updatePanelNames);
    rightSel.addEventListener('change', updatePanelNames);

  } catch {
    // Offline — set fallback defaults
    ['model-left', 'model-right'].forEach(id => {
      $(id).innerHTML = '<option value="mfg-base">mfg-base</option><option value="mfg-expert">mfg-expert</option>';
    });
    $('model-right').value = 'mfg-expert';
    updatePanelNames();
  }
}

function selectModelOption(select, preferred) {
  for (const p of preferred) {
    if (!p) continue;
    const opt = [...select.options].find(o => o.value.startsWith(p.split(':')[0]));
    if (opt) { select.value = opt.value; return; }
  }
}

function updatePanelNames() {
  $('panel-left-name').textContent  = $('model-left').value  || '—';
  $('panel-right-name').textContent = $('model-right').value || '—';
}

// ── Sample Prompts ────────────────────────────────────────────────────────────
async function loadSamplePrompts() {
  try {
    const resp = await fetch('/api/sample-prompts');
    const data = await resp.json();
    state.samplePrompts = data.prompts || [];
    renderSamplePicker();
  } catch {}
}

function renderSamplePicker() {
  const container = $('sample-picker');
  container.innerHTML = state.samplePrompts.map(p => {
    const safePrompt = escapeHtml(p.prompt);
    return `
    <div class="sample-item" data-prompt="${safePrompt}" onclick="useSample(this.dataset.prompt)">
      <div class="sample-cat">${escapeHtml(p.category)}</div>
      <div class="sample-text">${escapeHtml(p.prompt.slice(0, 90))}${p.prompt.length > 90 ? '…' : ''}</div>
    </div>`;
  }).join('');
}

function toggleSamplePicker() {
  $('sample-picker').classList.toggle('hidden');
}

function useSample(prompt) {
  const input = $('prompt-input');
  input.value = prompt;
  $('sample-picker').classList.add('hidden');
  updateCharCount();
  input.focus();
  input.classList.add('prompt-flash');
  setTimeout(() => input.classList.remove('prompt-flash'), 800);
}

// ── Prompt Input ──────────────────────────────────────────────────────────────
function setupPromptInput() {
  const input = $('prompt-input');
  input.addEventListener('input', updateCharCount);
  input.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      if (!state.isRunning) sendPrompt();
    }
  });
}

function updateCharCount() {
  const len = $('prompt-input').value.length;
  $('char-count').textContent = `${len} / 2000`;
  $('char-count').style.color = len > 1800 ? '#f59e0b' : '';
}

function setupTemperature() {
  const slider = $('temperature');
  slider.addEventListener('input', () => {
    $('temp-val').textContent = parseFloat(slider.value).toFixed(2);
  });
}

// ── Main Send ─────────────────────────────────────────────────────────────────
async function sendPrompt() {
  const prompt = $('prompt-input').value.trim();
  if (!prompt) {
    $('prompt-input').focus();
    return;
  }

  const modelLeft  = $('model-left').value;
  const modelRight = $('model-right').value;
  const temperature = parseFloat($('temperature').value);

  if (!modelLeft || !modelRight) {
    alert('Please select models for both panels.');
    return;
  }

  // Add to history
  addToHistory(prompt);

  // Reset panels
  setPanelContent('left',  '');
  setPanelContent('right', '');
  setMetrics('left',  null);
  setMetrics('right', null);
  setPanelStreaming('left',  true);
  setPanelStreaming('right', true);

  setRunning(true);

  // Launch both streams in parallel
  await Promise.allSettled([
    streamPanel('left',  prompt, modelLeft,  temperature),
    streamPanel('right', prompt, modelRight, temperature)
  ]);

  setRunning(false);
}

// ── Streaming ─────────────────────────────────────────────────────────────────
async function streamPanel(side, prompt, model, temperature) {
  const controller = new AbortController();
  if (side === 'left')  state.leftController  = controller;
  if (side === 'right') state.rightController = controller;

  const bodyEl = $(`panel-${side}-body`);
  bodyEl.classList.add('cursor-blink');
  setStatus(side, true);

  let accumulated = '';

  try {
    const resp = await fetch(`/api/stream/${side}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prompt, model, temperature, max_tokens: 1024 }),
      signal: controller.signal
    });

    if (!resp.ok) {
      const errData = await resp.json().catch(() => ({}));
      throw new Error(errData.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep incomplete line

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const jsonStr = line.slice(6).trim();
        if (!jsonStr) continue;

        let chunk;
        try { chunk = JSON.parse(jsonStr); } catch { continue; }

        if (chunk.error) {
          setPanelContent(side, `❌ Error: ${chunk.error}`);
          break;
        }

        if (chunk.token) {
          accumulated += chunk.token;
          renderMarkdown(bodyEl, accumulated);
        }

        if (chunk.done) {
          setMetrics(side, {
            elapsed_ms:     chunk.elapsed_ms,
            tokens_per_sec: chunk.tokens_per_sec,
            tokens:         chunk.tokens
          });
        }
      }
    }

  } catch (err) {
    if (err.name !== 'AbortError') {
      setPanelContent(side, `❌ ${err.message}`);
    }
  } finally {
    bodyEl.classList.remove('cursor-blink');
    setPanelStreaming(side, false);
    setStatus(side, false);
  }
}

// ── Markdown Renderer ─────────────────────────────────────────────────────────
function renderMarkdown(el, raw) {
  let html = escapeHtml(raw);

  // Code blocks (must come before inline)
  html = html.replace(/```[\s\S]*?```/g, match => {
    const code = match.slice(3, -3).replace(/^[^\n]*\n/, '');
    return `<div class="md-code-block">${code}</div>`;
  });

  // Headings
  html = html.replace(/^#### (.+)$/gm, '<div class="md-h3">$1</div>');
  html = html.replace(/^### (.+)$/gm,  '<div class="md-h3">$1</div>');
  html = html.replace(/^## (.+)$/gm,   '<div class="md-h2">$1</div>');
  html = html.replace(/^# (.+)$/gm,    '<div class="md-h1">$1</div>');

  // Bold, italic, inline code
  html = html.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*([^*]+)\*/g,     '<em>$1</em>');
  html = html.replace(/`([^`]+)`/g,       '<span class="md-code-inline">$1</span>');

  // Tables (simple)
  html = renderTables(html);

  // Bullet lists
  html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul class="md-list">$1</ul>');

  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li>$1</li>');

  // Blockquote
  html = html.replace(/^&gt; (.+)$/gm, '<div class="md-blockquote">$1</div>');

  // Horizontal rule
  html = html.replace(/^---+$/gm, '<hr style="border-color:#334155;margin:12px 0;">');

  // Newlines
  html = html.replace(/\n\n/g, '<br><br>');
  html = html.replace(/\n/g, '<br>');

  el.innerHTML = html;
  el.scrollTop = el.scrollHeight;
}

function renderTables(html) {
  // Match markdown table blocks
  return html.replace(/((\|[^\n]+\|\n)+)/g, tableBlock => {
    const rows = tableBlock.trim().split('\n');
    if (rows.length < 2) return tableBlock;

    let result = '<table class="md-table"><thead><tr>';
    const headers = rows[0].split('|').filter(Boolean);
    headers.forEach(h => { result += `<th>${h.trim()}</th>`; });
    result += '</tr></thead><tbody>';

    // Skip separator row
    for (let i = 2; i < rows.length; i++) {
      const cells = rows[i].split('|').filter(Boolean);
      if (!cells.length) continue;
      result += '<tr>';
      cells.forEach(c => { result += `<td>${c.trim()}</td>`; });
      result += '</tr>';
    }
    result += '</tbody></table>';
    return result;
  });
}

// ── Panel Helpers ─────────────────────────────────────────────────────────────
function setPanelContent(side, text) {
  const el = $(`panel-${side}-body`);
  if (text === '') {
    el.innerHTML = '';
  } else {
    renderMarkdown(el, text);
  }
}

function setPanelStreaming(side, active) {
  const panel = $(`panel-${side}`);
  panel.classList.toggle('streaming', active);
}

function setStatus(side, active) {
  const el = $(`${side}-status`);
  el.className = `panel-status${active ? ` active-${side}` : ''}`;
}

function setMetrics(side, data) {
  if (!data) {
    $(`${side}-time`).textContent = '—';
    $(`${side}-tps`).textContent  = '—';
    return;
  }
  const secs = (data.elapsed_ms / 1000).toFixed(1);
  $(`${side}-time`).textContent = `${secs}s`;
  $(`${side}-tps`).textContent  = `${data.tokens_per_sec} t/s`;
}

// ── Stop / Clear ──────────────────────────────────────────────────────────────
function stopAll() {
  if (state.leftController)  { state.leftController.abort();  state.leftController  = null; }
  if (state.rightController) { state.rightController.abort(); state.rightController = null; }
  setRunning(false);
  setPanelStreaming('left',  false);
  setPanelStreaming('right', false);
  setStatus('left',  false);
  setStatus('right', false);
  [$('panel-left-body'), $('panel-right-body')].forEach(el => el.classList.remove('cursor-blink'));
}

function clearAll() {
  stopAll();
  $('prompt-input').value = '';
  updateCharCount();
  setPanelContent('left',  '');
  setPanelContent('right', '');
  setMetrics('left',  null);
  setMetrics('right', null);
  $('panel-left-body').innerHTML  = emptyState('left');
  $('panel-right-body').innerHTML = emptyState('right');
}

function emptyState(side) {
  if (side === 'left') return `<div class="empty-state"><span class="empty-icon">🤖</span><p>Base model response will appear here</p><p class="empty-hint">No manufacturing context. General-purpose assistant.</p></div>`;
  return `<div class="empty-state"><span class="empty-icon">🏭</span><p>Tuned model response will appear here</p><p class="empty-hint">Manufacturing-domain expert. Knows OEE, FMEA, ISO standards.</p></div>`;
}

function setRunning(running) {
  state.isRunning = running;
  $('send-btn').disabled = running;
  $('stop-btn').classList.toggle('hidden', !running);
}

// ── History ───────────────────────────────────────────────────────────────────
function addToHistory(prompt) {
  state.history.unshift(prompt);
  if (state.history.length > 20) state.history.pop();
  renderHistory();
}

function renderHistory() {
  const container = $('history-list');
  if (state.history.length === 0) {
    container.innerHTML = '<div class="history-empty">No prompts sent yet.</div>';
    return;
  }
  container.innerHTML = state.history.map(p => `
    <div class="history-chip" title="${escapeHtml(p)}" onclick="reloadPrompt(${JSON.stringify(p)})">
      ${escapeHtml(p.slice(0, 60))}${p.length > 60 ? '…' : ''}
    </div>
  `).join('');
}

function reloadPrompt(prompt) {
  $('prompt-input').value = prompt;
  updateCharCount();
  $('prompt-input').focus();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function clearHistory() {
  state.history = [];
  renderHistory();
}

// ── Copy Panel ────────────────────────────────────────────────────────────────
async function copyPanel(side) {
  const el = $(`panel-${side}-body`);
  const text = el.innerText || el.textContent;
  try {
    await navigator.clipboard.writeText(text);
    const btn = el.closest('.response-panel').querySelector('.btn-copy');
    const orig = btn.textContent;
    btn.textContent = '✓';
    setTimeout(() => { btn.textContent = orig; }, 1500);
  } catch {}
}

// ── Utils ─────────────────────────────────────────────────────────────────────
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

// ════════════════════════════════════════════════════════════════════════════════
// TUTORIAL
// ════════════════════════════════════════════════════════════════════════════════

let currentStep = 1;
const TOTAL_STEPS = 9;

// ── Tab Switching ─────────────────────────────────────────────────────────────
function showTab(name) {
  ['compare', 'tutorial'].forEach(v => {
    const view = $(`view-${v}`);
    const btn  = $(`tab-${v}-btn`);
    if (view) view.classList.toggle('hidden', v !== name);
    if (btn)  btn.classList.toggle('active',  v === name);
  });
  if (name === 'tutorial') {
    // Trigger score bar animation when tutorial tab is opened
    setTimeout(animateScoreBars, 200);
  }
}

// ── Step Navigation ───────────────────────────────────────────────────────────
function gotoStep(n) {
  if (n < 1 || n > TOTAL_STEPS) return;

  // Deactivate current
  const prevEl  = $(`tut-step-${currentStep}`);
  const prevNav = document.querySelector(`.tut-nav-item[data-step="${currentStep}"]`);
  if (prevEl)  prevEl.classList.remove('active');
  if (prevNav) { prevNav.classList.remove('active'); prevNav.classList.add('done'); }

  // Activate new
  const nextEl  = $(`tut-step-${n}`);
  const nextNav = document.querySelector(`.tut-nav-item[data-step="${n}"]`);
  if (nextEl)  nextEl.classList.add('active');
  if (nextNav) { nextNav.classList.add('active'); nextNav.classList.remove('done'); }

  currentStep = n;
  updateTutProgress();

  // Scroll content to top
  const content = document.querySelector('.tut-content');
  if (content) content.scrollTop = 0;

  // Animate bars if we just landed on step 7
  if (n === 7) setTimeout(animateScoreBars, 300);
}

function stepNav(delta) {
  gotoStep(currentStep + delta);
}

function updateTutProgress() {
  const pct  = (currentStep / TOTAL_STEPS) * 100;
  const fill = $('tut-progress-fill');
  const text = $('tut-progress-text');
  if (fill) fill.style.width = `${pct}%`;
  if (text) text.textContent = `Step ${currentStep} of ${TOTAL_STEPS}`;

  const prevBtn = $('tut-prev-btn');
  const nextBtn = $('tut-next-btn');
  if (prevBtn) prevBtn.disabled = (currentStep === 1);
  if (nextBtn) {
    if (currentStep === TOTAL_STEPS) {
      nextBtn.textContent = '🎉 Start Comparing';
      nextBtn.onclick = () => showTab('compare');
    } else {
      nextBtn.textContent = 'Next Step →';
      nextBtn.onclick = () => stepNav(1);
    }
  }
}

// ── Load prompt into comparator and switch to compare tab ────────────────────
function tryPrompt(text) {
  const input = $('prompt-input');
  if (input) {
    input.value = text;
    updateCharCount();
  }
  showTab('compare');
  window.scrollTo({ top: 0, behavior: 'smooth' });
  setTimeout(() => {
    const btn = $('send-btn');
    if (btn) btn.focus();
  }, 300);
}

// ── Animate score bar fills (triggered on step 7 visibility) ─────────────────
function animateScoreBars() {
  document.querySelectorAll('.tut-bar-fill').forEach(bar => {
    const target = bar.dataset.width;
    if (!target) return;
    bar.style.width = '0%';
    // Small delay to let CSS transition pick it up
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        bar.style.width = `${target}%`;
      });
    });
  });
}
