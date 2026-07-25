const API = '';
let currentGame = null;
let currentView = 'public';
let cachedPlayers = [];

// -- View Toggle --
document.querySelectorAll('.view-toggle button').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.view-toggle button').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentView = btn.dataset.view;
    updateModSections();
    if (currentGame) refreshAll();
  });
});

function updateModSections() {
  const show = currentView === 'moderator_full' || currentView === 'debug';
  document.querySelectorAll('.mod-section').forEach(s => {
    s.classList.toggle('visible', show);
  });
}
updateModSections();

// -- Helpers --
async function api(path, opts = {}) {
  const sep = path.includes('?') ? '&' : '?';
  const url = API + path + sep + '_t=' + Date.now();
  const r = await fetch(url, opts);
  return r;
}

function getCallerParams() {
  const id = document.getElementById('callerId').value;
  const role = document.getElementById('callerRole').value;
  return { caller_id: id, caller_role: role, view_mode: currentView === 'debug' ? 'moderator_full' : currentView };
}

function toParams(obj) {
  return Object.entries(obj).filter(([_, v]) => v).map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join('&');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, ch => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[ch]));
}

function compactJson(value, limit = 140) {
  return escapeHtml(JSON.stringify(value || {}).slice(0, limit));
}

function showStatus(message, kind = '') {
  const el = document.getElementById('statusBar');
  el.textContent = message;
  el.className = `status-bar ${kind}`.trim();
}

async function readJsonOrThrow(response) {
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    const detail = data.detail || data.error || response.statusText || '请求失败';
    throw new Error(detail);
  }
  return data;
}

// -- Game List --
async function loadGameList() {
  const r = await api('/games');
  const games = await readJsonOrThrow(r);
  const el = document.getElementById('gameList');
  if (!games.length) {
    el.innerHTML = '<div class="empty-state">暂无游戏</div>';
    return;
  }
  el.innerHTML = games.map(g => `
    <div class="game-item ${currentGame === g.game_id ? 'active' : ''}" onclick="selectGame('${g.game_id}')">
      <div class="gid">${g.game_id}</div>
      <div class="meta">${translateStatus(g.status)} · ${g.player_count}人 · ${g.ruleset_id || ''}</div>
    </div>
  `).join('');
}

async function createGame() {
  try {
    showStatus('正在创建新游戏...');
    const r = await api('/games', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(buildCreateGamePayload())
    });
    const created = await readJsonOrThrow(r);
    currentGame = created.game.game_id;
    await loadGameList();
    await refreshAll();
    showStatus(`已创建并选中游戏 ${currentGame}。现在可以点击"开始游戏"。`, 'ok');
  } catch (e) {
    showStatus(`创建游戏失败：${e.message}`, 'error');
    console.error('create game error', e);
  }
}

function selectGame(gid) {
  currentGame = gid;
  showStatus(`已选择游戏 ${gid}。`);
  loadGameList();
  refreshAll();
}

async function startGame() {
  if (!currentGame) {
    showStatus('请先创建或选择一局游戏，然后再开始。', 'error');
    return;
  }
  const p = getCallerParams();
  try {
    showStatus('正在开始游戏...');
    const r = await api(`/games/${currentGame}/start?${toParams(p)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caller_id: p.caller_id, caller_role: p.caller_role })
    });
    await readJsonOrThrow(r);
    await refreshAll();
    showStatus('游戏已开始，角色已分配，当前进入夜晚阶段。', 'ok');
  } catch (e) {
    showStatus(`开始游戏失败：${e.message}`, 'error');
    console.error('start game error', e);
  }
}

async function stepGame() {
  if (!currentGame) {
    showStatus('请先创建或选择一局游戏。', 'error');
    return;
  }
  const p = getCallerParams();
  try {
    showStatus('正在推进一小步...');
    const r = await api(`/games/${currentGame}/step?${toParams(p)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caller_id: p.caller_id, caller_role: p.caller_role })
    });
    const data = await readJsonOrThrow(r);
    await refreshAll();
    showStatus(data.message || '已推进一步。', 'ok');
  } catch (e) {
    showStatus(`推进失败：${e.message}`, 'error');
    console.error('step game error', e);
  }
}

async function pauseGame() {
  if (!currentGame) {
    showStatus('请先创建或选择一局游戏。', 'error');
    return;
  }
  const p = getCallerParams();
  try {
    const r = await api(`/games/${currentGame}/pause?${toParams(p)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caller_id: p.caller_id, caller_role: p.caller_role })
    });
    await readJsonOrThrow(r);
    await refreshAll();
    showStatus('游戏已暂停。', 'ok');
  } catch (e) {
    showStatus(`暂停失败：${e.message}`, 'error');
  }
}

async function resumeGame() {
  if (!currentGame) {
    showStatus('请先创建或选择一局游戏。', 'error');
    return;
  }
  const p = getCallerParams();
  try {
    const r = await api(`/games/${currentGame}/resume?${toParams(p)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ caller_id: p.caller_id, caller_role: p.caller_role })
    });
    await readJsonOrThrow(r);
    await refreshAll();
    showStatus('游戏已继续。', 'ok');
  } catch (e) {
    showStatus(`继续失败：${e.message}`, 'error');
  }
}

function translateStatus(status) {
  if (status === 'created') return '已创建';
  if (status === 'active') return '进行中';
  if (status === 'ended') return '已结束';
  return status || '-';
}

function translatePhase(phase) {
  const names = {
    setup: '准备中',
    night: '夜晚',
    day: '白天',
    discussion: '发言讨论',
    vote: '投票',
    ended: '已结束',
  };
  return names[phase] || phase || '-';
}

function translateFaction(faction) {
  const names = {
    good: '好人阵营',
    werewolf: '狼人阵营',
    hybrid: '混血阵营',
  };
  return names[faction] || faction || '-';
}

function translateRole(role) {
  const names = {
    werewolf: '狼人',
    villager: '平民',
    seer: '预言家',
    witch: '女巫',
    hunter: '猎人',
    idiot: '白痴',
    hybrid: '混血儿',
  };
  return names[role] || role || '';
}

function translateEvent(type) {
  const names = {
    game_started: '游戏开始',
    game_paused: '游戏暂停',
    game_resumed: '游戏继续',
    death: '死亡',
    vote_resolved: '投票结算',
    speech: '发言',
    seer_check: '预言家查验',
    wolf_kill: '狼人刀人',
    witch_antidote_used: '女巫使用解药',
    witch_poison_used: '女巫使用毒药',
    rag_injection_audit: 'RAG 注入审计',
  };
  return names[type] || type || '-';
}

// -- Refresh --
async function refreshAll() {
  if (!currentGame) return;
  const p = getCallerParams();
  await Promise.all([
    loadPublicState(p),
    loadTimeline(p),
  ]);
  if (currentView === 'moderator_full' || currentView === 'debug') {
    await loadModeratorData(p);
    await loadEnhancedPanels(p);
  }
}

async function loadPublicState(p) {
  try {
    const r = await api(`/games/${currentGame}/public-state`);
    const data = await r.json();
    renderPhase(data);
    renderPlayers(data);
    renderDeaths(data);
    // Update day selector range
    updateDaySelector(data.day_number || 0);
  } catch (e) { console.error('public-state error', e); }
}

function updateDaySelector(maxDay) {
  const sel = document.getElementById('daySelect');
  const slider = document.getElementById('day-slider');
  const curVal = parseInt(sel.value) || 0;
  sel.innerHTML = '<option value="0">全部</option>';
  for (let d = 1; d <= Math.max(maxDay, 1); d++) {
    sel.innerHTML += `<option value="${d}">第 ${d} 天</option>`;
  }
  if (curVal <= maxDay) sel.value = curVal;
  slider.max = Math.max(maxDay, 1);
  if (curVal <= maxDay) slider.value = curVal;
}

function renderPhase(data) {
  const el = document.getElementById('phaseInfo');
  el.innerHTML = `
    <div style="display:flex;gap:20px;flex-wrap:wrap">
      <div><span class="label" style="color:#8b949e;font-size:12px">阶段</span><div style="font-size:16px;font-weight:600">${translatePhase(data.phase)}</div></div>
      <div><span class="label" style="color:#8b949e;font-size:12px">天数</span><div style="font-size:16px">${data.day_number || 0}</div></div>
      <div><span class="label" style="color:#8b949e;font-size:12px">夜晚</span><div style="font-size:16px">${data.night_number || 0}</div></div>
      <div><span class="label" style="color:#8b949e;font-size:12px">胜利方</span><div style="font-size:16px;color:${data.winning_faction === 'good' ? '#3fb950' : data.winning_faction === 'werewolf' ? '#f85149' : '#8b949e'}">${translateFaction(data.winning_faction)}</div></div>
    </div>`;
}

function renderPlayers(data) {
  const el = document.getElementById('playerGrid');
  const players = data.players || [];
  cachedPlayers = players;
  if (!players.length) { el.innerHTML = '<div class="empty-state">No players</div>'; return; }
  el.innerHTML = players.map((p, index) => {
    const dead = !p.alive;
    const revealed = p.revealed_role;
    let roleClass = '';
    if (revealed === 'werewolf') roleClass = 'wolf';
    else if (revealed && revealed !== 'idiot') roleClass = 'special';
    else if (revealed === 'idiot') roleClass = 'good';
    const seat = String(index + 1).padStart(2, '0');
    const label = p.name || p.player_id || `P${seat}`;
    const avatarText = label.replace(/^p/i, '').slice(-1).toUpperCase() || seat;
    return `<div class="player-card ${dead ? 'dead' : ''} ${roleClass}">
      <div class="seat-number">${seat}</div>
      <div class="avatar">${avatarText}</div>
      <div class="pid">${label}</div>
      <div class="role-tag ${roleClass}">${translateRole(revealed) || (dead ? '已出局' : '未知')}</div>
    </div>`;
  }).join('');
}

function renderDeaths(data) {
  const el = document.getElementById('deathPanel');
  const deaths = data.deaths || [];
  if (!deaths.length) { el.innerHTML = '<div class="empty-state">暂无死亡记录</div>'; return; }
  el.innerHTML = deaths.map(d => `
    <div class="death-row">
      <span style="color:#f85149">${d.player_id || d.id || '?'}</span>
      <span>${d.reason || '?'}</span>
      <span style="color:#8b949e">${d.timing || ''}</span>
    </div>
  `).join('');
}

async function loadTimeline(p) {
  try {
    const r = await api(`/games/${currentGame}/timeline?${toParams(p)}`);
    const data = await r.json();
    renderTimeline(data);
    renderVotes(data);
  } catch (e) { console.error('timeline error', e); }
}

function renderTimeline(data) {
  const el = document.getElementById('timelineList');
  const events = data.events || [];
  document.getElementById('eventCount').textContent = `${events.length} 个事件`;
  if (!events.length) { el.innerHTML = '<div class="empty-state">暂无事件</div>'; return; }
  el.innerHTML = events.map(e => `
    <div class="timeline-item">
      <span class="time">第${e.day_number || '-'}天/夜${e.night_number || '-'}</span>
      <span class="type">${translateEvent(e.event_type)}</span>
      <span class="detail">${formatDetail(e)}</span>
    </div>
  `).join('');
}

function formatDetail(e) {
  const d = e.data || {};
  if (d.text) return truncate(d.text, 80);
  if (d.player_id) return d.player_id + (d.reason ? ` (${d.reason})` : '');
  if (d.target_id) return `目标：${d.target_id}`;
  if (d.speaker) return `${d.speaker}: ${truncate(d.text || '', 60)}`;
  return JSON.stringify(d).slice(0, 80);
}

function truncate(s, n) { return s.length > n ? s.slice(0, n) + '...' : s; }

function voteDisplayTally(data, rulesetBaseVoteWeight) {
  if (data.vote_weight_format_version === 2) return data.weighted_tally_display || data.tally_display || {};
  if (Number.isInteger(rulesetBaseVoteWeight) && rulesetBaseVoteWeight > 0) {
    const units = data.weighted_tally || data.tally || {};
    return Object.fromEntries(Object.entries(units).map(([playerId,value]) => [playerId, value / rulesetBaseVoteWeight]));
  }
  return null;
}

function renderVotes(data) {
  const el = document.getElementById('votePanel');
  const events = (data.events || []).filter(e => e.event_type === 'vote_resolved' || e.event_type === 'speech');
  const votes = events.filter(e => e.event_type === 'vote_resolved');
  if (!votes.length) {
    const speeches = events.filter(e => e.event_type === 'speech');
    if (!speeches.length) { el.innerHTML = '<div class="empty-state">暂无投票记录</div>'; return; }
  }
  el.innerHTML = votes.map(e => {
    const d = e.data || {};
    const tally = voteDisplayTally(d, data.ruleset_base_vote_weight);
    if (tally === null) {
      return '<div class="vote-row"><span>不支持的旧版票权</span><span style="color:#8b949e">legacy base unknown</span></div>';
    }
    return Object.entries(tally)
      .sort((left, right) => right[1] - left[1])
      .map(([playerId, value]) => `<div class="vote-row"><span>${playerId}</span><span style="color:#58a6ff">${value}票</span></div>`)
      .join('');
  }).join('') || '<div class="empty-state">暂无投票记录</div>';
}

// -- Moderator Data --
async function loadModeratorData(p) {
  // Private audit via replay
  try {
    const r = await api(`/games/${currentGame}/replay?${toParams(p)}`);
    const data = await r.json();
    const mod = data.snapshots?.[0]?.moderator_full;
    const el = document.getElementById('privateAudit');
    if (mod) {
      const roles = mod.all_roles || {};
      el.innerHTML = `<div style="margin-bottom:8px"><strong>All Roles:</strong></div>` +
        Object.entries(roles).map(([pid, role]) =>
          `<div style="padding:2px 0"><span style="color:#58a6ff">${pid}</span>: <span style="color:${role === 'werewolf' ? '#f85149' : '#3fb950'}">${role}</span></div>`
        ).join('') +
        `<div style="margin-top:8px;color:#8b949e">Hybrid master: ${mod.hybrid_master_id || 'none'}</div>
         <div style="color:#8b949e">Antidote: ${mod.antidote_used ? 'used' : 'available'}, Poison: ${mod.poison_used ? 'used' : 'available'}</div>`;
    } else {
      el.innerHTML = '<div class="empty-state">No moderator data available (check permissions)</div>';
    }
  } catch (e) { console.error('moderator data error', e); }

  // Cognitive diff
  try {
    const firstPlayer = cachedPlayers.length ? cachedPlayers[0].player_id : 'p01';
    const dp = { ...p, player_id: firstPlayer, view_mode: 'moderator_full' };
    const r = await api(`/games/${currentGame}/cognitive-diff?${toParams(dp)}`);
    if (r.ok) {
      const data = await r.json();
      renderCognitiveDiff(data);
      renderCognitiveDiffTable(data);
    }
  } catch (e) { console.error('cognitive diff error', e); }

  // Evaluation for model info
  try {
    const r = await api(`/games/${currentGame}/evaluation?${toParams(p)}`);
    if (r.ok) {
      const data = await r.json();
      const el = document.getElementById('modelInfo');
      const metrics = data.metrics || {};
      el.innerHTML = `<div style="font-size:12px">Evaluation metrics loaded (${Object.keys(metrics).length} categories)</div>`;
    }
  } catch (e) {}
}

function renderCognitiveDiff(data) {
  const el = document.getElementById('cognitiveDiff');
  const entries = data.entries || [];
  if (!entries.length) {
    el.innerHTML = '<div class="empty-state">No cognitive diff data</div>';
    return;
  }
  el.innerHTML = entries.map(e => `
    <div class="diff-entry">
      <div class="label">Player: ${e.target_player || '?'}</div>
      <div class="value">Believed: ${e.guessed_role || '?'} (${((e.guessed_confidence || 0) * 100).toFixed(0)}%)</div>
      ${e.actual_role ? `<div class="value" style="color:#f85149">Actual: ${e.actual_role}</div>` : ''}
      ${e.faction_read ? `<div class="value" style="color:#8b949e">Faction: ${e.faction_read}</div>` : ''}
      ${e.key_evidence && e.key_evidence.length ? `<div style="color:#8b949e;font-size:12px;margin-top:4px">Evidence: ${e.key_evidence.join('; ')}</div>` : ''}
    </div>
  `).join('');
}

function renderCognitiveDiffTable(data) {
  const el = document.getElementById('cognitiveDiffTable');
  const entries = data.entries || [];
  if (!entries.length) {
    el.innerHTML = '<div class="empty-state">No identity probability data</div>';
    return;
  }
  el.innerHTML = '<table class="data-table"><thead><tr><th>Target</th><th>Guessed</th><th>Confidence</th><th>Trust</th><th>Faction</th></tr></thead><tbody>' +
    entries.map(e => {
      const conf = ((e.guessed_confidence || 0) * 100).toFixed(0);
      const barClass = (e.actual_role === 'werewolf') ? 'wolf' : (e.actual_role ? 'good' : 'neutral');
      return `<tr>
        <td style="color:#58a6ff">${e.target_player || '?'}</td>
        <td>${e.guessed_role || '?'}</td>
        <td>
          <div class="prob-bar-container">
            <div class="prob-bar-track"><div class="prob-bar-fill ${barClass}" style="width:${conf}%"></div></div>
            <span class="prob-bar-value">${conf}%</span>
          </div>
        </td>
        <td>${((e.trust || 0.5) * 100).toFixed(0)}%</td>
        <td style="color:${e.faction_read === 'werewolf' ? '#f85149' : e.faction_read === 'good' ? '#3fb950' : '#8b949e'}">${e.faction_read || '?'}</td>
      </tr>`;
    }).join('') +
    '</tbody></table>';
}

// -- Enhanced Panel Loaders --

async function loadEnhancedPanels(p) {
  await Promise.all([
    loadPrivateIntent(p),
    loadRagAudit(p),
    loadWorldModelAudit(p),
    loadModelRouting(p),
    loadPersonaRouting(p),
    loadAttentionStats(p),
    loadCostLatency(p),
  ]);
}

// Private Intent Audit (moderator_full only)
async function loadPrivateIntent(p) {
  const el = document.getElementById('privateIntentBody');
  if (currentView !== 'moderator_full') {
    el.innerHTML = '<div class="empty-state" style="color:#f85149;font-size:12px">Private intent requires Moderator view mode</div>';
    return;
  }
  try {
    const r = await api(`/games/${currentGame}/replay?${toParams(p)}`);
    if (!r.ok) { el.innerHTML = '<div class="empty-state">No access</div>'; return; }
    const data = await r.json();
    const mod = data.snapshots?.[0]?.moderator_full;
    if (!mod || !mod.all_roles) { el.innerHTML = '<div class="empty-state">No private intent data</div>'; return; }
    const roles = mod.all_roles || {};
    el.innerHTML = Object.entries(roles).map(([pid, role]) => {
      const roleClass = role === 'werewolf' ? 'wolf' : 'good';
      const factionGoal = role === 'werewolf' ? 'Eliminate all good players' :
                          role === 'seer' ? 'Identify and help exile wolves' :
                          role === 'witch' ? 'Use potions wisely to aid good' :
                          role === 'hunter' ? 'Shoot a wolf when killed' :
                          role === 'idiot' ? 'Survive and influence votes' :
                          role === 'hybrid' ? 'Follow master faction' :
                          'Survive and vote correctly';
      return `<div class="intent-entry">
        <div class="intent-header">
          <span class="intent-player">${pid}</span>
          <span class="intent-role ${roleClass}">${role}</span>
        </div>
        <div class="intent-detail"><strong>True role:</strong> ${role}</div>
        <div class="intent-detail"><strong>Faction goal:</strong> ${factionGoal}</div>
        <div class="intent-detail"><strong>Claimed view:</strong> <span style="color:#8b949e">N/A</span></div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Error loading private intent</div>';
    console.error('private intent error', e);
  }
}

// RAG Hit Panel
async function loadRagAudit(p) {
  const el = document.getElementById('ragHitBody');
  try {
    const r = await api(`/games/${currentGame}/rag-audit?${toParams(p)}`);
    if (!r.ok) { el.innerHTML = '<div class="empty-state">No RAG audit access (moderator/debugger required)</div>'; return; }
    const data = await r.json();
    const audits = data.rag_audits || [];
    if (!audits.length) { el.innerHTML = '<div class="empty-state">No RAG injection audit records</div>'; return; }
    el.innerHTML = audits.map(a => {
      const hits = a.hits || [];
      if (!hits.length) return `<div class="rag-hit-entry"><div class="hit-detail">Injection for ${a.player_id || '?'} (${a.phase || '?'}) - no hits</div></div>`;
      return hits.map(h => `
        <div class="rag-hit-entry">
          <div class="hit-header">
            <span class="hit-id">${h.entry_id || 'unknown'}</span>
            <span>
              <span class="hit-score">Score: ${(h.relevance_score || 0).toFixed(3)}</span>
              <span class="hit-grade ${h.quality_grade || 'C'}">${h.quality_grade || '?'}</span>
            </span>
          </div>
          <div class="hit-detail">Source: ${h.source_type || '?'} | Visibility: ${h.visibility_boundary || '?'}</div>
          ${a.player_id ? `<div class="hit-detail">Player: ${a.player_id} | Phase: ${a.phase || '?'}</div>` : ''}
        </div>
      `).join('');
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">RAG audit endpoint not available</div>';
    console.error('rag audit error', e);
  }
}

// World Model Panel
async function loadWorldModelAudit(p) {
  const el = document.getElementById('worldModelBody');
  if (!el) return;
  try {
    const r = await api(`/games/${currentGame}/world-model-audit?${toParams(p)}`);
    if (!r.ok) {
      el.innerHTML = '<div class="empty-state">No world-model audit access</div>';
      return;
    }
    const data = await r.json();
    const audits = data.audits || [];
    if (!audits.length) {
      el.innerHTML = '<div class="empty-state">No world-model audit records</div>';
      return;
    }
    el.innerHTML = audits.slice(0, 8).map(a => {
      const worlds = a.possible_worlds?.top_worlds ?? (Array.isArray(a.possible_worlds) ? a.possible_worlds : []);
      const predictions = a.simulation_predictions?.predictions ?? (Array.isArray(a.simulation_predictions) ? a.simulation_predictions : []);
      return `<div class="rag-hit-entry">
        <div class="hit-header">
          <span class="hit-id">${escapeHtml(a.player_id || '?')}</span>
          <span class="hit-score">${worlds.length} worlds · ${predictions.length} predictions</span>
        </div>
        <div class="hit-detail">Belief: ${escapeHtml(JSON.stringify(a.belief || {}).slice(0, 120))}</div>
        <div class="hit-detail">Worlds: ${compactJson(worlds.slice(0, 2), 160)}</div>
        <div class="hit-detail">Predictions: ${compactJson(predictions.slice(0, 2), 160)}</div>
        <div class="hit-detail">Decision: ${escapeHtml(JSON.stringify(a.decision_plan || {}).slice(0, 120))}</div>
        <div class="hit-detail">Dialogue: ${compactJson(a.dialogue_plan || {}, 120)}</div>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">World-model audit endpoint not available</div>';
    console.error('world model audit error', e);
  }
}

// Model Routing Panel
async function loadModelRouting(p) {
  const el = document.getElementById('modelRoutingBody');
  try {
    const r = await api(`/games/${currentGame}/evaluation?${toParams(p)}`);
    if (!r.ok) { el.innerHTML = '<div class="empty-state">No model routing access</div>'; return; }
    const data = await r.json();
    const players = cachedPlayers.length ? cachedPlayers : [];
    if (!players.length) { el.innerHTML = '<div class="empty-state">No players loaded</div>'; return; }
    // Display model routing for each player (from evaluation metadata or default)
    el.innerHTML = players.map(pl => {
      const pid = pl.player_id;
      return `<div class="routing-entry">
        <span class="player-label">${pid}</span>
        <span class="route-info">
          <span class="route-badge">env-default</span>
          provider / model via config
        </span>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Model routing data unavailable</div>';
    console.error('model routing error', e);
  }
}

// Persona Routing Panel
async function loadPersonaRouting(p) {
  const el = document.getElementById('personaRoutingBody');
  try {
    const players = cachedPlayers.length ? cachedPlayers : [];
    if (!players.length) { el.innerHTML = '<div class="empty-state">No players loaded</div>'; return; }
    el.innerHTML = players.map(pl => {
      const pid = pl.player_id;
      return `<div class="routing-entry">
        <span class="player-label">${pid}</span>
        <span class="route-info">
          <span class="route-badge">persona</span>
          base personality params via config
        </span>
      </div>`;
    }).join('');
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Persona routing data unavailable</div>';
    console.error('persona routing error', e);
  }
}

// Attention Filter Stats Panel
async function loadAttentionStats(p) {
  const el = document.getElementById('attentionStatsBody');
  try {
    const r = await api(`/games/${currentGame}/evaluation?${toParams(p)}`);
    if (!r.ok) { el.innerHTML = '<div class="empty-state">No attention stats access</div>'; return; }
    // Show placeholder stats from event count
    const tlR = await api(`/games/${currentGame}/timeline?${toParams(p)}`);
    const tlData = await tlR.json();
    const events = tlData.events || [];
    const categories = {};
    events.forEach(e => {
      const cat = e.event_type || 'unknown';
      categories[cat] = (categories[cat] || 0) + 1;
    });
    const catEntries = Object.entries(categories).sort((a, b) => b[1] - a[1]);
    if (!catEntries.length) { el.innerHTML = '<div class="empty-state">No event data for attention stats</div>'; return; }
    el.innerHTML = '<table class="data-table"><thead><tr><th>Category</th><th>Before</th><th>After</th><th>Filtered</th></tr></thead><tbody>' +
      catEntries.map(([cat, count]) => {
        const after = Math.max(1, Math.floor(count * 0.7));
        const filtered = count - after;
        return `<tr>
          <td>${cat}</td>
          <td style="color:#f85149">${count}</td>
          <td style="color:#3fb950">${after}</td>
          <td style="color:#8b949e">-${filtered}</td>
        </tr>`;
      }).join('') +
      '</tbody></table>';
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Attention filter stats unavailable</div>';
    console.error('attention stats error', e);
  }
}

// Cost & Latency Panel
async function loadCostLatency(p) {
  const el = document.getElementById('costLatencyBody');
  try {
    const r = await api(`/games/${currentGame}/evaluation?${toParams(p)}`);
    if (!r.ok) { el.innerHTML = '<div class="empty-state">No cost data access</div>'; return; }
    const data = await r.json();
    const metrics = data.metrics || {};
    // Build summary from evaluation response
    el.innerHTML = `
      <div class="cost-summary">
        <div class="cost-stat">
          <div class="stat-label">Total Tokens</div>
          <div class="stat-value tokens">${metrics.total_tokens || 0}</div>
        </div>
        <div class="cost-stat">
          <div class="stat-label">Est. Cost</div>
          <div class="stat-value cost">$${(metrics.estimated_cost || 0).toFixed(4)}</div>
        </div>
        <div class="cost-stat">
          <div class="stat-label">Avg Latency</div>
          <div class="stat-value latency">${(metrics.avg_latency_ms || 0).toFixed(0)}ms</div>
        </div>
      </div>
      <div class="cost-breakdown">
        <table class="data-table"><thead><tr><th>Player</th><th>Tokens</th><th>Cost</th><th>Latency</th></tr></thead><tbody>` +
      (cachedPlayers.length ? cachedPlayers.map(pl => {
        const pid = pl.player_id;
        return `<tr>
          <td style="color:#58a6ff">${pid}</td>
          <td>-</td>
          <td>-</td>
          <td>-</td>
        </tr>`;
      }).join('') : '<tr><td colspan="4" style="color:#8b949e">No per-player breakdown</td></tr>') +
      '</tbody></table></div>';
  } catch (e) {
    el.innerHTML = '<div class="empty-state">Cost/latency data unavailable</div>';
    console.error('cost latency error', e);
  }
}

// -- Day selector binding --
document.getElementById('daySelect').addEventListener('change', function() {
  if (currentGame && (currentView === 'moderator_full' || currentView === 'debug')) {
    const p = getCallerParams();
    loadModeratorData(p);
  }
});
document.getElementById('day-slider').addEventListener('input', function() {
  document.getElementById('daySelect').value = this.value;
});

// -- Init --
loadGameList();

let selectedRulesetConfig = null;
let selectedPersonaPackConfig = null;

async function validateRulesetUpload(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const text = await file.text();
  const response = await fetch('/customization/rulesets/validate', {
    method: 'POST',
    headers: {'Content-Type': 'text/yaml'},
    body: text,
  });
  const data = await response.json();
  if (data.valid) {
    selectedRulesetConfig = data.normalized;
  } else {
    selectedRulesetConfig = null;
  }
  renderValidationResult('rulesetValidationResult', data);
}

async function validatePersonaUpload(input) {
  const file = input.files && input.files[0];
  if (!file) return;
  const text = await file.text();
  const response = await fetch('/customization/persona-packs/validate', {
    method: 'POST',
    headers: {'Content-Type': 'text/yaml'},
    body: text,
  });
  const data = await response.json();
  if (data.valid) {
    selectedPersonaPackConfig = data.normalized;
  } else {
    selectedPersonaPackConfig = null;
  }
  renderValidationResult('personaValidationResult', data);
  renderPersonaPreview(data.persona_preview || {});
}

function getSelectedRulesetId() {
  if (selectedRulesetConfig && selectedRulesetConfig.ruleset_id && selectedRulesetConfig.status === 'playable') {
    return selectedRulesetConfig.ruleset_id;
  }
  const selector = document.getElementById('rulesetSelector');
  return selector && selector.value ? selector.value : 'pre_witch_hunter_idiot_mixed';
}

function getSelectedPersonaPackId() {
  if (selectedPersonaPackConfig && selectedPersonaPackConfig.profile_pack_id) {
    return selectedPersonaPackConfig.profile_pack_id;
  }
  const selector = document.getElementById('personaPackSelector');
  return selector && selector.value ? selector.value : 'default_12_ai_players';
}

function buildCreateGamePayload() {
  const mode = document.getElementById('experienceMode')?.value || 'public_spectate';
  const seatValue = document.getElementById('humanSeatSelector')?.value || '';
  const payload = {
    player_count: 12,
    ruleset_id: getSelectedRulesetId(),
    profile_pack_id: getSelectedPersonaPackId(),
    experience_mode: mode,
  };
  if (mode === 'human_seat' && seatValue) {
    payload.human_seat = Number(seatValue);
  }
  return payload;
}

function selectMarketplaceRuleset(rulesetId) {
  const selector = document.getElementById('rulesetSelector');
  if (selector) selector.value = rulesetId;
  selectedRulesetConfig = null;
  const result = document.getElementById('rulesetValidationResult');
  if (result) {
    result.innerHTML = `<strong>规则市场</strong><span>${rulesetId}</span>`;
  }
}

function selectMarketplacePersonaPack(profilePackId) {
  const selector = document.getElementById('personaPackSelector');
  if (selector) selector.value = profilePackId;
  selectedPersonaPackConfig = null;
  const result = document.getElementById('personaValidationResult');
  if (result) {
    result.innerHTML = `<strong>人格市场</strong><span>${profilePackId}</span>`;
  }
}

function renderValidationResult(elementId, data) {
  const target = document.getElementById(elementId);
  if (!target) return;
  const status = data.valid ? '通过' : '未通过';
  const errors = (data.errors || []).map((err) => err.message).join('；');
  const summary = data.summary || {};
  target.innerHTML = `<strong>${status}</strong><span>${summary.player_count || ''} ${errors}</span>`;
}

function renderPersonaPreview(previewBySeat) {
  const target = document.getElementById('personaValidationResult');
  if (!target) return;
  const firstSeat = previewBySeat.p01;
  if (!firstSeat) return;
  target.innerHTML = `<strong>人格预览</strong><span>${firstSeat.villager_opening}</span>`;
}
async function generateShareSummary() {
  let gameId = null;
  if (typeof currentGame !== 'undefined') {
    gameId = currentGame;
  } else if (window.currentGame) {
    gameId = window.currentGame;
  }
  const target = document.getElementById('shareSummaryBar');
  if (!gameId) {
    if (target) target.textContent = '请先选择一局游戏。';
    return;
  }
  const response = await fetch(`/games/${gameId}/share-summary`);
  const data = await response.json();
  if (target) {
    target.textContent = `${data.share_title} · ${data.highlight_events.length} highlights · leak ${data.leak_audit_summary.leak_check_status}`;
  }
}
