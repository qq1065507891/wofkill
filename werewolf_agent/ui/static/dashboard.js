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
