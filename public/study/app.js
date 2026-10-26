/* Morphos Ledger. Every number on the page is read from /data/study.json (signed). */
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const MINUS = '\u2212';
const f = (x, d = 3) => (x === null || x === undefined || Number.isNaN(x)) ? '\u2013' : (x < 0 ? MINUS : '') + Math.abs(x).toFixed(d);
const sg = (x, d = 3) => (x >= 0 ? '+' : MINUS) + Math.abs(x).toFixed(d);
const NAMES = {
  'qsvm_zz_x0.3': 'QSVM, ZZ kernel, inputs x0.3', 'qsvm_zz_x1.0': 'QSVM, ZZ kernel, inputs x1.0', 'qsvm_angle_x1.0': 'QSVM, angle kernel, x1.0',
  logistic_regression: 'Logistic regression', random_forest: 'Random forest', gradient_boosting: 'Gradient boosting', svm_rbf: 'SVM, RBF kernel', xgboost: 'XGBoost'
};
const nameOf = (k) => NAMES[k] || k;
const PRETTY = { LogisticRegression: 'Logistic regression', RandomForest: 'Random forest' };
let ENV = null, S = null, FACTS = {}, AUDIT = [];

/* ---------- scales and svg helpers ---------- */
const lin = (d0, d1, r0, r1) => (v) => r0 + ((v - d0) / (d1 - d0)) * (r1 - r0);
function ticks(lo, hi, step) { const out = []; for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(6)); return out; }
const svg = (w, h, inner, label, cls = '') => `<svg class="chart ${cls}" style="max-width:${w}px" viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(label)}">${inner}</svg>`;

function intervalPlot(rows, metric) {
  const mean = (r) => r[metric + '_mean'], ci = (r) => r[metric + '_ci'];
  const sorted = [...rows].sort((a, b) => mean(b) - mean(a));
  const bestC = sorted.find((r) => r.kind === 'classical');
  const lo = Math.min(...sorted.map((r) => ci(r)[0])), hi = Math.max(...sorted.map((r) => ci(r)[1]));
  const d0 = Math.floor((lo - 0.02) * 10) / 10, d1 = Math.ceil((hi + 0.02) * 10) / 10;
  const W = 720, L = 190, R = 640, T = 14, RH = 30, H = T + sorted.length * RH + 40;
  const x = lin(d0, d1, L, R);
  let g = '';
  ticks(d0, d1, (d1 - d0) > 0.55 ? 0.2 : 0.1).forEach((t) => { g += `<line class="tick" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - 34}"/><text x="${x(t)}" y="${H - 18}" text-anchor="middle">${f(t, 1)}</text>`; });
  if (d0 < 0 && d1 > 0) g += `<line class="zero" x1="${x(0)}" x2="${x(0)}" y1="${T}" y2="${H - 34}"/>`;
  sorted.forEach((r, i) => {
    const y = T + i * RH + RH / 2;
    const col = r.kind === 'quantum' ? 'var(--verdigris)' : (r === bestC ? 'var(--ink)' : 'var(--ink-2)');
    g += `<text class="lab ${r.kind === 'quantum' ? 'strong' : ''}" x="0" y="${y + 4}">${esc(nameOf(r.model))}</text>`;
    g += `<line x1="${x(ci(r)[0])}" x2="${x(ci(r)[1])}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2"/>`;
    g += `<line x1="${x(ci(r)[0])}" x2="${x(ci(r)[0])}" y1="${y - 4}" y2="${y + 4}" stroke="${col}" stroke-width="2"/><line x1="${x(ci(r)[1])}" x2="${x(ci(r)[1])}" y1="${y - 4}" y2="${y + 4}" stroke="${col}" stroke-width="2"/>`;
    g += `<circle cx="${x(mean(r))}" cy="${y}" r="5" fill="${col}"/><text x="${W - 4}" y="${y + 4}" text-anchor="end" class="strong">${f(mean(r))}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${R}" y1="${H - 34}" y2="${H - 34}"/>`;
  return svg(W, H, g, `Interval plot of ${metric} for ${sorted.length} models`, 'wide');
}

function lineChart({ series, xt, yd, yt, xlabel, w = 480, h = 260 }) {
  const L = 40, R = w - 84, T = 14, B = h - 44;
  const xs = series.flatMap((s) => s.pts.map((p) => p[0]));
  const x = lin(Math.min(...xs), Math.max(...xs), L, R), y = lin(yd[0], yd[1], B, T);
  let g = '';
  yt.forEach((t) => { g += `<line class="tick" x1="${L}" x2="${R}" y1="${y(t)}" y2="${y(t)}"/><text x="${L - 6}" y="${y(t) + 4}" text-anchor="end">${f(t, 1)}</text>`; });
  g += `<line class="zero" x1="${L}" x2="${R}" y1="${y(0)}" y2="${y(0)}"/>`;
  xt.forEach((t) => { g += `<text x="${x(t.v)}" y="${B + 18}" text-anchor="middle">${esc(t.l)}</text>`; });
  g += `<text x="${(L + R) / 2}" y="${h - 4}" text-anchor="middle">${esc(xlabel)}</text>`;
  const ends = series.map((s) => ({ s, y: y(s.pts[s.pts.length - 1][1]) })).sort((a, b) => a.y - b.y);
  for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < 15) ends[i].y = ends[i - 1].y + 15;
  series.forEach((s) => {
    g += `<polyline fill="none" stroke="${s.color}" stroke-width="2" points="${s.pts.map((p) => x(p[0]) + ',' + y(p[1])).join(' ')}"/>`;
    const last = s.pts[s.pts.length - 1], ly = ends.find((e) => e.s === s).y;
    g += `<text class="lab strong" x="${x(last[0]) + 8}" y="${ly + 4}" style="fill:${s.color}">${esc(s.name)}</text>`;
  });
  return svg(w, h, g, 'Line chart of cross-validated MCC');
}

function lollipop(rows) { // single-feature AUROC from the no-skill line at 0.5
  const W = 640, L = 150, R = 560, T = 14, RH = 34, H = T + rows.length * RH + 40, x = lin(0.5, 0.9, L, R);
  let g = '';
  ticks(0.5, 0.9, 0.1).forEach((t) => { g += `<line class="tick" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - 34}"/><text x="${x(t)}" y="${H - 18}" text-anchor="middle">${f(t, 1)}</text>`; });
  g += `<line class="zero" x1="${x(0.5)}" x2="${x(0.5)}" y1="${T}" y2="${H - 34}"/>`;
  [...rows].sort((a, b) => b.auroc_alone - a.auroc_alone).forEach((r, i) => {
    const y = T + i * RH + RH / 2, col = r.feature === 'time' ? 'var(--brass)' : 'var(--ink-2)';
    g += `<text class="lab ${r.feature === 'time' ? 'strong' : ''}" x="0" y="${y + 4}">${esc(r.feature)}</text><line x1="${x(0.5)}" x2="${x(r.auroc_alone)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2"/><circle cx="${x(r.auroc_alone)}" cy="${y}" r="5" fill="${col}"/><text x="${x(r.auroc_alone) + 10}" y="${y + 4}" class="strong">${f(r.auroc_alone)}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${R}" y1="${H - 34}" y2="${H - 34}"/>`;
  return svg(W, H, g, 'Single-feature AUROC', 'wide');
}

function dumbbell(rows) {
  const W = 720, L = 220, R = 600, T = 14, RH = 36, H = T + rows.length * RH + 40;
  const d0 = 0.2, d1 = 0.7, x = lin(d0, d1, L, R);
  let g = '';
  ticks(d0, d1, 0.1).forEach((t) => { g += `<line class="tick" x1="${x(t)}" x2="${x(t)}" y1="${T}" y2="${H - 34}"/><text x="${x(t)}" y="${H - 18}" text-anchor="middle">${f(t, 1)}</text>`; });
  rows.forEach((r, i) => {
    const y = T + i * RH + RH / 2, q = r.model.startsWith('QSVM'), col = q ? 'var(--verdigris)' : 'var(--ink)';
    g += `<text class="lab ${q ? 'strong' : ''}" x="0" y="${y + 4}">${esc(PRETTY[r.model] || r.model.replace(' (ZZ kernel, inputs x 0.3)', ', ZZ kernel'))}</text>`;
    g += `<line x1="${x(r.mcc_without_time)}" x2="${x(r.mcc_with_time)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="2"/>`;
    g += `<circle cx="${x(r.mcc_without_time)}" cy="${y}" r="5" fill="var(--paper)" stroke="${col}" stroke-width="2"/><circle cx="${x(r.mcc_with_time)}" cy="${y}" r="5" fill="${col}"/>`;
    g += `<text x="${W - 4}" y="${y + 4}" text-anchor="end" class="strong">${sg(-r.mcc_drop)}</text>`;
  });
  g += `<line class="axis" x1="${L}" x2="${R}" y1="${H - 34}" y2="${H - 34}"/><text x="${R}" y="${H - 2}" text-anchor="end">filled: with time, open: serum sodium instead</text>`;
  return svg(W, H, g, 'With-time versus without-time MCC for four models', 'wide');
}

/* ---------- facts used in the page and in the walkthrough ---------- */
function mccFrom(tn, fp, fn, tp) { const d = Math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)); return d === 0 ? 0 : (tp * tn - fp * fn) / d; }
function pill(status, label) { return `<span class="pill ${esc(status)}">${esc(label)}</span>`; }
function computeFacts() {
  const gw = S.guard.withtime, vm = gw.verdicts.mcc, e5 = S.e5, cm = e5.holdout_confusion, sens = S.sensitivity;
  const e3 = (S.sweeps.e3 || []).find((r) => r.data === 'withtime' && r.range === '0 to 2pi' && r.depth === 1 && r.class_weight === 'none');
  const drops = (S.leakage?.paired || []).map((r) => r.mcc_drop);
  const time = (S.leakage?.single_feature || []).find((r) => r.variant === 'withtime' && r.feature === 'time');
  const holdN = S.claims.filter((c) => ['reproduced', 'supported'].includes(c.status)).length;
  FACTS = {
    n_train: S.meta.dataset.train, n_test: S.meta.dataset.test, q_mcc: f(vm.quantum_score), c_mcc: f(vm.classical_score),
    e1_mcc: f(S.bloq.E1.output.mcc, 4), e2_mcc: f(S.bloq.E2.output.mcc, 4), e3_2pi: e3 ? f(e3.cv_mcc, 2) : 'n/a',
    gap_n: S.gap ? S.gap.notime.splits : 'n/a', n_hold: holdN, n_claims: S.claims.length, hold_mcc: f(e5.holdout.mcc), zz_same: f(e5.zz_same_range_holdout_mcc),
    n_err: cm[0][1] + cm[1][0], one_min: f(sens.one_patient_min), one_max: f(sens.one_patient_max), two_min: f(sens.two_patients_min), two_max: f(sens.two_patients_max),
    n_splits: (gw.protocol.match(/= (\d+) paired/) || [0, '15'])[1], configs: vm.configs_tried, verdict_cat: vm.category.replace(/_/g, ' ').toLowerCase(),
    time_auc: time ? f(time.auroc_alone) : 'n/a', drop_min: drops.length ? f(Math.min(...drops)) : 'n/a', drop_max: drops.length ? f(Math.max(...drops)) : 'n/a',
    audit_entries: ENV.audit.entries
  };
}

/* ---------- sections ---------- */
function renderMeta() {
  const d = new Date(S.meta.built_at);
  $('#meta').innerHTML = `built ${esc(d.toISOString().slice(0, 16).replace('T', ' '))} UTC<br>key ${esc(ENV.public_key_fingerprint.replace('SHA256:', '').slice(0, 19))} &middot; ${ENV.audit.entries} audit entries`;
}

function verdictHtml() {
  const v = S.guard.withtime.verdicts.mcc, n = S.meta.dataset.train, hold = S.e5.holdout.mcc, sens = S.sensitivity;
  const dev = Math.max(Math.abs(sens.one_patient_max - hold), Math.abs(hold - sens.one_patient_min));
  const gap = `${f(v.quantum_score)} against ${f(v.classical_score)}, a gap of ${sg(v.delta)} MCC (95% interval ${sg(v.ci_lower)} to ${sg(v.ci_upper)})`;
  const tail = `The submitted holdout score of ${f(hold)} comes from ${S.meta.dataset.test} patients, and one changed prediction moves it by up to ${f(dev)}.`;
  const lead = {
    NO_DIFFERENCE: `On ${n} training patients the best quantum kernel and the best classical model are <em>statistically tied</em>: ${gap}.`,
    ADVANTAGE_SUPPORTED: `On ${n} training patients the best quantum kernel <em>beats</em> the best classical model: ${gap}.`,
    CLASSICAL_BETTER: `On ${n} training patients the best classical model <em>beats</em> the best quantum kernel: ${gap}.`,
    INSUFFICIENT_DATA: `There are <em>too few paired splits</em> to compare the quantum kernel with classical models.`
  }[v.category];
  return `${lead} ${tail}`;
}

function renderAnswer() {
  const st = S.claims.reduce((a, c) => (a[c.status] = (a[c.status] || 0) + 1, a), {});
  const hold = (st.reproduced || 0) + (st.supported || 0), v = S.guard.withtime.verdicts.mcc;
  $('#answerBody').innerHTML = `
    <p class="q">One question first: on the heart-failure records, does a quantum-kernel SVM predict death better than classical models, and which of our own claims survive a re-check?</p>
    <p class="verdict">${verdictHtml()}</p>
    <div class="figs">
      <div class="fig"><div class="l">Submitted QSVM, holdout MCC (${S.meta.dataset.test} patients)</div><div class="v">${f(S.e5.holdout.mcc)}</div><div class="s">Reproduced to nine decimals with an independent kernel.</div></div>
      <div class="fig"><div class="l">Best quantum vs best classical, MCC over ${FACTS.n_splits} paired splits</div><div class="v">${f(v.quantum_score)} / ${f(v.classical_score)}</div><div class="s">Guard verdict: ${esc(v.category.replace(/_/g, ' ').toLowerCase())}.</div></div>
      <div class="fig"><div class="l">Claims from the report, re-checked</div><div class="v">${hold} of ${S.claims.length}</div><div class="s">hold. ${st.reproduced || 0} reproduced, ${st.supported || 0} supported; ${(st.partly || 0) + (st.not_reproduced || 0) + (st.hypothesis || 0) + (st.not_verifiable || 0)} do not fully hold.</div></div>
      <div class="fig"><div class="l">Seal on this study</div><div class="v">ML-DSA-65</div><div class="s">${ENV.audit.entries} signed audit entries, ${S.manifest.length} hashed files.</div></div>
    </div>
    <p class="note" style="margin-top:14px">Every figure is computed by code from the submitted files. MCC is the Matthews correlation coefficient, the hackathon's ranking metric: 0 is no skill, 1 is perfect.</p>`;
}

function renderLedger() {
  const rows = S.ledger.map((r) => `
    <tr class="row-btn" data-id="${esc(r.id)}" tabindex="0" role="button" aria-expanded="false">
      <td class="id">${esc(r.id)}</td>
      <td>${esc(r.what)}<span class="sub">${esc(r.span || r.provenance)}</span></td>
      <td>${esc(r.where)}</td><td>${esc(r.data)}</td>
      <td class="n">${r.cv_mcc === null || r.cv_mcc === undefined ? '<span class="dash">\u2013</span>' : f(r.cv_mcc)}</td>
      <td class="n">${r.holdout_mcc === null || r.holdout_mcc === undefined ? '<span class="dash">\u2013</span>' : f(r.holdout_mcc)}</td>
      <td>${pill(r.status, r.status_label)}</td>
    </tr>`).join('');
  $('#ledgerBody').innerHTML = `
    <p class="q">Click a row to see the claims behind its status. E1 and E2 list Bloq's own 48-row validation split, not the 60-patient holdout. E4 lists the best cross-validated range.</p>
    <div class="tscroll"><table><thead><tr><th>Id</th><th>Experiment</th><th>Where</th><th>Data</th><th class="n">CV MCC</th><th class="n">Holdout MCC</th><th>Re-check</th></tr></thead><tbody>${rows}</tbody></table></div>
    <p class="note" style="margin-top:14px">Re-check means: code re-ran the experiment, or recomputed the figure from the stored file, and compared. E5 stays partly reproduced because its tree-model baselines and one fold do not reproduce, and because the Bloq run itself cannot be verified from outside.</p>`;
  $$('#ledgerBody tr.row-btn').forEach((tr) => {
    const toggle = () => {
      const open = tr.getAttribute('aria-expanded') === 'true';
      const next = tr.nextElementSibling;
      if (next && next.classList.contains('detail')) next.remove();
      tr.setAttribute('aria-expanded', String(!open));
      if (open) return;
      const row = S.ledger.find((x) => x.id === tr.dataset.id);
      const items = row.claim_ids.map((id) => S.claims.find((c) => c.id === id)).filter(Boolean)
        .map((c) => `<tr><td style="width:150px">${pill(c.status, c.status_label)}</td><td>${esc(c.claim)}<span class="sub">${esc(c.recomputed)}</span></td></tr>`).join('');
      tr.insertAdjacentHTML('afterend', `<tr class="detail"><td></td><td colspan="6"><table><tbody>${items}</tbody></table></td></tr>`);
    };
    tr.addEventListener('click', toggle);
    tr.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
  });
}

function bloqCard(tag) {
  const d = S.bloq[tag], o = d.output, p = d.parameters, cm = o.confusion_matrix;
  const t = (type) => new Date(o.logs.find((l) => l.type === type).timestamp).getTime();
  const secs = ((t('completed') - t('queued')) / 1000).toFixed(1);
  return `<div><h3>${tag}: ${d.selectedFeatures.includes('time') ? 'with time' : 'no time'}</h3>
    <dl class="kv"><dt>Algorithm</dt><dd>${esc(d.algorithm)}, ${p['No. of Qubits']} qubits, depth ${p['Circuit Depth']}</dd>
    <dt>Feature map</dt><dd>${esc(p['Feature Map'].name)} (platform default)</dd>
    <dt>Device</dt><dd>${esc(p.Device.name)}, ${esc(p.Device.provider)}, simulator</dd>
    <dt>Features</dt><dd>${esc(d.selectedFeatures.join(', '))}</dd>
    <dt>Evaluation</dt><dd>${o.confusion_matrix.flat().reduce((a, b) => a + b, 0)} rows (platform split of the 239)</dd>
    <dt>Title as saved</dt><dd>${esc(d.title)}${d.title.includes('notime') && d.selectedFeatures.includes('time') ? ' (wrong label: this run used time)' : ''}</dd>
    <dt>Queue to done</dt><dd>${secs} s</dd></dl>
    <table class="cm"><thead><tr><th></th><th>pred. survived</th><th>pred. died</th></tr></thead><tbody>
    <tr><th>survived</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr><tr><th>died</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr></tbody></table>
    <p style="margin-top:9px">MCC <b class="mono">${f(o.mcc, 4)}</b>, accuracy <b class="mono">${f(o.accuracy, 4)}</b>. ${cm[1][1] === 0 ? 'No patient was predicted to die.' : `${cm[1][1]} of ${cm[1][0] + cm[1][1]} deaths found.`}</p></div>`;
}

function renderPlatform() {
  const e3 = S.sweeps.e3 || [], e4 = S.sweeps.e4 || [];
  const e3pt = (variant) => ['0 to 1', '0 to pi', '0 to 2pi'].map((r, i) => [i, e3.find((x) => x.data === variant && x.range === r && x.depth === 1 && x.class_weight === 'none')?.cv_mcc]).filter((p) => p[1] !== undefined);
  const e4pt = (variant) => e4.filter((x) => x.data === variant && x.mode === 'minmax').sort((a, b) => a.scale - b.scale).map((x) => [x.scale, x.cv_mcc]);
  const c3 = e3.length ? lineChart({ series: [{ name: 'with time', color: 'var(--verdigris)', pts: e3pt('withtime') }, { name: 'no time', color: 'var(--ink-2)', pts: e3pt('notime') }],
    xt: [{ v: 0, l: '0 to 1' }, { v: 1, l: '0 to \u03c0' }, { v: 2, l: '0 to 2\u03c0' }], yd: [-0.1, 0.6], yt: [0, 0.2, 0.4, 0.6], xlabel: 'input range fed to the ZZ feature map' }) : '';
  const c4 = e4.length ? lineChart({ series: [{ name: 'with time', color: 'var(--verdigris)', pts: e4pt('withtime') }, { name: 'no time', color: 'var(--ink-2)', pts: e4pt('notime') }],
    xt: [0.3, 0.5, 0.75, 1, 1.5].map((v) => ({ v, l: String(v) })), yd: [0, 0.7], yt: [0, 0.2, 0.4, 0.6], xlabel: 'input range multiplier (min-max scaled inputs)' }) : '';
  const g = S.gap, gn = g?.notime, gw = g?.withtime;
  $('#platformBody').innerHTML = `
    <p class="q">E1 and E2 are the raw Bloq platform outputs. They are close to no skill, which is why we ran local sweeps.</p>
    <div class="two even">${bloqCard('E1')}${bloqCard('E2')}</div>
    <h3>E3 and E4: does a wider input range make the kernel collapse?</h3>
    <div class="two even"><div><p class="cap">E3, depth 1: cross-validated MCC by range</p><div class="chartwrap">${c3}</div></div><div><p class="cap">E4, repeated 5-fold CV: MCC by range multiplier</p><div class="chartwrap">${c4}</div></div></div>
    ${g ? `<h3>Does that explain the Bloq collapse?</h3>
    <p>Not proven. With inputs in 0 to 1, our local ZZ kernel predicted <b>no deaths in ${gn.splits_predicting_no_deaths} of ${gn.splits} random splits</b> without time and in ${gw.splits_predicting_no_deaths} of ${gw.splits} with time. Its MCC was at most 0.05 in ${(gn.share_mcc_at_most_0_05 * 100).toFixed(1)}% of the no-time splits (mean ${f(gn.mcc_mean)}, 5th to 95th percentile ${f(gn.mcc_p05)} to ${f(gn.mcc_p95)}). So split luck does not explain E1 and E2. The collapse appears locally only when inputs are stretched to 0 to 2\u03c0. Whether the platform rescales inputs internally cannot be checked from outside, so we call it a hypothesis.</p>` : ''}
    <p class="note">E4 also shows why the range was chosen by cross-validation: with-time CV MCC changes little across ranges while the 60-patient holdout swings widely.</p>`;
}

function renderRepro() {
  const FILTERS = [['all', 'All'], ['hold', 'Hold'], ['not', 'Do not fully hold']];
  const draw = (flt) => {
    const rows = S.claims.filter((c) => flt === 'all' || (flt === 'hold' ? ['reproduced', 'supported'].includes(c.status) : !['reproduced', 'supported'].includes(c.status)))
      .map((c) => `<tr data-claim="${esc(c.id)}"><td style="width:150px">${pill(c.status, c.status_label)}</td><td>${esc(c.claim)}<span class="sub">${esc(c.source)}</span></td><td>${esc(c.recomputed)}<span class="sub">${esc(c.evidence)}</span></td></tr>`).join('');
    $('#claimRows').innerHTML = rows;
  };
  $('#reproBody').innerHTML = `
    <p class="q">Each row is a sentence from the submitted report. The middle column is the value code recomputed from the stored files. Status says what happened.</p>
    <div class="chips" role="group" aria-label="Filter claims">${FILTERS.map(([k, l]) => `<button type="button" data-f="${k}" aria-pressed="${k === 'all'}">${l}</button>`).join('')}</div>
    <div class="tscroll"><table><thead><tr><th>Status</th><th>Claim and source</th><th>Recomputed value and note</th></tr></thead><tbody id="claimRows"></tbody></table></div>`;
  draw('all');
  $$('#reproBody .chips button').forEach((b) => b.addEventListener('click', () => { $$('#reproBody .chips button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); draw(b.dataset.f); }));
}

function renderHoldout() {
  const cm0 = S.e5.holdout_confusion, [[tn0, fp0], [fn0, tp0]] = cm0;
  const surv = [...Array(tn0).fill(true), ...Array(fp0).fill(false)], dead = [...Array(tp0).fill(true), ...Array(fn0).fill(false)];
  const base = mccFrom(tn0, fp0, fn0, tp0);
  const cell = (ok, cls, i, grp) => `<button type="button" class="${cls}${ok ? '' : ' wrong'}" data-g="${grp}" data-i="${i}" aria-label="${grp === 's' ? 'survivor' : 'death'} ${i + 1}, prediction ${ok ? 'correct' : 'wrong'}" title="${grp === 's' ? 'survivor' : 'died'}, ${ok ? 'predicted correctly' : 'predicted wrongly'} (click to flip)"></button>`;
  const draw = () => {
    const tn = surv.filter(Boolean).length, fp = surv.length - tn, tp = dead.filter(Boolean).length, fn = dead.length - tp, m = mccFrom(tn, fp, fn, tp);
    $('#sGrid').innerHTML = surv.map((ok, i) => cell(ok, 'surv', i, 's')).join('');
    $('#dGrid').innerHTML = dead.map((ok, i) => cell(ok, 'dead', i, 'd')).join('');
    $('#holdMcc').textContent = f(m);
    $('#holdDelta').textContent = (Math.abs(m - base) < 5e-10 ? 'as submitted' : `${sg(m - base)} from the submitted ${f(base)}`);
    $('#cmBody').innerHTML = `<tr><th>survived</th><td>${tn}</td><td>${fp}</td></tr><tr><th>died</th><td>${fn}</td><td>${tp}</td></tr>`;
    $$('#holdBody .dotgrid button').forEach((b) => b.addEventListener('click', () => { const a = b.dataset.g === 's' ? surv : dead; a[+b.dataset.i] = !a[+b.dataset.i]; draw(); }));
  };
  const s = S.sensitivity;
  $('#holdBody').innerHTML = `
    <p class="q">Each dot is one holdout patient. Circles survived, squares died. Grey means the submitted QSVM predicted that patient correctly, red means it did not. Click any dot to flip its prediction and watch the score move.</p>
    <div class="dots">
      <div><p class="cap">${surv.length} patients survived</p><div class="dotgrid" id="sGrid" style="grid-template-columns:repeat(7,22px)"></div></div>
      <div><p class="cap">${dead.length} patients died</p><div class="dotgrid" id="dGrid" style="grid-template-columns:repeat(5,22px)"></div></div>
      <div><p class="cap">Holdout MCC, live</p><div class="bignum" id="holdMcc">0</div><p class="note" id="holdDelta" style="margin:5px 0 14px"></p>
        <table class="cm"><thead><tr><th></th><th>pred. survived</th><th>pred. died</th></tr></thead><tbody id="cmBody"></tbody></table>
        <p style="margin-top:14px"><button type="button" id="holdReset">Reset to submitted</button></p></div>
    </div>
    <p style="margin-top:23px">Flipping one prediction moves MCC between <b class="mono">${f(s.one_patient_min)}</b> and <b class="mono">${f(s.one_patient_max)}</b>. Flipping two moves it between <b class="mono">${f(s.two_patients_min)}</b> and <b class="mono">${f(s.two_patients_max)}</b>. Gaps of a few hundredths between models on this holdout are inside that noise.</p>`;
  draw();
  $('#holdReset').addEventListener('click', () => { surv.fill(true, 0, tn0); surv.fill(false, tn0); dead.fill(true, 0, tp0); dead.fill(false, tp0); draw(); });
}

function renderFair() {
  let variant = 'withtime', metric = 'mcc';
  const draw = () => {
    const g = S.guard[variant], v = g.verdicts[metric], ml = metric === 'mcc' ? 'MCC' : 'AUROC';
    $('#fairOut').innerHTML = `
      <div class="chartwrap">${intervalPlot(g.table, metric)}</div>
      <p class="note">${esc(ml)}, mean and 95% t-interval. Green: quantum kernels. Black: best classical model. ${esc(g.protocol)}.</p>
      <h3>Honesty Guard verdict: ${pill(v.category === 'ADVANTAGE_SUPPORTED' ? 'reproduced' : 'partly', v.category.replace(/_/g, ' '))}</h3>
      <dl class="kv"><dt>Best quantum</dt><dd>${esc(nameOf(v.best_quantum_model))}, ${esc(ml)} ${f(v.quantum_score)}</dd><dt>Best classical</dt><dd>${esc(nameOf(v.best_classical_model))}, ${esc(ml)} ${f(v.classical_score)}</dd>
      <dt>Difference</dt><dd>${sg(v.delta)} (interval ${sg(v.ci_lower)} to ${sg(v.ci_upper)})</dd><dt>Corrected p</dt><dd>${f(v.p_value, 4)} after Nadeau-Bengio and Holm over ${v.configs_tried} quantum configurations</dd></dl>
      <p class="sentence">${esc(v.sentence)}</p>`;
  };
  $('#fairBody').innerHTML = `
    <p class="q">Same splits for every model, scikit-learn defaults for the classical models, and a guard that must correct for how many quantum variants were tried. Quantum and classical rows are judged by code that did not choose them.</p>
    <div class="seg" id="segV" role="group" aria-label="Feature set"><button type="button" data-v="withtime" aria-pressed="true">With follow-up time</button>${S.guard.notime ? '<button type="button" data-v="notime" aria-pressed="false">Without (serum sodium)</button>' : ''}</div>
    <div class="seg" id="segM" role="group" aria-label="Metric" style="margin-left:14px"><button type="button" data-m="mcc" aria-pressed="true">MCC</button><button type="button" data-m="auroc" aria-pressed="false">AUROC</button></div>
    <div id="fairOut"></div>`;
  draw();
  $$('#segV button').forEach((b) => b.addEventListener('click', () => { variant = b.dataset.v; $$('#segV button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); draw(); }));
  $$('#segM button').forEach((b) => b.addEventListener('click', () => { metric = b.dataset.m; $$('#segM button').forEach((x) => x.setAttribute('aria-pressed', String(x === b))); draw(); }));
}

function renderTime() {
  const L = S.leakage;
  if (!L) { $('#timeBody').innerHTML = '<div class="empty">This build skipped the leakage study. Run <code>qneuro study build</code> without <code>--fast</code>.</div>'; return; }
  const tn = L.single_feature.filter((r) => r.variant === 'withtime');
  const t = tn.find((r) => r.feature === 'time');
  $('#timeBody').innerHTML = `
    <p class="q">The hackathon track allows follow-up time as an input. It records how long a patient was observed, which depends on whether they died, so a model that needs it cannot predict ahead.</p>
    <div class="stack"><div><p class="cap">Each feature alone: AUROC on the 239 training patients (0.5 means no signal)</p><div class="chartwrap">${lollipop(tn)}</div></div>
    <div><p class="cap">Cross-validated MCC, with time versus serum sodium in its place</p><div class="chartwrap">${dumbbell(L.paired)}</div></div></div>
    <p style="margin-top:23px"><span class="mono">time</span> alone scores AUROC <b class="mono">${f(t.auroc_alone)}</b>, ${t.direction}. Every model loses between <b class="mono">${f(Math.min(...L.paired.map((r) => r.mcc_drop)))}</b> and <b class="mono">${f(Math.max(...L.paired.map((r) => r.mcc_drop)))}</b> MCC when it is removed. The with-time scores are therefore optimistic for a deployable model. ${esc(L.note)}</p>
    <p class="note">Our submitted report listed this no-time comparison as future work. It was run here locally, not on the Bloq platform.</p>`;
}

/* ---------- integrity: in-browser hashing ---------- */
const hex = (buf) => [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, '0')).join('');
async function sha256(bytes) { return hex(await crypto.subtle.digest('SHA-256', bytes)); }
function flipDigit(bytes, anchor) {
  const txt = new TextDecoder('latin1').decode(bytes), i = txt.indexOf(anchor);
  let k = txt.indexOf('.', i) + 1; while (/\d/.test(txt[k])) k++;
  const pos = k - 1, out = bytes.slice(); const before = txt.slice(pos - 4, pos + 2);
  out[pos] = 48 + ((out[pos] - 48 + 1) % 10);
  return { out, pos, before, after: new TextDecoder('latin1').decode(out).slice(pos - 4, pos + 2) };
}
async function verifyAll(corrupt) {
  const band = $('#verifyBand'), rows = $$('#manifestBody tr');
  if (!(window.crypto && crypto.subtle)) { band.innerHTML = '<div class="error"><b>Hashing needs https or localhost.</b> Open the site with <code>qneuro study serve</code>.</div>'; return; }
  let bad = 0, note = '';
  for (const m of S.manifest) {
    const tr = rows.find((r) => r.dataset.path === m.path), cell = tr.querySelector('.res');
    let bytes = new Uint8Array(await (await fetch('/data/artifacts/' + m.path)).arrayBuffer());
    if (corrupt && m.path.endsWith('SUBMITTED_corrected.json')) { const r = flipDigit(bytes, '"mcc_mean"'); bytes = r.out; note = `Byte ${r.pos} changed in memory: ...${r.before} became ...${r.after}`; }
    const got = await sha256(bytes), ok = got === m.sha256;
    if (!ok) { bad++; note += ` Expected ${m.sha256.slice(0, 16)}..., got ${got.slice(0, 16)}...`; }
    cell.innerHTML = ok ? '<span class="ok">match</span>' : '<span class="bad">MISMATCH</span>';
  }
  band.innerHTML = bad ? `<div class="band">TAMPERED: ${bad} of ${S.manifest.length} files fail their signed hash. ${esc(note)}</div>` : `<p class="status-line ok">All ${S.manifest.length} files match the hashes recorded in the signed manifest.</p>`;
}

function renderIntegrity() {
  const D = S.integrity.diff;
  const diff = `<div class="diff"><div class="h">Field</div><div class="h">Bloq export (before)</div><div class="h">Submitted (after)</div>${D.map((d) => `<div class="mono">${esc(d.field)}</div><div class="del">${esc(d.before)}</div><div class="add">${esc(d.after)}</div>`).join('')}</div>`;
  const man = S.manifest.map((m) => `<tr data-path="${esc(m.path)}"><td class="mono" style="word-break:break-all">${esc(m.path)}</td><td>${esc(m.role)}</td><td class="mono" title="${esc(m.sha256)}">${esc(m.sha256.slice(0, 16))}...</td><td class="res"><span class="dash">not checked</span></td></tr>`).join('');
  $('#intBody').innerHTML = `
    <p class="q">Three text fields were corrected between the Bloq pipeline export and the file we submitted. The diff below is computed from the two files. ${S.integrity.only_text_fields_changed ? 'Only text fields differ, every number is identical.' : 'Non-text fields also differ.'}</p>
    ${diff}
    <h3>Hash every input and output, in this browser</h3>
    <p class="note">The signed study contains a manifest of SHA-256 hashes. These buttons fetch each file from this site and recompute its hash with the browser's own crypto.</p>
    <p><button type="button" class="primary" id="btnVerify">Verify all files</button> <button type="button" id="btnCorrupt" style="margin-left:9px">Corrupt one byte of the submitted file, then verify</button></p>
    <div id="verifyBand" aria-live="polite"></div>
    <div class="tscroll"><table><thead><tr><th>File</th><th>What it is</th><th>SHA-256 (signed)</th><th>This browser</th></tr></thead><tbody id="manifestBody">${man}</tbody></table></div>
    <h3>Signature and audit chain</h3>
    <dl class="kv"><dt>Algorithm</dt><dd>${esc(ENV.algorithm)} (NIST FIPS 204)</dd><dt>Public key</dt><dd>${esc(ENV.public_key_fingerprint)}</dd><dt>Study SHA-256</dt><dd>${esc(ENV.study_sha256)}</dd><dt>Audit chain</dt><dd>${ENV.audit.entries} entries, head ${esc(ENV.audit.head_hash.slice(0, 24))}...</dd></dl>
    <p class="note">A browser can hash files, but it cannot check an ML-DSA signature. The command line can:</p>
<pre class="cmd">qneuro study verify
qneuro study tamper-demo
qneuro study reproduce</pre>
    <details><summary style="cursor:pointer;margin:9px 0">Show the ${ENV.audit.entries} audit entries</summary><div class="tscroll"><table><thead><tr><th class="n">#</th><th>Action</th><th>Detail</th><th>Entry hash</th><th>Previous hash</th></tr></thead><tbody id="auditBody"></tbody></table></div></details>`;
  $('#btnVerify').addEventListener('click', () => verifyAll(false));
  $('#btnCorrupt').addEventListener('click', () => verifyAll(true));
  $('#auditBody').innerHTML = AUDIT.map((a) => `<tr><td class="n">${a.index}</td><td class="mono">${esc(a.action)}</td><td>${esc(a.detail)}</td><td class="mono">${esc(a.entry_hash.slice(0, 12))}</td><td class="mono">${esc(a.prev_hash.slice(0, 12))}</td></tr>`).join('');
}

function renderLimits() {
  $('#limBody').innerHTML = `<ul style="margin:0;padding-left:20px;max-width:78ch">
    <li>Everything quantum here is a classical simulation. No hardware ran any of these circuits.</li>
    <li>The holdout has ${S.meta.dataset.test} patients. Differences of a few hundredths in MCC are noise there.</li>
    <li>That E5 was produced on the Bloq platform cannot be verified by any tool outside the platform. Keep the experiment ids and screenshots.</li>
    <li>E3 and E4 numbers are exact re-runs. The tree-model baselines of E5 do not reproduce with default settings.</li>
    <li>The post-quantum libraries are pure-Python reference implementations. They are unaudited and meant for demonstration.</li>
    <li>The 299 patients come from one hospital pair in Faisalabad, Pakistan (105 women, 194 men, NYHA class III or IV). Results do not carry over to other populations by default.</li>
    <li>This is research tooling on a public dataset (UCI Heart Failure Clinical Records, id 519, CC BY 4.0). It is not a diagnostic tool.</li></ul>
    <p class="note" style="margin-top:23px">The earlier synthetic-cohort pipeline test is kept at <a href="/pipeline-demo.html">pipeline-demo.html</a>. It is a software test fixture, not a result.</p>`;
}

/* ---------- walkthrough ---------- */
const Tour = { steps: [], i: -1,
  fill(t) { return t.replace(/\{\{(\w+)\}\}/g, (_, k) => (k in FACTS ? FACTS[k] : `[missing ${k}]`)); },
  async start() { document.body.classList.add('touring'); this.go(0); },
  stop() { document.body.classList.remove('touring'); $$('section.focus').forEach((s) => s.classList.remove('focus')); this.i = -1; },
  async go(i) {
    if (i < 0 || i >= this.steps.length) return this.stop();
    this.i = i; const s = this.steps[i];
    $$('section.focus').forEach((x) => x.classList.remove('focus'));
    const el = $(s.target); el.classList.add('focus'); el.scrollIntoView({ block: 'start', behavior: 'auto' });
    $('#tour').innerHTML = `<div class="tourbar"><i style="width:${((i + 1) / this.steps.length) * 100}%"></i></div><div class="grid">
      <div class="step">STEP ${i + 1} OF ${this.steps.length}<b>${esc(s.title)}</b></div>
      <div><h4>On screen</h4><p>${esc(s.show)}</p>${s.terminal ? `<h4 style="margin-top:9px">Run in the terminal</h4><p class="mono">${s.terminal.map(esc).join('<br>')}</p>` : ''}</div>
      <div class="say"><h4>Say</h4><p>${esc(this.fill(s.say))}</p></div>
      <div class="btns"><button type="button" id="tPrev" ${i === 0 ? 'disabled' : ''}>Back</button><button type="button" class="primary" id="tNext">${i === this.steps.length - 1 ? 'Finish' : 'Next'}</button><button type="button" id="tStop">Close</button></div></div>`;
    $('#tPrev').onclick = () => this.go(i - 1); $('#tNext').onclick = () => this.go(i + 1); $('#tStop').onclick = () => this.stop();
    if (s.action) this.act(s.action);
  },
  async act(a) {
    const click = (sel) => { const e = $(sel); if (e) e.click(); };
    await new Promise((r) => setTimeout(r, 450)); click(a.click);
    if (a.then) { await new Promise((r) => setTimeout(r, 1800)); click(a.then); }
  },
  next() { return this.go(this.i + 1); }, prev() { return this.go(this.i - 1); }
};
document.addEventListener('keydown', (e) => {
  if (!document.body.classList.contains('touring')) return;
  if (e.key === 'ArrowRight') Tour.next(); else if (e.key === 'ArrowLeft') Tour.prev(); else if (e.key === 'Escape') Tour.stop();
});

/* ---------- boot ---------- */
async function boot() {
  try {
    const [env, steps, audit] = await Promise.all([
      fetch('/data/study.json').then((r) => { if (!r.ok) throw new Error('data/study.json returned ' + r.status); return r.json(); }),
      fetch('/data/walkthrough.json').then((r) => r.json()),
      fetch('/data/audit.jsonl').then((r) => r.text())
    ]);
    ENV = env; S = env.study; Tour.steps = steps;
    AUDIT = audit.trim().split('\n').map((l) => JSON.parse(l));
    computeFacts(); renderMeta(); renderAnswer(); renderLedger(); renderPlatform(); renderRepro(); renderHoldout(); renderFair(); renderTime(); renderIntegrity(); renderLimits();
    $('#startTour').addEventListener('click', () => Tour.start());
    const links = $$('#navlist a'), obs = new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) links.forEach((a) => a.classList.toggle('on', a.getAttribute('href') === '#' + e.target.id)); }), { rootMargin: '-20% 0px -70% 0px' });
    $$('main section').forEach((s) => obs.observe(s));
    window.QN = { tour: Tour, facts: FACTS, ready: true };
    document.dispatchEvent(new Event('qn-ready'));
  } catch (err) {
    $('#answerBody').innerHTML = `<div class="error"><b>Could not load the study data.</b> ${esc(err.message)}. Stage: loading /data/study.json. Open this page through a web server, not as a file. Retry with: <code>qneuro study serve</code></div>`;
    window.QN = { ready: false, error: String(err) };
  }
}
boot();
})();
