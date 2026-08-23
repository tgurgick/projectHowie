// node --test tests/ui — pure-function tests for the cockpit helpers.
import test from 'node:test';
import assert from 'node:assert/strict';
import {createRequire} from 'node:module';
const require = createRequire(import.meta.url);
const {esc, h, raw, textHtml, classifyInput, pickAction, availClass, fmtDelta, reasonLine} = require('../../howie3/ui/lib.js');

test('esc neutralizes markup and quotes', () => {
  assert.equal(esc('<img src=x onerror="alert(1)">'), '&lt;img src=x onerror=&quot;alert(1)&quot;&gt;');
  assert.equal(esc("O'Neil & Sons"), 'O&#39;Neil &amp; Sons');
  assert.equal(esc(null), ''); assert.equal(esc(undefined), ''); assert.equal(esc(12.5), '12.5');
});

test('h escapes interpolations, passes Raw and nested templates, joins arrays', () => {
  const name = '<b>x</b>';
  assert.equal(h`<td>${name}</td>`.s, '<td>&lt;b&gt;x&lt;/b&gt;</td>');
  assert.equal(h`<p>${raw('<i>ok</i>')}</p>`.s, '<p><i>ok</i></p>');
  assert.equal(h`<ul>${[1, 2].map(n => h`<li>${n}</li>`)}</ul>`.s, '<ul><li>1</li><li>2</li></ul>');
  assert.equal(h`<p>${['<a>', raw('<b>')]}</p>`.s, '<p>&lt;a&gt;<b></p>');
  assert.equal(h`${undefined}${null}${''}`.s, '');
  // attribute context: a quote in a research fact cannot break out
  assert.equal(h`<span title="${'" onmouseover="x'}">`.s, '<span title="&quot; onmouseover=&quot;x">');
  assert.equal(String(h`a${1}`), 'a1');
});

test('textHtml escapes then breaks lines', () => {
  assert.equal(textHtml('a<b\nc').s, 'a&lt;b<br>c');
});

test('classifyInput: commands, questions, players, no match', () => {
  const items = [{uid: 'u1', name: 'Bijan Robinson', kind: 'player'}, {uid: 'u2', name: 'Brian Robinson Jr.', kind: 'player'}, {id: 't', name: 'Dallas Cowboys', kind: 'team'}];
  assert.deepEqual(classifyInput('/mine bijan', items, -1), {kind: 'cmd', cmd: 'mine', rest: ['bijan'], arg: 'bijan'});
  assert.equal(classifyInput('?who should I take', items, -1).kind, 'ask');
  assert.equal(classifyInput('?who should I take', items, -1).question, 'who should I take');
  assert.equal(classifyInput('howie, is Bijan worth it', items, -1).kind, 'ask');
  assert.equal(classifyInput('is Bijan worth a first round pick?', items, -1).kind, 'ask');
  assert.equal(classifyInput('bijan robinson', items, -1).hit.uid, 'u1');         // exact, case-insensitive
  assert.equal(classifyInput('bijan', items, -1).hit.uid, 'u1');                  // unique prefix
  assert.equal(classifyInput('xyz', items, 1).hit.uid, 'u2');                     // arrow-key selection wins
  assert.equal(classifyInput('b', items, -1).kind, 'nomatch');                    // too short / ambiguous
  assert.equal(classifyInput('robinson', items, -1).kind, 'nomatch');             // two candidates
  assert.deepEqual(classifyInput('robinson', items, -1).suggestions, ['Bijan Robinson', 'Brian Robinson Jr.']);
  assert.equal(classifyInput('zzz', [], -1).kind, 'nomatch');
  assert.equal(classifyInput('  ', [], -1).kind, 'empty');
  const withTeams = [...items, {id: 'team:DAL', kind: 'team', name: 'Dallas Cowboys', team: 'DAL'}, {id: 'unit:DAL-WR', kind: 'unit', name: 'DAL WR room', team: 'DAL'}];
  assert.deepEqual(classifyInput('dallas cowboys', withTeams, -1).team, 'DAL');   // team name -> team report
  assert.equal(classifyInput('dal', [{id: 'team:DAL', kind: 'team', name: 'Dallas Cowboys', team: 'DAL'}], -1).team, 'DAL'); // abbreviation
  assert.equal(classifyInput('cowboys', [{id: 'team:DAL', kind: 'team', name: 'Dallas Cowboys', team: 'DAL'}, {id: 'unit:DAL-WR', kind: 'unit', name: 'DAL WR room', team: 'DAL'}], -1).team, 'DAL'); // unique team across hits
  assert.equal(classifyInput('xyz', withTeams, 3).kind, 'team');                  // arrow-key selection of a team row
  assert.equal(classifyInput('robinson', withTeams, -1).kind, 'nomatch');         // players still ambiguous
  assert.equal(classifyInput('dallas cowboys', items, -1).kind, 'nomatch');       // no team hits -> nomatch
});

test('pickAction: Enter is taken while drafting, card otherwise; shift is mine', () => {
  const k = (o) => ({shiftKey: false, altKey: false, metaKey: false, ctrlKey: false, ...o});
  assert.equal(pickAction(k({}), true), 'taken');
  assert.equal(pickAction(k({}), false), 'card');
  assert.equal(pickAction(k({shiftKey: true}), true), 'mine');
  assert.equal(pickAction(k({shiftKey: true}), false), 'mine');
  assert.equal(pickAction(k({altKey: true}), true), 'card');
});

test('small formatters', () => {
  assert.equal(availClass(0.8), 'acc'); assert.equal(availClass(0.4), 'amber'); assert.equal(availClass(0.1), 'red');
  assert.equal(fmtDelta(3.2), '+3'); assert.equal(fmtDelta(-0.4), '0'); assert.equal(fmtDelta(-9), '-9');
  const line = reasonLine({pos: 'RB', mv_vs_wait: 12.4, avail_next: 0.64, next_pick: 41}, {positions: {RB: {level: 'danger', reasons: ['x']}}});
  assert.equal(line, '+12 vs waiting at RB · 64% there at 41 · RB THIN');
  assert.equal(reasonLine({pos: 'WR'}, null), '');
});
