// test_world_core.mjs — unit tests for js/world-core.js (ticket 04). No browser.
//   node --test talk/world/tools/test_world_core.mjs      (WORLD_CORE=<path> tests a mutated copy)
import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import { createRequire } from 'node:module';
const here = path.dirname(new URL(import.meta.url).pathname);
const C = createRequire(import.meta.url)(process.env.WORLD_CORE || path.join(here, '../js/world-core.js'));

test('keyAction: clicker keys move one beat', () => {
  for (const k of [' ', 'ArrowRight', 'PageDown']) assert.equal(C.keyAction(k), 'next', k);
  for (const k of ['ArrowLeft', 'PageUp']) assert.equal(C.keyAction(k), 'prev', k);
});

test('keyAction: A (narrated auto-run) and N (notes rail) are dead in the World', () => {
  for (const k of ['a', 'A', 'n', 'N']) assert.equal(C.keyAction(k), 'block', k);
});

test('keyAction: V toggles the backup clip; every other key is left to the Deck', () => {
  assert.equal(C.keyAction('v'), 'backup'); assert.equal(C.keyAction('V'), 'backup');
  for (const k of ['ArrowUp', 'ArrowDown', 'r', 'p', 'f', '3', 'Enter', 'b', '.']) assert.equal(C.keyAction(k), null, k);
});

test('forwards: the presenter passes clicker, scene and backup keys to the World, nothing else', () => {
  for (const k of [' ', 'ArrowRight', 'PageDown', 'ArrowLeft', 'PageUp', 'ArrowUp', 'ArrowDown', 'v', 'V']) assert.ok(C.forwards(k), k);
  for (const k of ['a', 'n', 'r', 'f', '3', 'F5', 'Tab']) assert.ok(!C.forwards(k), k);
});

test('label: scene index + settled beat count as SNNbK', () => {
  assert.equal(C.label(0, 0), 'S01b0'); assert.equal(C.label(13, 4), 'S14b4');
});

// A 3-scene toy talk: beat counts 2, 4, 1; notes[k-1] belongs to beat k.
const S = [{ title: 'Title', n: 2, notes: ['t1', 't2'] }, { title: 'Why', n: 4, notes: ['w1', 'w2', 'w3', 'w4'] }, { title: 'Refs', n: 1, notes: ['r1'] }];

test('beatView: scene opening shows no note yet; next is beat 1 with its note', () => {
  const v = C.beatView(S, 0, 0, null);
  assert.deepEqual(v.now, { label: 'S01b0', note: null });
  assert.deepEqual(v.next, { label: 'S01b1', note: 't1', title: 'Title' });
});

test('beatView: settled mid-scene beat k shows notes[k-1]; next shows notes[k]', () => {
  const v = C.beatView(S, 1, 2, null);
  assert.deepEqual(v.now, { label: 'S02b2', note: 'w2' });
  assert.deepEqual(v.next, { label: 'S02b3', note: 'w3', title: 'Why' });
});

test('beatView: a running beat already counts as now (the speaker has clicked it)', () => {
  const v = C.beatView(S, 1, 2, 2);
  assert.deepEqual(v.now, { label: 'S02b3', note: 'w3' });
  assert.equal(v.next.label, 'S02b4');
});

test('beatView: after the last beat, next is the following scene opening', () => {
  assert.deepEqual(C.beatView(S, 0, 2, null).next, { label: 'S02b0', note: null, title: 'Why' });
});

test('beatView: the end of the talk has no next', () => {
  const v = C.beatView(S, 2, 1, null);
  assert.deepEqual(v.now, { label: 'S03b1', note: 'r1' }); assert.equal(v.next, null);
});

test('parseBackupParam: 1-based scene numbers in ?backup= become sorted 0-based indices', () => {
  assert.deepEqual(C.parseBackupParam('?backup=5,4', 15), [3, 4]);
  assert.deepEqual(C.parseBackupParam('?x=1&backup=1', 15), [0]);
});

test('parseBackupParam: junk, 0, out-of-range and missing are ignored', () => {
  assert.deepEqual(C.parseBackupParam('?backup=0,16,x,4,4', 15), [3]);
  assert.deepEqual(C.parseBackupParam('', 15), []);
  assert.deepEqual(C.parseBackupParam('?foo=2', 15), []);
});

test('validClip: one mark per settled beat plus the opening, strictly rising', () => {
  assert.equal(C.validClip({ src: 'clips/s04.webm', marks: [0, 1.5, 3] }, 2), true);
  assert.equal(C.validClip({ src: 'clips/s04.webm', marks: [0, 1.5] }, 2), false, 'one mark short');
  assert.equal(C.validClip({ src: 'clips/s04.webm', marks: [0, 2, 2] }, 2), false, 'not rising');
  assert.equal(C.validClip({ src: 'clips/s04.webm', marks: [-1, 1, 2] }, 2), false, 'negative');
  assert.equal(C.validClip({ marks: [0, 1, 2] }, 2), false, 'no src');
  assert.equal(C.validClip(undefined, 2), false, 'no entry');
});

const M = [1, 3, 5, 7]; // a 3-beat scene's clip marks

test('clipPlan: showing the clip (or entering its scene) seeks to the current beat and holds', () => {
  assert.deepEqual(C.clipPlan(null, { si: 3, k: 0 }, M), { op: 'seek', t: 1 });
  assert.deepEqual(C.clipPlan(null, { si: 3, k: 2 }, M), { op: 'seek', t: 5 });
  assert.deepEqual(C.clipPlan({ si: 2, k: 4 }, { si: 3, k: 3 }, M), { op: 'seek', t: 7 });
});

test('clipPlan: same beat does nothing', () => {
  assert.deepEqual(C.clipPlan({ si: 3, k: 1 }, { si: 3, k: 1 }, M), { op: 'none' });
});

test('clipPlan: one beat forward plays on to the next mark and holds there', () => {
  assert.deepEqual(C.clipPlan({ si: 3, k: 0 }, { si: 3, k: 1 }, M), { op: 'play', to: 3 });
  assert.deepEqual(C.clipPlan({ si: 3, k: 2 }, { si: 3, k: 3 }, M), { op: 'play', to: 7 });
});

test('clipPlan: back, or a jump of more than one beat, seeks and holds', () => {
  assert.deepEqual(C.clipPlan({ si: 3, k: 2 }, { si: 3, k: 1 }, M), { op: 'seek', t: 3 });
  assert.deepEqual(C.clipPlan({ si: 3, k: 0 }, { si: 3, k: 3 }, M), { op: 'seek', t: 7 });
});

test('fmtClock: elapsed ms as m:ss, h:mm:ss past the hour', () => {
  assert.equal(C.fmtClock(0), '0:00'); assert.equal(C.fmtClock(65_400), '1:05');
  assert.equal(C.fmtClock(17 * 60_000), '17:00'); assert.equal(C.fmtClock(3_725_000), '1:02:05');
  assert.equal(C.fmtClock(-5), '0:00');
});

test('isWorldMsg: only the two message shapes of this protocol are accepted', () => {
  assert.equal(C.isWorldMsg({ type: 'world-state', v: 1, si: 0, done: 0 }), true);
  assert.equal(C.isWorldMsg({ type: 'world-key', key: 'ArrowRight' }), true);
  for (const d of [null, 'hello', { type: 'world-state', v: 2 }, { type: 'world-key' }, { type: 'other', v: 1 }])
    assert.equal(C.isWorldMsg(d), false, JSON.stringify(d));
});
