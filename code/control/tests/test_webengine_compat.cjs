// Exercise the actual injected helpers with replaceAll deliberately absent.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../qt_gui.py'), 'utf8');
const html = fs.readFileSync(path.join(__dirname, '../../pam_shoulder_gui.html'), 'utf8');
assert.ok(!/\.replaceAll\s*\(/.test(source + html));
const calls = [];
const context = {
  qtOpEffectiveLanguage: () => 'en',
  qtOpTranslations: {en: {repeat: '{n}, {n}, {other}'}, zh: {fallback: 'fallback {n}'}},
  tcSay: function(...args) {calls.push({args, receiver: this}); return 'original-result';},
};
context.window = context;
vm.createContext(context);
vm.runInContext('String.prototype.replaceAll = undefined;', context);
const textStart = source.indexOf('const qtOpText = (key, vars={}) => {');
const textEndMarker = 'window.__qtOpText = qtOpText;';
const textEnd = source.indexOf(textEndMarker, textStart) + textEndMarker.length;
vm.runInContext(source.slice(textStart, textEnd), context);
assert.equal(context.__qtOpText('repeat', {n: '$&', other: 0}), '$&, $&, 0');
assert.equal(context.__qtOpText('fallback', {n: 3}), 'fallback 3');
assert.equal(context.__qtOpText('unknown'), 'unknown');
const sayStart = source.indexOf('if (!window.__qtNoMinVoltageText');
const sayEnd = source.indexOf('if (!window.__qtMpcMountFetch', sayStart);
vm.runInContext(source.slice(sayStart, sayEnd), context);
const receiver = {};
assert.equal(context.tcSay.call(receiver,
  '（&lt;60 固件视为 0）≤2V；&lt;60 视为 0 ≤2V', true), 'original-result');
assert.deepEqual(calls[0].args, ['≤2.2V ≤2.2V', true]);
assert.equal(calls[0].receiver, receiver);
context.tcSay(null);
assert.equal(calls[1].args[0], 'null');
console.log('PASS: localization placeholders and status messages without String.replaceAll.');
