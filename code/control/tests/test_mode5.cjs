// Offline scheduling tests: no serial port or GUI is opened.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(require('node:path').join(__dirname,'../qt_gui.py'),'utf8');
const start = source.indexOf('const installFixedOpenLoop = () => {');
const end = source.indexOf('installFixedOpenLoop();',start)+'installFixedOpenLoop();'.length;
const installer = source.slice(start,end);
const fullJs = source.split('runJavaScript(r"""')[1].split('""")')[0];
new vm.Script(fullJs);
function harness(modeName, config={loop:false}) {
  let now=0, callback=null;
  const sent=[];
  const element=()=>({style:{},value:'',textContent:'',disabled:false,
    onclick:()=>{},addEventListener:()=>{},setAttribute:()=>{},
    querySelector:()=>({}),closest:()=>({insertAdjacentElement:()=>{}})});
  const elements=Object.fromEntries(['tc-play','tc-drvmode','qt-open-fixed-mode',
    'qt-open-fixed-note','tc-drv'].map(id=>[id,element()]));
  elements['tc-drvmode'].value='open';
  elements['qt-open-fixed-mode'].value=modeName;
  const context={console, Math, Number, Infinity,
    document:{getElementById:id=>elements[id],createElement:element},
    performance:{now:()=>now},imuOn:true,imuSend:line=>sent.push(line),
    imuMeasS:{sx:0,sy:0,psi:0},tcDrive:true,tcPlaying:false,DEBUG:false,
    tcStop:()=>{},tcSay:()=>{},__qtGetFixedOpenConfig:()=>config,
    setTimeout:()=>{throw Error('Unexpected retry');},
    setInterval:fn=>{callback=fn;return 1;},clearInterval:()=>{callback=null;}};
  context.window=context;
  vm.createContext(context);vm.runInContext(installer,context);
  elements['tc-play'].onclick();
  return {sent,context,elements,at:t=>{now=t*1000;if(callback)callback();
    return sent.at(-1);},running:()=>context.__qtFixedOpenIsRunning()};
}
const output=line=>line.slice(1).trim().split(',').map(Number);
let h=harness('mode5',{loop:true,limitValue:2,limitUnit:'runs'});
assert.ok(h.sent[0].startsWith('C'));
assert.equal(output(h.at(0.4))[1],0);
assert.deepEqual(output(h.at(0.75)).slice(0,2),[1500,1500]);
assert.equal(output(h.at(1.0))[1],3000);
assert.deepEqual(output(h.at(1.5)).slice(0,2),[3000,3000]);
assert.deepEqual(output(h.at(2.0)).slice(0,2),[1500,1500]);
for(const t of [2.5,4.0,5.5,7.5,9.0,10.5,12.0]) {
  const index=Math.round((t%6.5-1)/1.5);
  const primary=[0,2,1,3][index];
  const u=output(h.at(t));
  assert.equal(u[primary^1],3000,'opposite plateau '+t);
  assert.ok(u.every(v=>v>=0 && v<=3000));
}
// A fresh scheduler verifies the boundary tail is not truncated.
h=harness('mode5',{loop:true,limitValue:2,limitUnit:'runs'});
let u=output(h.at(6.75));
assert.ok(u[0]>0 && u[2]>0 && u[3]>0);
assert.ok(h.running());
h.at(13.49);assert.ok(h.running());
assert.deepEqual(output(h.at(13.5)),[0,0,0,0,0,0]);assert.ok(!h.running());
// Single-run and minute-limited operation complete the final 1 s fall.
for(const cfg of [{loop:false},{loop:true,limitValue:0.1,limitUnit:'min'}]) {
  h=harness('mode5',cfg);h.at(6.0);h.at(6.75);assert.ok(h.running());
  assert.deepEqual(output(h.at(7.0)),[0,0,0,0,0,0]);assert.ok(!h.running());
}
h=harness('mode5',{loop:true});h.at(0.75);h.context.tcStop();
assert.ok(!h.running());assert.deepEqual(output(h.sent.at(-1)),[0,0,0,0,0,0]);
h=harness('mode3');assert.ok(h.at(0.75).startsWith('O'));
assert.equal(output(h.sent.at(-1))[1],0);
h=harness('mode1');assert.ok(h.sent[0].startsWith('M'));
assert.ok(source.includes("id=\"qt-op-m5-loop\""));
console.log('PASS: injected JS syntax, Mode 5 timing/pairs/loops/runs/minutes/stop, Mode 3/1 regression.');
