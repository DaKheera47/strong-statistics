// Chart rendering functions extracted from main.js (reuse global window.charts)
window.charts = window.charts || {};
function _clearLoading(el){ if(!el) return; const pulse=el.querySelector('.animate-pulse'); if(pulse) pulse.remove(); }
function getChart(id){
  const el=document.getElementById(id);
  if(!el) return null;
  _clearLoading(el);
  if(!el.dataset.fixedHeight){
    if((!el.style.height || el.clientHeight<120)){
      el.style.height = (id==='progressiveOverloadChart'?'300px':'230px');
    }
  }
  if(!window.charts[id]) window.charts[id]=echarts.init(el);
  setTimeout(()=>{ try { window.charts[id].resize(); } catch(_){} }, 40);
  return window.charts[id];
}

function baseTimeAxis(){ return { type:'time', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa', formatter: v=> new Date(v).toISOString().slice(5,10)}, splitLine:{show:false} }; }
function baseValueAxis(name){ return { type:'value', name, nameTextStyle:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}, scale:true }; }

function renderSparklines(){
  const container=document.getElementById('sparklineContainer');
    if(!state.data || !state.data.exercises_daily_max || !state.data.exercises_daily_max.length){
      if(container) container.innerHTML='<div class="text-sm text-zinc-500 italic">No exercise data in range</div>';
      return;
    }
  console.log('[dashboard] renderSparklines count', state.data.exercises_daily_max.length);
    container.innerHTML='';
  const metricKey= 'max_weight';
    const byEx={};
    state.data.exercises_daily_max.forEach(r=>{ (byEx[r.exercise] ||= []).push(r); });
    Object.entries(byEx).forEach(([ex, arr], idx)=>{
      const card=document.createElement('div');
      card.className='bg-zinc-900 rounded-2xl ring-1 ring-zinc-800 shadow-sm p-4 flex flex-col';
      const last = arr[arr.length-1];
      card.innerHTML=`<div class='flex items-center justify-between mb-2'><span class='text-sm font-medium text-zinc-300 truncate'>${ex}</span><span class='text-xs text-zinc-500'>${fmt1(last[metricKey])}</span></div><div class='flex-1' id='spark_${idx}' style='height:60px;'></div>`;
      container.appendChild(card);
      const chart=echarts.init(card.querySelector('#spark_'+idx));
      const prPoints = arr.filter(a=>a.is_pr).map(a=> [a.date, a[metricKey]]);
      chart.setOption({ animation:false, grid:{left:2,right:2,top:0,bottom:0}, xAxis:{type:'time',show:false}, yAxis:{type:'value',show:false}, tooltip:{trigger:'axis', formatter: params=>{
        const p=params[0]; return `${ex}<br>${p.axisValueLabel}: ${fmt1(p.data[1])}`;}}, series:[{type:'line',data:arr.map(a=>[a.date,a[metricKey]]), showSymbol:false, smooth:true, lineStyle:{width:1.2,color:SERIES_COLORS[idx%SERIES_COLORS.length]}, areaStyle:{color:SERIES_COLORS[idx%SERIES_COLORS.length]+'33'}},{type:'scatter', data: prPoints.slice(-8), symbolSize:6, itemStyle:{color:'#fde047'}}] });
    });
}

function renderProgressiveOverload(){
  if(!state.data || !state.data.exercises_daily_max || !state.data.exercises_daily_max.length){ const el=document.getElementById('progressiveOverloadChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No data</div>'; return; }
  const metricKey= 'max_weight';
  const byEx={}; state.data.exercises_daily_max.forEach(r=>{ (byEx[r.exercise] ||= []).push(r); });
  const names = Object.keys(byEx).sort();
  const colorMap={}; names.forEach((n,i)=> colorMap[n]= SERIES_COLORS[i%SERIES_COLORS.length]);
  const activeSet = new Set(names.slice(0,5));
  const legendRoot=document.getElementById('poLegend'); if(legendRoot) legendRoot.innerHTML='';
  names.forEach(name=>{
    const pill=document.createElement('button');
    pill.className='px-2 py-0.5 rounded-full border text-xs flex items-center gap-1 transition-colors';
    pill.dataset.fullName = name;
    pill.textContent = name.split('(')[0].trim();
    const applyStyles = ()=>{
      const on = activeSet.has(name);
      pill.style.borderColor = on? colorMap[name]: '#3f3f46';
      pill.style.background = on? colorMap[name]+'22':'#18181b';
      pill.style.color = on? '#e4e4e7':'#a1a1aa';
      pill.title = (on? 'Hide ':'Show ')+ name;
    };
    applyStyles();
    pill.onclick = ()=>{ if(activeSet.has(name)) activeSet.delete(name); else activeSet.add(name); applyStyles(); renderChart(); };
    legendRoot.appendChild(pill);
  });
  function buildSeries(){
    const series=[];
    names.forEach(name=>{
      if(!activeSet.has(name)) return;
      const arr = byEx[name].slice().sort((a,b)=> a.date.localeCompare(b.date));
  const lname = name.toLowerCase();
  const isCable = lname.includes('cable');
  const isMachine = lname.includes('machine');
  series.push({ name, type:'line', showSymbol:false, smooth:true, data: arr.map(a=> [a.date, a[metricKey]]), lineStyle:{width:2, color: colorMap[name], type: isCable? 'dashed':'solid'}, areaStyle:{color: colorMap[name] + (isMachine? '35':'25')}, symbol: isCable? 'circle':'none', symbolSize: isCable? 5:4 });
      const ma=[]; const vals=[]; arr.forEach(a=>{ vals.push(a[metricKey]); if(vals.length>7) vals.shift(); ma.push([a.date, vals.reduce((s,v)=>s+v,0)/vals.length]); });
      series.push({ name: name+' 7MA', type:'line', showSymbol:false, smooth:true, data: ma, lineStyle:{width:1, type:'dashed', color: colorMap[name]}, emphasis:{disabled:true}, tooltip:{show:false} });
    });
    return series;
  }
  function renderChart(){
    const chart=getChart('progressiveOverloadChart');
    chart.setOption({animationDuration:250, grid:{left:42,right:12,top:10,bottom:55}, legend:{show:false}, dataZoom:[{type:'inside'},{type:'slider',height:16,bottom:18}], xAxis: baseTimeAxis(), yAxis: baseValueAxis('Max Weight (kg)'), tooltip:{trigger:'axis', valueFormatter:v=>fmt1(v)}, series: buildSeries().map(s=> ({...s, emphasis:{focus:'none'}})) }, true);
    chart.off('dataZoom'); chart.on('dataZoom', ()=> updateSlopes(chart, byEx, metricKey));
    updateSlopes(chart, byEx, metricKey);
  }
  document.getElementById('resetOverloadZoom').onclick=()=>{ const chart=getChart('progressiveOverloadChart'); chart.dispatchAction({type:'dataZoom', start:0, end:100}); };
  renderChart();
}

function updateSlopes(chart, byEx, metricKey){
  const opt=chart.getOption(); const [min,max]= opt.xAxis[0].range || [opt.xAxis[0].min, opt.xAxis[0].max];
  const start = min? new Date(min): null; const end = max? new Date(max): null;
  const container=document.getElementById('overloadSlopes'); if(container) container.innerHTML='';
  Object.entries(byEx).forEach(([ex, arr], idx)=>{
  const pts= arr.filter(a=> (!start || window.parseISO(a.date)>=start) && (!end || window.parseISO(a.date)<=end));
  if(pts.length<2) return;
  const t0 = window.parseISO(pts[0].date).getTime();
  const xs = pts.map(p=> (window.parseISO(p.date).getTime()-t0)/ (86400000*7));
    const ys = pts.map(p=> p[metricKey]);
    const mean = xs.reduce((s,v)=>s+v,0)/xs.length; const meanY= ys.reduce((s,v)=>s+v,0)/ys.length;
    let num=0, den=0; for(let i=0;i<xs.length;i++){ const dx=xs[i]-mean; num+= dx*(ys[i]-meanY); den+= dx*dx; }
    const slope = den? num/den:0; // units per week
    const pill=document.createElement('span'); pill.className='px-2 py-1 rounded bg-zinc-800 text-zinc-300'; pill.textContent=`${ex}: ${slope>=0?'+':''}${fmt1(slope)} /wk`; container.appendChild(pill);
  });
}

function renderVolumeTrend(){
  if(!state.data || !state.data.sessions || !state.data.sessions.length){ const el=document.getElementById('volumeTrendChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No sessions</div>'; return; }
  const sessions = state.data.sessions.slice(); sessions.sort((a,b)=> a.date.localeCompare(b.date));
  const seriesBar = sessions.map(s=> [s.date, s.total_volume]);
  const rolling=[]; for(let i=0;i<sessions.length;i++){ const di=window.parseISO(sessions[i].date); const since = di.getTime()-27*86400000; const subset=sessions.filter(s=> window.parseISO(s.date).getTime()>=since && window.parseISO(s.date)<=di); const avg=subset.reduce((s,v)=>s+v.total_volume,0)/subset.length; rolling.push([sessions[i].date, avg]); }
  const chart=getChart('volumeTrendChart');
  chart.setOption({ grid:{left:50,right:16,top:20,bottom:55}, xAxis: baseTimeAxis(), yAxis: baseValueAxis('Volume (kg)'), dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:20}], tooltip:{trigger:'axis'}, series:[{type:'bar', name:'Session Volume', data:seriesBar, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{type:'line', name:'4W Avg', data:rolling, smooth:true, showSymbol:false, lineStyle:{width:2,color:COLORS.secondary}, emphasis:{focus:'none'}}], markLine:{symbol:'none', silent:true, lineStyle:{color:'#3f3f46', width:1}, data: sessions.map(s=> ({xAxis:s.date}))} });
}

function renderWeeklyPPL(){
  if(!state.data || !state.data.weekly_ppl || !state.data.weekly_ppl.length){ const el=document.getElementById('weeklyPPLChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No weekly data</div>'; return; } const mode = document.getElementById('pplModeToggle').dataset.mode; 
  const weeks = state.data.weekly_ppl.map(w=> w.week_start);
  const push = state.data.weekly_ppl.map(w=> w.push);
  const pull = state.data.weekly_ppl.map(w=> w.pull);
  const legs = state.data.weekly_ppl.map(w=> w.legs);
  let pushD=push, pullD=pull, legsD=legs; let yAxis={type:'value', name: mode==='absolute'? 'Weekly Volume (kg)':'% Volume', nameTextStyle:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}};
  if(mode!=='absolute'){
    pushD=[]; pullD=[]; legsD=[];
    for(let i=0;i<weeks.length;i++){ const tot=push[i]+pull[i]+legs[i]; if(tot===0){ pushD.push(0); pullD.push(0); legsD.push(0);} else { pushD.push(push[i]/tot*100); pullD.push(pull[i]/tot*100); legsD.push(legs[i]/tot*100);} }
    yAxis.max=100;
  }
  const chart=getChart('weeklyPPLChart');
  chart.setOption({ grid:{left:50,right:16,top:28,bottom:40}, legend:{top:0,textStyle:{color:'#d4d4d8'}}, tooltip:{trigger:'axis', axisPointer:{type:'shadow'}}, xAxis:{type:'category', data:weeks, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}}, yAxis, series:[{name:'Push', type:'bar', stack:'ppl', data:pushD, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{name:'Pull', type:'bar', stack:'ppl', data:pullD, itemStyle:{color:COLORS.tertiary}, emphasis:{focus:'none'}},{name:'Legs', type:'bar', stack:'ppl', data:legsD, itemStyle:{color:COLORS.quaternary}, emphasis:{focus:'none'}}] });
}

function renderMuscleBalance(){
  if(!state.data || !state.data.muscle_28d){ const el=document.getElementById('muscleBalanceChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No data</div>'; return; } const data = state.data.muscle_28d;
  const names=data.map(d=> d.group); const vals=data.map(d=> d.volume);
  getChart('muscleBalanceChart').setOption({ grid:{left:110,right:30,top:10,bottom:25}, xAxis:{type:'value', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, yAxis:{type:'category', data:names, axisLine:{show:false}, axisLabel:{color:'#d4d4d8'}}, tooltip:{trigger:'item', formatter: p=> `${p.name}: ${fmtInt(p.value)} kg`}, series:[{type:'bar', data:vals, barWidth:'40%', itemStyle:{color:(p)=> SERIES_COLORS[p.dataIndex]}, emphasis:{focus:'none'}, label:{show:true, position:'right', color:'#a1a1aa', formatter: p=> fmtInt(p.value)}}] });
}

function renderRepDistribution(){
  const mode = document.getElementById('repModeToggle').dataset.mode; 
  if(!state.data) return;
  if(mode==='weekly'){
    const weeks = state.data.rep_bins_weekly.map(r=> r.week_start);
    const b1 = state.data.rep_bins_weekly.map(r=> r.bin_1_5);
    const b2 = state.data.rep_bins_weekly.map(r=> r.bin_6_12);
    const b3 = state.data.rep_bins_weekly.map(r=> r.bin_13_20);
    getChart('repDistributionChart').setOption({ grid:{left:55,right:16,top:28,bottom:40}, legend:{top:0,textStyle:{color:'#d4d4d8'}}, tooltip:{trigger:'axis', axisPointer:{type:'shadow'}}, xAxis:{type:'category', data:weeks, axisLabel:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}}, yAxis:{type:'value', name:'Volume (kg)', nameTextStyle:{color:'#a1a1aa'}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, series:[{name:'1–5', type:'bar', stack:'reps', data:b1, itemStyle:{color:COLORS.secondary}, emphasis:{focus:'none'}},{name:'6–12', type:'bar', stack:'reps', data:b2, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{name:'13–20', type:'bar', stack:'reps', data:b3, itemStyle:{color:COLORS.tertiary}, emphasis:{focus:'none'}}] });
  } else {
    const t = state.data.rep_bins_total; const total=t.total||1; const bars=[{name:'1–5', val:t.bin_1_5},{name:'6–12', val:t.bin_6_12},{name:'13–20', val:t.bin_13_20}];
    getChart('repDistributionChart').setOption({ grid:{left:110,right:30,top:10,bottom:25}, xAxis:{type:'value', axisLabel:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, splitLine:{lineStyle:{color:'#27272a'}}}, yAxis:{type:'category', data:bars.map(b=> b.name), axisLabel:{color:'#d4d4d8'}}, tooltip:{trigger:'item', formatter: p=>{ const v=bars[p.dataIndex].val; return `${p.name}: ${fmtInt(v)} kg (${(v/total*100).toFixed(1)}%)`; }}, series:[{type:'bar', data:bars.map(b=> b.val), itemStyle:{color:(p)=> [COLORS.secondary,COLORS.primary,COLORS.tertiary][p.dataIndex]}, emphasis:{focus:'none'}, barWidth:'45%', label:{show:true, position:'right', formatter: p=> (bars[p.dataIndex].val/total*100).toFixed(1)+'%', color:'#a1a1aa'}}] });
  }
}

// Exported single function to render all echarts-based charts
function renderAll(){
  renderSparklines();
  renderProgressiveOverload();
  renderVolumeTrend();
  // NEW: exercise volume chart (stacked/grouped)
  try { if(typeof renderExerciseVolume === 'function') renderExerciseVolume(); } catch(e){ console.warn('renderExerciseVolume failed', e); }
  renderWeeklyPPL();
  renderMuscleBalance();
  renderRepDistribution();
}

// Expose
window.getChart = getChart;
window.renderAllCharts = renderAll;

// Attach volume mode toggle handler (was missing so chart never appeared)
document.addEventListener('DOMContentLoaded', ()=>{
  const btn = document.getElementById('volumeModeToggle');
  if(btn && !btn.dataset._bound){
    btn.dataset._bound = '1';
    btn.addEventListener('click', function(){
      this.dataset.mode = this.dataset.mode === 'stacked' ? 'grouped' : 'stacked';
      this.textContent = this.dataset.mode === 'stacked' ? 'Stacked' : 'Grouped';
      if(typeof renderExerciseVolume === 'function') renderExerciseVolume();
    });
  }
});
