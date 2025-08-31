// Refactored progression-focused dashboard JS (ECharts + Tailwind)
// Only required 7 charts + filters.

// ------------------------------ State ----------------------------------
const state = {
  start: null, // date filters removed (backend auto range) but keep keys for cache key stability
  end: null,
  exercises: [], // applies ONLY to sparklines
  data: null,
  cache: new Map()
};

const COLORS = {
  primary: '#6366F1',
  secondary: '#EC4899',
  tertiary: '#10B981',
  quaternary: '#F59E0B',
  quinary: '#8B5CF6'
};
const SERIES_COLORS = [COLORS.primary, COLORS.secondary, COLORS.tertiary, COLORS.quaternary, COLORS.quinary];

function limitLegendSelection(series, maxVisible){
  const sel={};
  let count=0;
  series.forEach(s=>{
    if(!s.name.endsWith(' 7MA') && count<maxVisible){ sel[s.name]=true; count++; } else { sel[s.name]=false; }
  });
  return sel;
}

function fetchJSON(url) { return fetch(url).then(r => { if(!r.ok) throw new Error(r.statusText); return r.json(); }); }
function fmtInt(x){ return x == null ? '-' : x.toLocaleString(); }
function fmt1(x){ return x==null?'-': (Math.round(x*10)/10).toString(); }
function parseISO(d){ return new Date(d+ (d.length===10?'T00:00:00Z':'')); }

// --------------------------- Filters UI --------------------------------
function initFilters(){
  // Only metric placeholder remains (weight-only)
  const metricWrap=document.getElementById('metricToggle');
  metricWrap.innerHTML='<span class="text-xs text-zinc-500">Weight mode</span>';
}

function updateMetricButtons(){}

// Date range presets removed

// Exercise multi-select (simple dropdown)
function initExerciseMulti(exNames){
  const root=document.getElementById('exerciseMulti');
  root.innerHTML='';
  const btn=document.createElement('button');
  btn.className='px-3 py-1.5 rounded-md bg-zinc-800 text-sm';
  btn.textContent='Exercises ▾';
  const panel=document.createElement('div');
  panel.className='absolute z-20 mt-2 w-64 max-h-72 overflow-auto bg-zinc-900 ring-1 ring-zinc-800 rounded-lg shadow-lg p-2 hidden';
  exNames.forEach(name=>{
    const id= 'ex_'+btoa(name).replace(/=/g,'');
    const label=document.createElement('label');
    label.className='flex items-center gap-2 px-2 py-1 rounded hover:bg-zinc-800 text-xs';
    label.innerHTML=`<input type="checkbox" class="accent-indigo-600" id="${id}" value="${name}" ${state.exercises.includes(name)?'checked':''}/> <span>${name}</span>`;
    panel.appendChild(label);
  });
  btn.addEventListener('click',()=> panel.classList.toggle('hidden'));
  root.appendChild(btn);root.appendChild(panel);
  panel.addEventListener('change',()=>{
    state.exercises=[...panel.querySelectorAll('input:checked')].map(i=>i.value);
    refreshData();
  });
  document.addEventListener('click',e=>{ if(!root.contains(e.target)) panel.classList.add('hidden'); });
}

// ------------------------- Data Fetch & Cache ---------------------------
async function fetchDashboard(){
  const key = JSON.stringify({start:state.start,end:state.end});
  if(state.cache.has(key)) return state.cache.get(key);
  const params = new URLSearchParams();
  if(state.start) params.set('start', state.start);
  if(state.end) params.set('end', state.end);
  // exercises & metric no longer passed to backend
  const data = await fetchJSON('/api/dashboard?'+params.toString());
  state.cache.set(key,data); return data;
}

async function refreshData(){
  try {
  console.log('[dashboard] refreshData start', {exercises:state.exercises});
    const loadingTargets=['sparklineContainer','progressiveOverloadChart','volumeTrendChart','weeklyPPLChart','muscleBalanceChart','repDistributionChart','recoveryChart','calendarChart'];
    loadingTargets.forEach(id=>{ const el=document.getElementById(id); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 animate-pulse">Loading...</div>'; });
    const data = await fetchDashboard();
  window.__dashboardDebug = { phase:'afterFetch', fetchedAt: Date.now(), filters: data?.filters, params:{start:state.start,end:state.end,exercises:[...state.exercises]}, keys: data? Object.keys(data):[] };
  console.log('[dashboard] data fetched', window.__dashboardDebug);
  // No date preset logic
  state.data=data;
    document.getElementById('lastIngested').textContent = data.filters.end || '-';
    if(!state.exercises.length){
      // Preselect by most improvement (delta)
      const prog = (data.exercise_progression || []).map(p=> p.exercise);
      state.exercises = prog.slice(0,12);
      if(state.exercises.length===0) state.exercises = data.filters.exercises || (data.top_exercises || []);
      initExerciseMulti(unique(state.data.exercises_daily_max.map(d=>d.exercise)));
    }
  renderAll();
  window.__dashboardDebug.phase='renderComplete';
  console.log('[dashboard] render complete');
  } catch(e){
    console.error(e);
    const msg='<div class="flex items-center justify-center h-full text-sm text-rose-400">Error loading data</div>';
    ['sparklineContainer','progressiveOverloadChart','volumeTrendChart','weeklyPPLChart','muscleBalanceChart','repDistributionChart','recoveryChart','calendarChart'].forEach(id=>{ const el=document.getElementById(id); if(el) el.innerHTML=msg; });
  window.__dashboardDebug = { phase:'error', error: e?.message || String(e) };
  }
}

// Range helper removed

function unique(arr){ return [...new Set(arr)]; }

// ------------------------------ Charts ----------------------------------
let charts = {};
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
  if(!charts[id]) charts[id]=echarts.init(el);
  setTimeout(()=>{ try { charts[id].resize(); } catch(_){} }, 40);
  return charts[id];
}

function baseTimeAxis(){ return { type:'time', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa', formatter: v=> new Date(v).toISOString().slice(5,10)}, splitLine:{show:false} }; }
function baseValueAxis(name){ return { type:'value', name, nameTextStyle:{color:'#a1a1aa'}, axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}, scale:true }; }

function renderSparklines(){
  const container=document.getElementById('sparklineContainer');
    if(!state.data || !state.data.exercises_daily_max || !state.data.exercises_daily_max.length){
      container.innerHTML='<div class="text-sm text-zinc-500 italic">No exercise data in range</div>';
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
  // Determine palette (repeat but shift for variety)
  const colorMap={}; names.forEach((n,i)=> colorMap[n]= SERIES_COLORS[i%SERIES_COLORS.length]);
  const activeSet = new Set(names.slice(0,5)); // default first 5 visible
  // Build legend pills
  const legendRoot=document.getElementById('poLegend'); legendRoot.innerHTML='';
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
  const container=document.getElementById('overloadSlopes'); container.innerHTML='';
  Object.entries(byEx).forEach(([ex, arr], idx)=>{
    const pts= arr.filter(a=> (!start || parseISO(a.date)>=start) && (!end || parseISO(a.date)<=end));
    if(pts.length<2) return;
    // Linear regression
    const t0 = parseISO(pts[0].date).getTime();
    const xs = pts.map(p=> (parseISO(p.date).getTime()-t0)/ (86400000*7));
    const ys = pts.map(p=> p[metricKey]);
    const mean = xs.reduce((s,v)=>s+v,0)/xs.length; const meanY= ys.reduce((s,v)=>s+v,0)/ys.length;
    let num=0, den=0; for(let i=0;i<xs.length;i++){ const dx=xs[i]-mean; num+= dx*(ys[i]-meanY); den+= dx*dx; }
    const slope = den? num/den:0; // units per week
    const pill=document.createElement('span'); pill.className='px-2 py-1 rounded bg-zinc-800 text-zinc-300'; pill.textContent=`${ex}: ${slope>=0?'+':''}${fmt1(slope)} /wk`; container.appendChild(pill);
  });
}

function renderVolumeTrend(){
  if(!state.data || !state.data.sessions || !state.data.sessions.length){ const el=document.getElementById('volumeTrendChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No sessions</div>'; return; } // compute 4-week rolling avg client side
  console.log('[dashboard] renderVolumeTrend sessions', state.data.sessions.length);
  const sessions = state.data.sessions.slice(); sessions.sort((a,b)=> a.date.localeCompare(b.date));
  const seriesBar = sessions.map(s=> [s.date, s.total_volume]);
  const rolling=[]; for(let i=0;i<sessions.length;i++){ const di=parseISO(sessions[i].date); const since = di.getTime()-27*86400000; const subset=sessions.filter(s=> parseISO(s.date).getTime()>=since && parseISO(s.date)<=di); const avg=subset.reduce((s,v)=>s+v.total_volume,0)/subset.length; rolling.push([sessions[i].date, avg]); }
  const mondays = sessions.map(s=> s.date).filter(d=> parseISO(d).getUTCDay()===1);
  const chart=getChart('volumeTrendChart');
  chart.setOption({ grid:{left:50,right:16,top:20,bottom:55}, xAxis: baseTimeAxis(), yAxis: baseValueAxis('Volume (kg)'), dataZoom:[{type:'inside'},{type:'slider',height:18,bottom:20}], tooltip:{trigger:'axis'}, series:[{type:'bar', name:'Session Volume', data:seriesBar, itemStyle:{color:COLORS.primary}, emphasis:{focus:'none'}},{type:'line', name:'4W Avg', data:rolling, smooth:true, showSymbol:false, lineStyle:{width:2,color:COLORS.secondary}, emphasis:{focus:'none'}}], markLine:{symbol:'none', silent:true, lineStyle:{color:'#3f3f46', width:1}, data: mondays.map(m=> ({xAxis:m}))} });
}

function renderWeeklyPPL(){
  if(!state.data || !state.data.weekly_ppl || !state.data.weekly_ppl.length){ const el=document.getElementById('weeklyPPLChart'); if(el) el.innerHTML='<div class="flex items-center justify-center h-full text-sm text-zinc-500 italic">No weekly data</div>'; return; } const mode = document.getElementById('pplModeToggle').dataset.mode; // absolute|percent
  console.log('[dashboard] renderWeeklyPPL weeks', state.data.weekly_ppl.length);
  const weeks = state.data.weekly_ppl.map(w=> w.week_start); // show ISO week start date (YYYY-MM-DD)
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
  console.log('[dashboard] renderMuscleBalance entries', data.length);
  const names=data.map(d=> d.group); const vals=data.map(d=> d.volume);
  getChart('muscleBalanceChart').setOption({ grid:{left:110,right:30,top:10,bottom:25}, xAxis:{type:'value', axisLine:{lineStyle:{color:'#3f3f46'}}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, yAxis:{type:'category', data:names, axisLine:{show:false}, axisLabel:{color:'#d4d4d8'}}, tooltip:{trigger:'item', formatter: p=> `${p.name}: ${fmtInt(p.value)} kg`}, series:[{type:'bar', data:vals, barWidth:'40%', itemStyle:{color:(p)=> SERIES_COLORS[p.dataIndex]}, emphasis:{focus:'none'}, label:{show:true, position:'right', color:'#a1a1aa', formatter: p=> fmtInt(p.value)}}] });
}

function renderRepDistribution(){
  const mode = document.getElementById('repModeToggle').dataset.mode; // weekly|summary if(!state.data){ return; }
  console.log('[dashboard] renderRepDistribution mode', mode);
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

// Recovery chart removed

function renderAll(){
  renderSparklines();
  renderProgressiveOverload();
  renderVolumeTrend();
  renderWeeklyPPL();
  renderMuscleBalance();
  renderRepDistribution();
  loadTrainingCalendar(); // Add calendar loading
  // recovery removed
}

// Toggle handlers
document.getElementById('pplModeToggle').addEventListener('click', function(){ this.dataset.mode = this.dataset.mode==='absolute' ? 'percent':'absolute'; this.textContent= this.dataset.mode==='absolute'?'Absolute':'Percent'; renderWeeklyPPL(); });
document.getElementById('repModeToggle').addEventListener('click', function(){ this.dataset.mode = this.dataset.mode==='weekly' ? 'summary':'weekly'; this.textContent= this.dataset.mode==='weekly'?'Weekly':'Summary'; renderRepDistribution(); });

// Init
function bootstrapDashboard(){
  console.log('[dashboard] bootstrap');
  initFilters();
  updateMetricButtons();
  refreshData();
}
if(document.readyState === 'loading'){
  document.addEventListener('DOMContentLoaded', bootstrapDashboard);
} else {
  // DOM already parsed
  bootstrapDashboard();
}

window.addEventListener('resize', ()=>{ Object.values(charts).forEach(c=> c.resize()); });
function renderExerciseVolume(){
  if(!state.data || !state.data.exercises_daily_volume) return;
  const mode = document.getElementById('volumeModeToggle').dataset.mode; // stacked|grouped
  const volsByEx={};
  state.data.exercises_daily_volume.forEach(r=> { volsByEx[r.exercise] = (volsByEx[r.exercise]||0)+ r.volume; });
  const top = Object.entries(volsByEx).sort((a,b)=> b[1]-a[1]).slice(0,6).map(e=> e[0]);
  const filtered = state.data.exercises_daily_volume.filter(r=> top.includes(r.exercise));
  const dates = Array.from(new Set(filtered.map(r=> r.date))).sort();
  const series = top.map((ex,i)=>{
    const data = dates.map(d=> { const rec = filtered.find(r=> r.exercise===ex && r.date===d); return rec? rec.volume:0; });
    return { name: ex.split('(')[0].trim(), type:'bar', stack: mode==='stacked'? 'vol': undefined, data, itemStyle:{color: SERIES_COLORS[i%SERIES_COLORS.length]}, emphasis:{focus:'none'} };
  });
  const chart=getChart('exerciseVolumeChart');
  chart.setOption({ grid:{left:50,right:12,top:30,bottom:55}, legend:{top:0,textStyle:{color:'#d4d4d8'}}, tooltip:{trigger:'axis', axisPointer:{type:'shadow'}}, xAxis:{type:'category', data:dates, axisLabel:{color:'#a1a1aa', formatter:v=> v.slice(5)}, axisLine:{lineStyle:{color:'#3f3f46'}}}, yAxis:{type:'value', name:'Volume (kg)', nameTextStyle:{color:'#a1a1aa'}, axisLabel:{color:'#a1a1aa'}, splitLine:{lineStyle:{color:'#27272a'}}}, series });
}

async function loadPersonalRecords() {
  try {
    const data = await fetchJSON('/api/personal-records');
    
    if (!data || data.length === 0) return;

    // Group PRs by exercise for better visualization
    const exerciseGroups = {};
    data.forEach(pr => {
      if (!exerciseGroups[pr.exercise]) {
        exerciseGroups[pr.exercise] = [];
      }
      exerciseGroups[pr.exercise].push(pr);
    });

    // Create scatter plot showing PR timeline
    const traces = [];
    let colorIndex = 0;

    Object.entries(exerciseGroups).forEach(([exercise, prs]) => {
      traces.push({
        x: prs.map(pr => pr.date),
        y: prs.map(pr => pr.weight),
        type: 'scatter',
        mode: 'markers+text',
        name: exercise,
        marker: { 
          size: 12, 
          color: exerciseColors[colorIndex % exerciseColors.length],
          symbol: 'star'
        },
        text: prs.map(pr => `${pr.weight}kg`),
        textposition: 'top center',
        hovertemplate: `<b>${exercise}</b><br>` +
                      `Date: %{x}<br>` +
                      `Weight: %{y} kg<br>` +
                      `<extra></extra>`
      });
      colorIndex++;
    });

    const layout = {
      title: 'Personal Records Achievement',
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { title: 'Weight (kg)' },
      hovermode: 'closest',
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('personalRecordsChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading personal records:', error);
  }
}

// Training Consistency Chart
async function loadTrainingConsistency() {
  try {
    const data = await fetchJSON('/api/training-consistency');
    
    if (!data || !data.weekly || data.weekly.length === 0) return;

    const traces = [
      {
        x: data.weekly.map(d => d.week),
        y: data.weekly.map(d => d.workouts),
        type: 'bar',
        name: 'Workouts/Week',
        marker: { color: colors.success },
        hovertemplate: 'Week: %{x}<br>Workouts: %{y}<extra></extra>'
      }
    ];

    // Add target line (e.g., 3 workouts per week)
    const targetWorkouts = 3;
    traces.push({
      x: data.weekly.map(d => d.week),
      y: Array(data.weekly.length).fill(targetWorkouts),
      type: 'scatter',
      mode: 'lines',
      name: 'Target (3/week)',
      line: { dash: 'dash', color: colors.danger, width: 2 },
      hovertemplate: 'Target: %{y} workouts/week<extra></extra>'
    });

    const layout = {
      title: 'Weekly Workout Frequency',
      xaxis: { 
        title: 'Week',
        tickangle: -45
      },
      yaxis: { 
        title: 'Workouts per Week',
        dtick: 1
      },
      hovermode: 'x unified'
    };

    Plotly.newPlot('trainingConsistencyChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading training consistency:', error);
  }
}

// Strength Balance Chart
async function loadStrengthBalance() {
  try {
    const data = await fetchJSON('/api/strength-balance');
    
    if (!data || Object.keys(data).length === 0) return;

    const traces = [];
    let colorIndex = 0;

    // Show estimated 1RM progression for each movement pattern
    Object.entries(data).forEach(([pattern, sessions]) => {
      if (sessions.length > 0) {
        traces.push({
          x: sessions.map(s => s.date),
          y: sessions.map(s => s.estimated_1rm),
          type: 'scatter',
          mode: 'lines+markers',
          name: pattern,
          line: { width: 2 },
          marker: { size: 6 },
          hovertemplate: `<b>${pattern}</b><br>` +
                        `Date: %{x}<br>` +
                        `Est. 1RM: %{y:.1f} kg<br>` +
                        `<extra></extra>`
        });
      }
      colorIndex++;
    });

    const layout = {
      title: 'Movement Pattern Balance',
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { 
        title: 'Estimated 1RM (kg)'
      },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('strengthBalanceChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading strength balance:', error);
  }
}

// Exercise Analysis (detailed view for selected exercise)
async function loadExerciseAnalysis(exercise) {
  try {
    if (!exercise) {
      Plotly.newPlot('exerciseAnalysisChart', [], {
        title: 'Select an exercise above for detailed analysis',
        font: { size: 14 }
      });
      return;
    }

    const data = await fetchJSON(`/api/exercise-analysis?exercise=${encodeURIComponent(exercise)}`);
    
    if (!data || !data.session_summary || data.session_summary.length === 0) {
      Plotly.newPlot('exerciseAnalysisChart', [], {
        title: `No data available for ${exercise}`,
        font: { size: 14 }
      });
      return;
    }

    const traces = [
      // Max weight progression
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.max_weight),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Max Weight',
        yaxis: 'y',
        line: { width: 3, color: colors.primary },
        marker: { size: 8 },
        hovertemplate: 'Date: %{x}<br>Max Weight: %{y} kg<extra></extra>'
      },
      // Session volume bars
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.session_volume),
        type: 'bar',
        name: 'Session Volume',
        yaxis: 'y2',
        opacity: 0.6,
        marker: { color: colors.info },
        hovertemplate: 'Date: %{x}<br>Volume: %{y} kg<extra></extra>'
      },
      // Estimated 1RM
      {
        x: data.session_summary.map(s => s.date),
        y: data.session_summary.map(s => s.estimated_1rm),
        type: 'scatter',
        mode: 'lines+markers',
        name: 'Est. 1RM',
        yaxis: 'y',
        line: { width: 2, color: colors.success, dash: 'dot' },
        marker: { size: 6 },
        hovertemplate: 'Date: %{x}<br>Est. 1RM: %{y:.1f} kg<extra></extra>'
      }
    ];

    const layout = {
      title: `${exercise} - Detailed Progress Analysis`,
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { 
        title: 'Weight (kg)',
        side: 'left'
      },
      yaxis2: { 
        title: 'Volume (kg)',
        side: 'right',
        overlaying: 'y'
      },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.15 }
    };

    Plotly.newPlot('exerciseAnalysisChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading exercise analysis:', error);
  }
}

// Load exercises list for dropdown
async function loadExerciseOptions() {
  try {
    const exercises = await fetchJSON('/api/exercises');
    const select = document.getElementById('exerciseSelect');
    
    select.innerHTML = '<option value="">-- Select for detailed analysis --</option>' + 
      exercises.map(e => `<option value="${e}">${e}</option>`).join('');
    
    select.addEventListener('change', () => {
      if (select.value) {
        loadExerciseAnalysis(select.value);
      } else {
        loadExerciseAnalysis(null);
      }
    });

  } catch (error) {
    console.error('Error loading exercises list:', error);
  }
}

// Strong-inspired analytics functions

async function loadPersonalRecordsTable() {
  try {
    const data = await fetchJSON('/api/records');
    
    const container = document.getElementById('recordsTable');
    
    if (!data || data.length === 0) {
      container.innerHTML = '<p style="text-align: center; padding: 20px;">No records found</p>';
      return;
    }
    
    let html = `
      <table style="width: 100%; border-collapse: collapse; font-size: 0.9rem;">
        <thead>
          <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
            <th style="padding: 12px 8px; text-align: left; font-weight: bold;">Exercise</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Max Weight</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Max Reps</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Est. 1RM</th>
            <th style="padding: 12px 8px; text-align: center; font-weight: bold;">Total Sets</th>
          </tr>
        </thead>
        <tbody>
    `;
    
    data.slice(0, 15).forEach((record, index) => {
      html += `
        <tr style="border-bottom: 1px solid #dee2e6; ${index % 2 === 0 ? 'background-color: #f8f9fa;' : ''} transition: background-color 0.2s;">
          <td style="padding: 10px 8px; font-weight: 600; color: #2c3e50;">${record.exercise}</td>
          <td style="padding: 10px 8px; text-align: center; color: #27ae60; font-weight: 500;">${record.max_weight} kg</td>
          <td style="padding: 10px 8px; text-align: center; color: #8e44ad; font-weight: 500;">${record.max_reps}</td>
          <td style="padding: 10px 8px; text-align: center; font-weight: bold; color: #e74c3c; font-size: 1.1rem;">${record.estimated_1rm} kg</td>
          <td style="padding: 10px 8px; text-align: center; color: #7f8c8d;">${record.total_sets}</td>
        </tr>
      `;
    });
    
    html += '</tbody></table>';
    container.innerHTML = html;

  } catch (error) {
    console.error('Error loading personal records table:', error);
  }
}

async function loadTrainingCalendar() {
  try {
    const data = await fetchJSON('/api/calendar');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('calendarChart', [], {
        title: 'No training calendar data available'
      });
      return;
    }

    const trace = {
      x: data.map(d => d.date),
      y: data.map(d => d.exercises_performed),
      name: 'Exercises per Day',
      type: 'bar',
      marker: { 
        color: data.map(d => d.exercises_performed),
        colorscale: 'Viridis',
        showscale: true
      },
      hovertemplate: 'Date: %{x}<br>Exercises: %{y}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<br><i>Click to view workout details</i><extra></extra>',
      customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets }))
    };

    const layout = {
      title: '📅 Training Calendar - Workout Intensity',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Exercises Performed' },
      showlegend: false
    };

    const plotDiv = document.getElementById('calendarChart');
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });
    
    // Add click event listener for workout details
    plotDiv.on('plotly_click', function(data) {
      if (data.points && data.points.length > 0) {
        const point = data.points[0];
        const workoutDate = point.x; // Date from the clicked point
        showWorkoutDetail(workoutDate);
      }
    });

    // Populate recent workouts list (clickable entries)
    const recentContainer = document.getElementById('recentWorkouts');
    if (recentContainer) {
      recentContainer.innerHTML = '';
      // Show up to 12 most recent workouts
      data.slice().reverse().slice(-12).reverse().forEach(d => {
        const card = document.createElement('button');
        card.className = 'w-full text-left px-4 py-3 rounded-lg bg-zinc-800/40 hover:bg-zinc-800 transition-colors flex items-center justify-between';
        const left = document.createElement('div');
        left.innerHTML = `<div class="text-sm font-medium text-zinc-100">${d.date}</div><div class="text-xs text-zinc-400">${d.exercises_performed} exercises • ${Math.round(d.total_volume)} kg</div>`;
        const right = document.createElement('div');
        right.className = 'text-xs text-zinc-400';
        right.textContent = 'View';
        card.appendChild(left);
        card.appendChild(right);
        card.addEventListener('click', () => {
          // Update URL for shareable link
          if (window.location.pathname !== `/workout/${d.date}`) {
            window.history.pushState(null, '', `/workout/${d.date}`);
          }
          showWorkoutDetail(d.date);
        });
        recentContainer.appendChild(card);
      });
    }

  } catch (error) {
    console.error('Error loading training calendar:', error);
  }
}

async function loadMuscleGroupBalance() {
  try {
    const data = await fetchJSON('/api/muscle-balance');
    
    if (!data || Object.keys(data).length === 0) {
      Plotly.newPlot('muscleBalanceChart', [], {
        title: 'No muscle balance data available'
      });
      return;
    }

    const muscleGroups = Object.keys(data);
    const values = muscleGroups.map(group => data[group].max_estimated_1rm);
    
    const trace = {
      labels: muscleGroups,
      values: values,
      type: 'pie',
      hovertemplate: '<b>%{label}</b><br>Max Est. 1RM: %{value} kg<br>%{percent}<extra></extra>',
      textinfo: 'label+percent',
      marker: {
        colors: exerciseColors,
        line: { color: '#FFFFFF', width: 2 }
      }
    };

    const layout = {
      title: '💪 Muscle Group Strength Distribution',
      showlegend: false,
      margin: { l: 20, r: 20, t: 50, b: 20 }
    };

    Plotly.newPlot('muscleBalanceChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading muscle balance:', error);
  }
}

async function loadBodyMeasurements() {
  try {
    const data = await fetchJSON('/api/measurements');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('measurementsChart', [], {
        title: 'No body measurements data available'
      });
      return;
    }

    const trace = {
      x: data.map(m => m.date),
      y: data.map(m => m.weight),
      name: 'Body Weight',
      type: 'scatter',
      mode: 'lines+markers',
      fill: 'tonexty',
      fillcolor: 'rgba(52, 152, 219, 0.2)',
      line: { color: colors.secondary, width: 3 },
      marker: { size: 6 },
      hovertemplate: 'Date: %{x}<br>Weight: %{y} kg<extra></extra>'
    };

    const layout = {
      title: '📊 Body Weight Progression',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Weight (kg)' },
      showlegend: false
    };

    Plotly.newPlot('measurementsChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading body measurements:', error);
  }
}

async function loadTrainingStreak() {
  try {
    const data = await fetchJSON('/api/training-streak');
    
    document.getElementById('currentStreak').textContent = data.current_streak || '0';
    document.getElementById('longestStreak').textContent = data.longest_streak || '0';
    document.getElementById('totalWorkouts').textContent = data.total_workout_days || '0';
    
    // Add some animation
    const elements = ['currentStreak', 'longestStreak', 'totalWorkouts'];
    elements.forEach((id, index) => {
      const element = document.getElementById(id);
      setTimeout(() => {
        element.style.transform = 'scale(1.1)';
        setTimeout(() => {
          element.style.transform = 'scale(1)';
        }, 200);
      }, index * 100);
    });

  } catch (error) {
    console.error('Error loading training streak:', error);
  }
}

// Load metadata
async function loadLastIngested() {
  try {
    const health = await fetchJSON('/health');
    const lastIngested = health.last_ingested_at;
    const element = document.getElementById('lastIngested');
    
    if (lastIngested) {
      const date = new Date(lastIngested);
      element.textContent = `Last data update: ${date.toLocaleString()}`;
      element.style.color = '#27ae60';
    } else {
      element.textContent = 'No data ingested yet';
      element.style.color = '#e74c3c';
    }

  } catch (error) {
    console.error('Error loading last ingested:', error);
    document.getElementById('lastIngested').textContent = 'Status unknown';
  }
}

// Add some global styles for better UX
const style = document.createElement('style');
style.textContent = `
  .streak-item {
    transition: all 0.3s ease;
  }
  .streak-item:hover {
    transform: translateY(-5px);
  }
  table tbody tr {
    transition: background-color 0.2s ease;
  }
  .chart-container {
    transition: box-shadow 0.3s ease;
  }
  .chart-container:hover {
    box-shadow: 0 8px 25px rgba(0,0,0,0.15);
  }
`;
document.head.appendChild(style);

// ADVANCED ANALYTICS FUNCTIONS - THE FULL ARSENAL 🔥

async function loadVolumeHeatmap() {
  try {
    const data = await fetchJSON('/api/volume-heatmap');
    
    if (!data || data.length === 0) {
      Plotly.newPlot('volumeHeatmapChart', [], {
        title: 'No volume heatmap data available'
      });
      return;
    }

    // Create calendar heatmap similar to GitHub contributions
    const trace = {
      x: data.map(d => d.date),
      y: data.map(d => ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'][d.day_of_week]),
      z: data.map(d => d.intensity),
      type: 'scatter',
      mode: 'markers',
      marker: {
        size: 15,
        color: data.map(d => d.intensity),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Training Intensity' }
      },
      hovertemplate: 'Date: %{x}<br>Day: %{y}<br>Volume: %{customdata} kg<extra></extra>',
      customdata: data.map(d => d.volume)
    };

    const layout = {
      title: '🔥 Training Volume Heatmap (GitHub Style)',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Day of Week' },
      showlegend: false
    };

    Plotly.newPlot('volumeHeatmapChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading volume heatmap:', error);
  }
}

async function loadRepDistribution() {
  try {
    const data = await fetchJSON('/api/rep-distribution');
    
    if (!data || data.length === 0) return;

    // Calculate percentages
    const totalSets = data.reduce((sum, item) => sum + item.set_count, 0);
    data.forEach(item => {
      item.percentage = ((item.set_count / totalSets) * 100).toFixed(1);
    });

    const trace = {
      labels: data.map(d => d.rep_range),
      values: data.map(d => d.set_count),
      type: 'pie',
      hovertemplate: '<b>%{label}</b><br>Sets: %{value}<br>%{percent}<br>Volume: %{customdata} kg<extra></extra>',
      customdata: data.map(d => d.total_volume),
      marker: {
        colors: ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'],
        line: { color: '#FFFFFF', width: 2 }
      }
    };

    const layout = {
      title: '📊 Rep Range Distribution - Training Focus',
      showlegend: true,
      legend: { orientation: 'h', y: -0.1 }
    };

    Plotly.newPlot('repDistributionChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading rep distribution:', error);
  }
}

async function loadExerciseFrequency() {
  try {
    const data = await fetchJSON('/api/exercise-frequency');
    
    if (!data || data.length === 0) return;

    const trace = {
      x: data.slice(0, 10).map(d => d.exercise),
      y: data.slice(0, 10).map(d => d.workout_days),
      type: 'bar',
      marker: {
        color: data.slice(0, 10).map(d => d.workout_days),
        colorscale: 'Viridis',
        showscale: true
      },
      hovertemplate: '<b>%{x}</b><br>Workout Days: %{y}<br>Total Sets: %{customdata.sets}<br>Avg Weight: %{customdata.weight} kg<extra></extra>',
      customdata: data.slice(0, 10).map(d => ({
        sets: d.total_sets,
        weight: d.avg_weight
      }))
    };

    const layout = {
      title: '🎯 Exercise Frequency - Top 10 Most Trained',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Workout Days' },
      showlegend: false
    };

    Plotly.newPlot('exerciseFrequencyChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading exercise frequency:', error);
  }
}

async function loadStrengthRatios() {
  try {
    const data = await fetchJSON('/api/strength-ratios');
    
    if (!data || !data.ratios || Object.keys(data.ratios).length === 0) {
      Plotly.newPlot('strengthRatiosChart', [], {
        title: 'No strength ratio data available'
      });
      return;
    }

    const ratioNames = Object.keys(data.ratios);
    const actualRatios = Object.values(data.ratios);
    const idealRatios = ratioNames.map(name => data.ideal_ratios[name] || 1);

    const traces = [
      {
        x: ratioNames,
        y: actualRatios,
        name: 'Your Ratios',
        type: 'bar',
        marker: { color: '#3498db' }
      },
      {
        x: ratioNames,
        y: idealRatios,
        name: 'Ideal Ratios',
        type: 'scatter',
        mode: 'markers',
        marker: { size: 12, color: '#e74c3c', symbol: 'diamond' }
      }
    ];

    const layout = {
      title: '⚡ Strength Ratios vs Ideal Standards',
      xaxis: { title: 'Lift Ratios' },
      yaxis: { title: 'Ratio Value' },
      showlegend: true
    };

    Plotly.newPlot('strengthRatiosChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading strength ratios:', error);
  }
}

async function loadRecoveryTracking() {
  try {
    const data = await fetchJSON('/api/recovery-tracking');
    
    if (!data || Object.keys(data).length === 0) return;

    const muscleGroups = Object.keys(data);
    const avgRecovery = muscleGroups.map(group => data[group].avg_recovery);
    const colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7'];

    const trace = {
      x: muscleGroups,
      y: avgRecovery,
      type: 'bar',
      marker: {
        color: colors.slice(0, muscleGroups.length)
      },
      hovertemplate: '<b>%{x}</b><br>Avg Recovery: %{y} days<br>Sessions: %{customdata.sessions}<br>Range: %{customdata.min}-%{customdata.max} days<extra></extra>',
      customdata: muscleGroups.map(group => ({
        sessions: data[group].total_sessions,
        min: data[group].min_recovery,
        max: data[group].max_recovery
      }))
    };

    const layout = {
      title: '🔄 Recovery Time Between Sessions',
      xaxis: { title: 'Muscle Group' },
      yaxis: { title: 'Average Days Between Sessions' },
      showlegend: false
    };

    Plotly.newPlot('recoveryChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading recovery tracking:', error);
  }
}

async function loadProgressionRate() {
  try {
    const data = await fetchJSON('/api/progression-rate');
    
    if (!data || data.length === 0) return;

    const topProgressors = data.slice(0, 8);

    const trace = {
      x: topProgressors.map(d => d.exercise),
      y: topProgressors.map(d => d.percentage_gain),
      type: 'bar',
      marker: {
        color: topProgressors.map(d => d.percentage_gain),
        colorscale: 'RdYlGn',
        showscale: true,
        colorbar: { title: '% Gain' }
      },
      hovertemplate: '<b>%{x}</b><br>Total Gain: %{y}%<br>Weekly Rate: %{customdata.weekly} kg/week<br>Days Tracked: %{customdata.days}<extra></extra>',
      customdata: topProgressors.map(d => ({
        weekly: d.weekly_rate,
        days: d.days_tracked
      }))
    };

    const layout = {
      title: '📈 Progressive Overload Rate - Top Performers',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Percentage Gain (%)' },
      showlegend: false
    };

    Plotly.newPlot('progressionRateChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading progression rate:', error);
  }
}

async function loadDurationTrends() {
  try {
    const data = await fetchJSON('/api/workout-duration');
    
    if (!data || data.length === 0) return;

    const traces = [
      {
        x: data.map(d => d.date),
        y: data.map(d => d.duration),
        name: 'Duration (min)',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#3498db' },
        yaxis: 'y'
      },
      {
        x: data.map(d => d.date),
        y: data.map(d => d.efficiency_score),
        name: 'Efficiency Score',
        type: 'scatter',
        mode: 'lines+markers',
        line: { color: '#e74c3c', dash: 'dash' },
        yaxis: 'y2'
      }
    ];

    const layout = {
      title: '⏱️ Workout Duration & Efficiency Trends',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Duration (minutes)', side: 'left' },
      yaxis2: { title: 'Efficiency Score', side: 'right', overlaying: 'y' },
      hovermode: 'x unified',
      showlegend: true
    };

    Plotly.newPlot('durationTrendsChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading duration trends:', error);
  }
}

async function loadBestSets() {
  try {
    const data = await fetchJSON('/api/best-sets');
    
    if (!data || data.length === 0) return;

    const topSets = data.slice(0, 12);

    const trace = {
      x: topSets.map(d => d.exercise),
      y: topSets.map(d => d.estimated_1rm),
      type: 'bar',
      marker: {
        color: topSets.map(d => d.estimated_1rm),
        colorscale: 'Viridis',
        showscale: true,
        colorbar: { title: 'Est. 1RM (kg)' }
      },
      hovertemplate: '<b>%{x}</b><br>Est. 1RM: %{y} kg<br>Weight: %{customdata.weight} kg<br>Reps: %{customdata.reps}<br>Date: %{customdata.date}<extra></extra>',
      customdata: topSets.map(d => ({
        weight: d.weight,
        reps: d.reps,
        date: d.date
      }))
    };

    const layout = {
      title: '🏆 Best Set Performance - Personal Records',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Estimated 1RM (kg)' },
      showlegend: false
    };

    Plotly.newPlot('bestSetsChart', [trace], layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading best sets:', error);
  }
}

async function loadPlateauDetection() {
  try {
    const data = await fetchJSON('/api/plateau-detection');
    
    if (!data || data.length === 0) return;

    // Group by status for better visualization
    const statusColors = {
      'Progressing': '#27ae60',
      'Plateau': '#f39c12', 
      'Declining': '#e74c3c',
      'Variable': '#8e44ad'
    };

    const traces = [];
    const statuses = ['Progressing', 'Plateau', 'Declining', 'Variable'];
    
    statuses.forEach(status => {
      const statusData = data.filter(d => d.status === status);
      if (statusData.length > 0) {
        traces.push({
          x: statusData.map(d => d.exercise),
          y: statusData.map(d => d.current_1rm),
          name: status,
          type: 'scatter',
          mode: 'markers',
          marker: {
            size: 12,
            color: statusColors[status]
          },
          hovertemplate: '<b>%{x}</b><br>Status: ' + status + '<br>Current 1RM: %{y} kg<br>Best Recent: %{customdata} kg<extra></extra>',
          customdata: statusData.map(d => d.best_recent_1rm)
        });
      }
    });

    const layout = {
      title: '🚨 Plateau Detection - Exercise Performance Status',
      xaxis: { title: 'Exercise', tickangle: -45 },
      yaxis: { title: 'Current Est. 1RM (kg)' },
      hovermode: 'closest',
      showlegend: true,
      legend: { orientation: 'h', y: -0.2 }
    };

    Plotly.newPlot('plateauDetectionChart', traces, layout, { responsive: true, displayModeBar: false });

  } catch (error) {
    console.error('Error loading plateau detection:', error);
  }
}

// ----------------------- Workout Detail Modal Functions -----------------------

async function showWorkoutDetail(workoutDate) {
  try {
    const modal = document.getElementById('workoutModal');
    const content = document.getElementById('workoutModalContent');
    const title = document.getElementById('workoutModalTitle');
    
    // Show modal with loading state
    modal.classList.remove('hidden');
    content.innerHTML = '<div class="flex items-center justify-center h-32 text-zinc-500"><div class="animate-pulse">Loading workout details...</div></div>';
    
    // Fetch workout data
    const response = await fetch(`/api/workout/${workoutDate}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch workout: ${response.statusText}`);
    }
    
    const workout = await response.json();
    
    // Update modal title and badges
    title.textContent = workout.workout_name || `Workout - ${workout.date}`;
    
    const badges = document.getElementById('workoutModalBadges');
    badges.innerHTML = `
      <span class="px-2 py-1 bg-indigo-600/20 text-indigo-300 rounded text-xs font-medium">
        ${workout.total_exercises} exercises
      </span>
      <span class="px-2 py-1 bg-green-600/20 text-green-300 rounded text-xs font-medium">
        ${workout.duration_minutes}min
      </span>
      ${workout.total_prs > 0 ? `<span class="px-2 py-1 bg-yellow-600/20 text-yellow-300 rounded text-xs font-medium">🏆 ${workout.total_prs} PR${workout.total_prs > 1 ? 's' : ''}</span>` : ''}
    `;
    
    // Generate workout content
    content.innerHTML = generateWorkoutHTML(workout);
    
    // Set up share functionality
    const shareBtn = document.getElementById('shareWorkoutBtn');
    shareBtn.onclick = () => shareWorkout(workoutDate);
    
  } catch (error) {
    console.error('Error loading workout detail:', error);
    document.getElementById('workoutModalContent').innerHTML = `
      <div class="flex items-center justify-center h-32 text-red-400">
        <div>Error loading workout details</div>
      </div>
    `;
  }
}

function generateWorkoutHTML(workout) {
  let html = `
    <div class="space-y-6">
      <!-- Workout Summary -->
      <div class="grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-zinc-800/50 rounded-xl">
        <div class="text-center">
          <div class="text-2xl font-bold text-indigo-400">${workout.total_sets}</div>
          <div class="text-sm text-zinc-400">Total Sets</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-green-400">${fmtInt(workout.total_volume)}</div>
          <div class="text-sm text-zinc-400">Volume (kg)</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-blue-400">${workout.duration_minutes}</div>
          <div class="text-sm text-zinc-400">Duration (min)</div>
        </div>
        <div class="text-center">
          <div class="text-2xl font-bold text-yellow-400">${workout.total_prs}</div>
          <div class="text-sm text-zinc-400">Personal Records</div>
        </div>
      </div>
      
      <!-- Exercise Details -->
      <div class="space-y-4">
  `;
  
  workout.exercises.forEach((exercise, index) => {
    const isFirstExercise = index === 0;
    const bgColor = isFirstExercise ? 'bg-indigo-600/10 border-indigo-600/30' : 'bg-zinc-800/30 border-zinc-700';
    const prBadge = exercise.personal_records > 0 ? '<span class="text-xs bg-yellow-600/20 text-yellow-300 px-2 py-0.5 rounded font-medium ml-2">🏆 PR</span>' : '';
    
    html += `
      <div class="border ${bgColor} rounded-xl p-4">
        <div class="flex items-center justify-between mb-3">
          <h3 class="text-lg font-semibold text-zinc-100">${exercise.exercise_name}${prBadge}</h3>
          <div class="text-sm text-zinc-400">
            ${exercise.total_sets} sets • ${fmt1(exercise.total_volume)} kg total
          </div>
        </div>
        
        <!-- Sets Table -->
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead>
              <tr class="border-b border-zinc-700">
                <th class="text-left py-2 text-zinc-400 font-medium">Set</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Weight</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Reps</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Volume</th>
                <th class="text-right py-2 text-zinc-400 font-medium">Est. 1RM</th>
              </tr>
            </thead>
            <tbody>
    `;
    
    exercise.sets.forEach(set => {
      const isBestSet = set.volume === exercise.best_set.volume;
      const rowClass = isBestSet ? 'bg-green-600/10 text-green-300' : 'text-zinc-300';
      const bestSetIcon = isBestSet ? ' <span class="text-green-400">★</span>' : '';
      
      html += `
        <tr class="${rowClass} border-b border-zinc-800/50">
          <td class="py-2">${set.set_number}${bestSetIcon}</td>
          <td class="text-right py-2">${set.weight ? fmt1(set.weight) + ' kg' : '-'}</td>
          <td class="text-right py-2">${set.reps || '-'}</td>
          <td class="text-right py-2">${set.volume ? fmt1(set.volume) + ' kg' : '-'}</td>
          <td class="text-right py-2">${set.estimated_1rm ? fmt1(set.estimated_1rm) + ' kg' : '-'}</td>
        </tr>
      `;
    });
    
    html += `
            </tbody>
          </table>
        </div>
      </div>
    `;
  });
  
  html += `
      </div>
    </div>
  `;
  
  return html;
}

function shareWorkout(workoutDate) {
  const url = `${window.location.origin}/workout/${workoutDate}`;
  
  if (navigator.share) {
    // Use native share API if available (mobile)
    navigator.share({
      title: 'My Workout',
      text: `Check out my workout from ${workoutDate}`,
      url: url
    }).catch(console.error);
  } else {
    // Fallback to clipboard
    navigator.clipboard.writeText(url).then(() => {
      // Show success feedback
      const btn = document.getElementById('shareWorkoutBtn');
      const originalText = btn.textContent;
      btn.textContent = 'Copied!';
      btn.className = btn.className.replace('bg-indigo-600 hover:bg-indigo-700', 'bg-green-600 hover:bg-green-700');
      
      setTimeout(() => {
        btn.textContent = originalText;
        btn.className = btn.className.replace('bg-green-600 hover:bg-green-700', 'bg-indigo-600 hover:bg-indigo-700');
      }, 2000);
    }).catch(console.error);
  }
}

// Set up modal event listeners
document.addEventListener('DOMContentLoaded', () => {
  const modal = document.getElementById('workoutModal');
  const closeBtn = document.getElementById('closeWorkoutModal');
  
  // Close modal handlers
  closeBtn?.addEventListener('click', () => {
    modal.classList.add('hidden');
    // Update URL to remove workout parameter
    if (window.location.pathname !== '/') {
      window.history.pushState(null, '', '/');
    }
  });
  
  // Close on backdrop click
  modal?.addEventListener('click', (e) => {
    if (e.target === modal) {
      modal.classList.add('hidden');
      if (window.location.pathname !== '/') {
        window.history.pushState(null, '', '/');
      }
    }
  });
  
  // Close on escape key
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !modal.classList.contains('hidden')) {
      modal.classList.add('hidden');
      if (window.location.pathname !== '/') {
        window.history.pushState(null, '', '/');
      }
    }
  });
});

// Make showWorkoutDetail globally available
window.showWorkoutDetail = showWorkoutDetail;
