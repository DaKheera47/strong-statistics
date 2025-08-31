// Core shared state, helpers, and data fetching for dashboard
// (extracted from original main.js)

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
  if(metricWrap) metricWrap.innerHTML='<span class="text-xs text-zinc-500">Weight mode</span>';
}

function updateMetricButtons(){}

// Exercise multi-select (simple dropdown)
function initExerciseMulti(exNames){
  const root=document.getElementById('exerciseMulti');
  if(!root) return;
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
    const lastIngestedEl = document.getElementById('lastIngested');
    if(lastIngestedEl) lastIngestedEl.textContent = data.filters.end || '-';
    if(!state.exercises.length){
      // Preselect by most improvement (delta)
      const prog = (data.exercise_progression || []).map(p=> p.exercise);
      state.exercises = prog.slice(0,12);
      if(state.exercises.length===0) state.exercises = data.filters.exercises || (data.top_exercises || []);
      initExerciseMulti(unique(state.data.exercises_daily_max.map(d=>d.exercise)));
    }
  // Call whichever aggregate render function is available (charts split across modules)
  try { (window.renderAllCharts || window.renderAll || (()=>{}))(); } catch(e){ console.error('Error during renderAll:', e); }
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

// Load training streak and metadata helpers
async function loadTrainingStreak() {
  try {
    const data = await fetchJSON('/api/training-streak');
    
    const current = document.getElementById('currentStreak');
    const longest = document.getElementById('longestStreak');
    const total = document.getElementById('totalWorkouts');
    if(current) current.textContent = data.current_streak || '0';
    if(longest) longest.textContent = data.longest_streak || '0';
    if(total) total.textContent = data.total_workout_days || '0';
    
    // Add some animation
    const elements = ['currentStreak', 'longestStreak', 'totalWorkouts'];
    elements.forEach((id, index) => {
      const element = document.getElementById(id);
      if(!element) return;
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

async function loadLastIngested() {
  try {
    const health = await fetchJSON('/health');
    const lastIngested = health.last_ingested_at;
    const element = document.getElementById('lastIngested');
    
    if (element) {
      if (lastIngested) {
        const date = new Date(lastIngested);
        element.textContent = `Last data update: ${date.toLocaleString()}`;
        element.style.color = '#27ae60';
      } else {
        element.textContent = 'No data ingested yet';
        element.style.color = '#e74c3c';
      }
    }

  } catch (error) {
    console.error('Error loading last ingested:', error);
    const element = document.getElementById('lastIngested');
    if(element) element.textContent = 'Status unknown';
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

// Bootstrap
function bootstrapDashboard(){
  console.log('[dashboard] bootstrap');
  initFilters();
  updateMetricButtons();
  refreshData();
  // load some non-chart metadata
  loadLastIngested();
  loadTrainingStreak();
}

// Do not auto-run bootstrap here — wait until all modular scripts are loaded.
// Expose the bootstrap function so the page can call it after loading chart modules.
window.bootstrapDashboard = bootstrapDashboard;
