async function fetchJSON(url){
  const r = await fetch(url); if(!r.ok) throw new Error(await r.text()); return r.json();
}

function formatDate(d){return d;}

async function loadSessions(){
  const data = await fetchJSON('/api/sessions');
  if(!data.length) return;
  const dates = data.map(d=>d.session_date);
  const volume = data.map(d=>d.volume);
  const duration = data.map(d=>d.duration);

  Plotly.newPlot('volumeChart', [{x: dates, y: volume, type:'scatter', mode:'lines+markers', name:'Volume'}], {title:'Volume Over Time', xaxis:{tickangle: -30}});
  Plotly.newPlot('durationChart', [{x: dates, y: duration, type:'scatter', mode:'lines+markers', name:'Duration (min)'}], {title:'Duration Trend', xaxis:{tickangle: -30}});
}

async function loadTopExercises(){
  const data = await fetchJSON('/api/top-exercises?limit=15');
  if(!data.length) return;
  Plotly.newPlot('topExercisesChart', [{x: data.map(d=>d.exercise), y: data.map(d=>d.volume), type:'bar'}], {title:'Top Exercises by Volume', xaxis:{tickangle:-30}});
}

async function loadTuesdayStrength(){
  const data = await fetchJSON('/api/tuesday-strength');
  const byExercise = {};
  data.forEach(r=>{(byExercise[r.exercise] ||= []).push(r);});
  const traces = Object.entries(byExercise).map(([ex, rows])=>({x: rows.map(r=>r.date), y: rows.map(r=>r.max_weight), type:'scatter', mode:'lines+markers', name: ex}));
  Plotly.newPlot('tuesdayStrengthChart', traces, {title:'Tuesday Strength (Max Weight)', xaxis:{tickangle:-30}});
}

async function loadExercisesList(){
  const list = await fetchJSON('/api/exercises');
  const sel = document.getElementById('exerciseSelect');
  sel.innerHTML = '<option value="">-- Pick --</option>' + list.map(e=>`<option>${e}</option>`).join('');
  sel.addEventListener('change', ()=>{ if(sel.value) loadExerciseProgress(sel.value); });
}

async function loadExerciseProgress(ex){
  const data = await fetchJSON('/api/exercise-progress?exercise=' + encodeURIComponent(ex));
  if(!data.length){
    Plotly.newPlot('exerciseProgressChart', [], {title:'Exercise Progress'});
    return;
  }
  Plotly.newPlot('exerciseProgressChart', [
    {x: data.map(r=>r.date), y: data.map(r=>r.max_weight), type:'scatter', mode:'lines+markers', name:'Max Weight'},
    {x: data.map(r=>r.date), y: data.map(r=>r.top_set_volume), yaxis:'y2', type:'bar', opacity:0.3, name:'Top Set Volume'}
  ], {title: 'Exercise Progress', xaxis:{tickangle:-30}, yaxis:{title:'Weight'}, yaxis2:{title:'Volume', overlaying:'y', side:'right'}});
}

async function refreshMeta(){
  const h = await fetchJSON('/health');
  if(h.last_ingested_at){
    document.getElementById('lastIngested').textContent = 'Last ingest: ' + h.last_ingested_at;
  }
}

(async function init(){
  await Promise.all([
    loadSessions(),
    loadTopExercises(),
    loadTuesdayStrength(),
    loadExercisesList(),
    refreshMeta()
  ]);
})();
