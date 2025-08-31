// Plotly-driven sections of the dashboard (personal records, calendar, advanced analytics)

async function loadPersonalRecords() {
  try {
    const data = await fetchJSON('/api/personal-records');
    if (!data || data.length === 0) return;
    const exerciseGroups = {};
    data.forEach(pr => { (exerciseGroups[pr.exercise] ||= []).push(pr); });
    const traces = [];
    let colorIndex = 0;
    Object.entries(exerciseGroups).forEach(([exercise, prs]) => {
      traces.push({ x: prs.map(pr => pr.date), y: prs.map(pr => pr.weight), type: 'scatter', mode: 'markers+text', name: exercise, marker: { size: 12, color: '#f59e0b', symbol: 'star' }, text: prs.map(pr => `${pr.weight}kg`), textposition: 'top center', hovertemplate: `<b>${exercise}</b><br>Date: %{x}<br>Weight: %{y} kg<extra></extra>` });
      colorIndex++;
    });
    const layout = { title: 'Personal Records Achievement', xaxis:{title:'Date', type:'date'}, yaxis:{title:'Weight (kg)'}, hovermode:'closest', legend:{orientation:'h', y:-0.2} };
    Plotly.newPlot('personalRecordsChart', traces, layout, { responsive: true, displayModeBar: false });
  } catch (error) { console.error('Error loading personal records:', error); }
}

async function loadTrainingCalendar() {
  try {
    const data = await fetchJSON('/api/calendar');
    if (!data || data.length === 0) { Plotly.newPlot('calendarChart', [], { title: 'No training calendar data available' }); return; }
    const trace = { x: data.map(d => d.date), y: data.map(d => d.exercises_performed), name: 'Exercises per Day', type: 'bar', marker: { color: data.map(d => d.exercises_performed), colorscale: 'Viridis', showscale: true }, hovertemplate: 'Date: %{x}<br>Exercises: %{y}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<br><i>Click to view workout details</i><extra></extra>', customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets })) };
    const layout = { title: '📅 Training Calendar - Workout Intensity', xaxis:{title:'Date', type:'date'}, yaxis:{title:'Exercises Performed'}, showlegend:false };
    const plotDiv = document.getElementById('calendarChart');
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });
    plotDiv.on('plotly_click', function(data) { if (data.points && data.points.length > 0) { const point = data.points[0]; const workoutDate = point.x; showWorkoutDetail(workoutDate); } });

    // Populate recent workouts
    const recentContainer = document.getElementById('recentWorkouts');
    if (recentContainer) {
      recentContainer.innerHTML='';
      data.slice().reverse().slice(-12).reverse().forEach(d=>{
        const card=document.createElement('button');
        card.className='w-full text-left px-4 py-3 rounded-lg bg-zinc-800/40 hover:bg-zinc-800 transition-colors flex items-center justify-between';
        const left=document.createElement('div'); left.innerHTML=`<div class="text-sm font-medium text-zinc-100">${d.date}</div><div class="text-xs text-zinc-400">${d.exercises_performed} exercises • ${Math.round(d.total_volume)} kg</div>`;
        const right=document.createElement('div'); right.className='text-xs text-zinc-400'; right.textContent='View';
        card.appendChild(left); card.appendChild(right);
        card.addEventListener('click', ()=>{ if(window.location.pathname !== `/workout/${d.date}`) window.history.pushState(null,'',`/workout/${d.date}`); showWorkoutDetail(d.date); });
        recentContainer.appendChild(card);
      });
    }
  } catch (error) { console.error('Error loading training calendar:', error); }
}

// Other Plotly-based loaders are left in this file for clarity (muscle balance, measurements, etc.)
async function loadMuscleGroupBalance() {
  try { const data = await fetchJSON('/api/muscle-balance'); if (!data || Object.keys(data).length === 0) { Plotly.newPlot('muscleBalanceChart', [], { title: 'No muscle balance data available' }); return; } const muscleGroups = Object.keys(data); const values = muscleGroups.map(group => data[group].max_estimated_1rm); const trace = { labels: muscleGroups, values: values, type: 'pie', hovertemplate: '<b>%{label}</b><br>Max Est. 1RM: %{value} kg<br>%{percent}<extra></extra>', textinfo: 'label+percent', marker: { colors: ['#8B5CF6','#10B981','#6366F1','#EC4899','#F59E0B'], line: { color: '#FFFFFF', width: 2 } } }; const layout = { title: '💪 Muscle Group Strength Distribution', showlegend: false, margin: { l: 20, r: 20, t: 50, b: 20 } }; Plotly.newPlot('muscleBalanceChart', [trace], layout, { responsive: true, displayModeBar: false }); } catch (error) { console.error('Error loading muscle balance:', error); } }

// Expose plotly loaders
window.loadPersonalRecords = loadPersonalRecords;
window.loadTrainingCalendar = loadTrainingCalendar;
window.loadMuscleGroupBalance = loadMuscleGroupBalance;
