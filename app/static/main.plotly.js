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
    const plotDiv = document.getElementById('calendarChart');
    if(!plotDiv) return;
    // Clear any loading placeholder
    plotDiv.innerHTML='';
    if (!data || data.length === 0) {
      Plotly.newPlot(plotDiv, [], { title: 'No training calendar data available', paper_bgcolor:'#18181b', plot_bgcolor:'#18181b', font:{color:'#e4e4e7'} });
      return;
    }
  // Intensity metric based on relative session volume (weight normalized 0-1 -> scaled)
  const volumes = data.map(d=> d.total_volume || 0);
  const vMin = Math.min(...volumes);
  const vMax = Math.max(...volumes);
  const intensity = volumes.map(v=> vMax===vMin? 1 : ( (v - vMin) / (vMax - vMin) )*0.7 + 0.3 ); // keep within 0.3-1 range for color separation
    const trace = {
      x: data.map(d => d.date),
  y: data.map(d => d.exercises_performed),
  name: 'Session Intensity',
      type: 'bar',
      marker: {
        color: intensity,
        colorscale: 'Inferno',
        showscale: true,
        colorbar: {
          title: 'Intensity',
          tickcolor:'#a1a1aa',
          tickfont:{color:'#a1a1aa'},
          titlefont:{color:'#d4d4d8'}
        }
      },
      hovertemplate: 'Date: %{x}<br>Exercises: %{customdata.ex}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<br>Intensity: %{marker.color:.2f}<br><i>Click to view workout details</i><extra></extra>',
      customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets, ex: d.exercises_performed }))
    };
    const axisStyle = { tickcolor:'#3f3f46', color:'#a1a1aa', gridcolor:'#27272a', linecolor:'#3f3f46' };
    const layout = {
      title: { text:'📅 Training Calendar - Workout Intensity', font:{color:'#e4e4e7'} },
      paper_bgcolor:'#18181b',
      plot_bgcolor:'#18181b',
      xaxis:{ title:'Date', type:'date', tickfont:{color:'#a1a1aa'}, titlefont:{color:'#a1a1aa'}, gridcolor:'#27272a', linecolor:axisStyle.linecolor },
      yaxis:{ title:'Exercises Performed', tickfont:{color:'#a1a1aa'}, titlefont:{color:'#a1a1aa'}, gridcolor:'#27272a', linecolor:axisStyle.linecolor },
      font:{color:'#d4d4d8'},
      showlegend:false,
      margin:{l:60,r:40,t:60,b:60}
    };
    Plotly.newPlot(plotDiv, [trace], layout, { responsive: true, displayModeBar: false });
    plotDiv.on('plotly_click', function(ev) {
      if (ev.points && ev.points.length > 0) {
        let raw = ev.points[0].x;
        // Normalize to YYYY-MM-DD
        const workoutDate = raw instanceof Date ? raw.toISOString().slice(0,10) : String(raw).slice(0,10);
        openWorkout(workoutDate);
      }
    });

    // Populate recent workouts table (new view)
    if (typeof window.populateRecentWorkouts === 'function') {
      try { window.populateRecentWorkouts(data); } catch(e){ console.warn('populateRecentWorkouts failed', e); }
    }
  } catch (error) { console.error('Error loading training calendar:', error); }
}

function openWorkout(date){
  if(!date) return;
  // Update URL (pushState) then show modal
  try {
    if(window.location.pathname !== `/workout/${date}`){ window.history.pushState(null,'',`/workout/${date}`); }
    if(typeof window.showWorkoutDetail === 'function'){
      window.showWorkoutDetail(date);
    } else {
      console.warn('showWorkoutDetail not ready, retrying...');
      setTimeout(()=> window.showWorkoutDetail && window.showWorkoutDetail(date), 300);
    }
  } catch(e){ console.error('openWorkout failed', e); }
}

// Other Plotly-based loaders are left in this file for clarity (muscle balance, measurements, etc.)
async function loadMuscleGroupBalance() {
  try { const data = await fetchJSON('/api/muscle-balance'); if (!data || Object.keys(data).length === 0) { Plotly.newPlot('muscleBalanceChart', [], { title: 'No muscle balance data available' }); return; } const muscleGroups = Object.keys(data); const values = muscleGroups.map(group => data[group].max_estimated_1rm); const trace = { labels: muscleGroups, values: values, type: 'pie', hovertemplate: '<b>%{label}</b><br>Max Est. 1RM: %{value} kg<br>%{percent}<extra></extra>', textinfo: 'label+percent', marker: { colors: ['#8B5CF6','#10B981','#6366F1','#EC4899','#F59E0B'], line: { color: '#FFFFFF', width: 2 } } }; const layout = { title: '💪 Muscle Group Strength Distribution', showlegend: false, margin: { l: 20, r: 20, t: 50, b: 20 } }; Plotly.newPlot('muscleBalanceChart', [trace], layout, { responsive: true, displayModeBar: false }); } catch (error) { console.error('Error loading muscle balance:', error); } }

// Expose plotly loaders
window.loadPersonalRecords = loadPersonalRecords;
window.loadTrainingCalendar = loadTrainingCalendar;
window.loadMuscleGroupBalance = loadMuscleGroupBalance;
