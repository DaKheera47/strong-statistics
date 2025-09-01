// Slim main.js: only bootstrap + workout modal + share functionality (charts in core/charts files)
// NOTE: File replaced to remove duplicated logic.

(function(){
  function bootstrapDashboard(){
    if(window.__dashboardBooted) return; window.__dashboardBooted = true;
    if(typeof window.refreshData === 'function') window.refreshData(); else setTimeout(()=> window.refreshData && window.refreshData(), 150);
  }
  if(document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootstrapDashboard); else bootstrapDashboard();

  window.addEventListener('resize', ()=>{ if(window.charts) Object.values(window.charts).forEach(c=> { try { c.resize(); } catch(_){ } }); });

  function fmt1(x){ return window.fmt1? window.fmt1(x): (x==null?'-': (Math.round(x*10)/10).toString()); }
  function fmtInt(x){ return window.fmtInt? window.fmtInt(x): (x==null?'-': x.toLocaleString()); }

  async function showWorkoutDetail(workoutDate) {
    try {
      const modal = document.getElementById('workoutModal');
      const content = document.getElementById('workoutModalContent');
      const title = document.getElementById('workoutModalTitle');
      if(!modal) return;
      modal.classList.remove('hidden');
      if(content) content.innerHTML = '<div class="flex items-center justify-center h-32 text-zinc-500"><div class="animate-pulse">Loading workout details...</div></div>';
      const resp = await fetch(`/api/workout/${workoutDate}`);
      if(!resp.ok) throw new Error('Failed to fetch workout');
      const workout = await resp.json();
      if(title) title.textContent = workout.workout_name || `Workout - ${workout.date}`;
      const badges = document.getElementById('workoutModalBadges');
      if(badges) badges.innerHTML = `
        <span class=\"px-2 py-1 bg-indigo-600/20 text-indigo-300 rounded text-xs font-medium\">${workout.total_exercises} exercises</span>
        <span class=\"px-2 py-1 bg-green-600/20 text-green-300 rounded text-xs font-medium\">${workout.duration_minutes}min</span>
        ${workout.total_prs > 0 ? `<span class=\"px-2 py-1 bg-yellow-600/20 text-yellow-300 rounded text-xs font-medium\">🏆 ${workout.total_prs} PR${workout.total_prs>1?'s':''}</span>`:''}
      `;
      if(content) content.innerHTML = generateWorkoutHTML(workout);
      const shareBtn=document.getElementById('shareWorkoutBtn');
      if(shareBtn) shareBtn.onclick=()=> shareWorkout(workoutDate);
    } catch(e){
      console.error('Error loading workout detail', e);
      const content = document.getElementById('workoutModalContent');
      if(content) content.innerHTML = '<div class="flex items-center justify-center h-32 text-red-400">Error loading workout details</div>';
    }
  }

  function generateWorkoutHTML(workout){
    let html = '<div class="space-y-6">';
    html += `<div class=\"grid grid-cols-2 md:grid-cols-4 gap-4 p-4 bg-zinc-800/50 rounded-xl\">`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-indigo-400\">${workout.total_sets}</div><div class=\"text-sm text-zinc-400\">Total Sets</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-green-400\">${fmtInt(workout.total_volume)}</div><div class=\"text-sm text-zinc-400\">Volume (kg)</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-blue-400\">${workout.duration_minutes}</div><div class=\"text-sm text-zinc-400\">Duration (min)</div></div>`+
      `<div class=\"text-center\"><div class=\"text-2xl font-bold text-yellow-400\">${workout.total_prs}</div><div class=\"text-sm text-zinc-400\">Personal Records</div></div>`+
    `</div>`;
    html += '<div class="space-y-4">';
    workout.exercises.forEach((ex,i)=>{
      const prBadge = ex.personal_records>0? '<span class="text-xs bg-yellow-600/20 text-yellow-300 px-2 py-0.5 rounded font-medium ml-2">🏆 PR</span>':'';
      html += `<div class=\"border ${i===0?'bg-indigo-600/10 border-indigo-600/30':'bg-zinc-800/30 border-zinc-700'} rounded-xl p-4\">`+
        `<div class=\"flex items-center justify-between mb-3\"><h3 class=\"text-lg font-semibold text-zinc-100\">${ex.exercise_name}${prBadge}</h3>`+
        `<div class=\"text-sm text-zinc-400\">${ex.total_sets} sets • ${fmt1(ex.total_volume)} kg total</div></div>`+
        `<div class=\"overflow-x-auto\"><table class=\"w-full text-sm\"><thead><tr class=\"border-b border-zinc-700\"><th class=\"text-left py-2 text-zinc-400 font-medium\">Set</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Weight</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Reps</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Volume</th><th class=\"text-right py-2 text-zinc-400 font-medium\">Est. 1RM</th></tr></thead><tbody>`;
      ex.sets.forEach(set=>{
        const isBest = set.volume === ex.best_set.volume;
        html += `<tr class=\"${isBest?'bg-green-600/10 text-green-300':'text-zinc-300'} border-b border-zinc-800/50\"><td class=\"py-2\">${set.set_number}${isBest?' <span class=\\"text-green-400\\">★</span>':''}</td><td class=\"text-right py-2\">${set.weight? fmt1(set.weight)+' kg':'-'}</td><td class=\"text-right py-2\">${set.reps||'-'}</td><td class=\"text-right py-2\">${set.volume? fmt1(set.volume)+' kg':'-'}</td><td class=\"text-right py-2\">${set.estimated_1rm? fmt1(set.estimated_1rm)+' kg':'-'}</td></tr>`;
      });
      html += '</tbody></table></div></div>';
    });
    html += '</div></div>';
    return html;
  }

  function shareWorkout(workoutDate){
    const url = `${window.location.origin}/workout/${workoutDate}`;
    const btn = document.getElementById('shareWorkoutBtn');
    if(navigator.share){
      navigator.share({title:'My Workout', text:`Check out my workout from ${workoutDate}`, url}).catch(()=>{});
    } else if(navigator.clipboard){
      navigator.clipboard.writeText(url).then(()=>{
        if(!btn) return; const orig=btn.textContent; btn.textContent='Copied!'; btn.classList.add('bg-green-600'); setTimeout(()=>{ btn.textContent=orig; btn.classList.remove('bg-green-600'); },1500);
      });
    }
  }

  document.addEventListener('DOMContentLoaded', ()=>{
    const modal=document.getElementById('workoutModal');
    const closeBtn=document.getElementById('closeWorkoutModal');
    closeBtn?.addEventListener('click', ()=>{ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); });
    modal?.addEventListener('click', e=>{ if(e.target===modal){ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); } });
    document.addEventListener('keydown', e=>{ if(e.key==='Escape' && !modal.classList.contains('hidden')){ modal.classList.add('hidden'); if(window.location.pathname!=='/') history.pushState(null,'','/'); } });
  });

    window.showWorkoutDetail = showWorkoutDetail;
    window.shareWorkout = shareWorkout;
  })();

// Prefer helpers from main.core.js if present
const fetchJSON = window.fetchJSON || (url => fetch(url).then(r => { if(!r.ok) throw new Error(r.statusText); return r.json(); }));
const fmtInt = window.fmtInt || (x => x == null ? '-' : x.toLocaleString());
const fmt1 = window.fmt1 || (x => x==null?'-': (Math.round(x*10)/10).toString());
// Use window.parseISO provided by core; do not declare a global named parseISO here to avoid collisions

// --------------------------- Filters UI --------------------------------
// (legacy code removed)

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

// Training streak & last ingested now exposed by core (avoid dup)

// (duplicate style block removed; core injects once)

// Ensure workout modal functions are exposed early if calendar loaded first
if(typeof window.showWorkoutDetail !== 'function'){
  window.showWorkoutDetail = (...args)=>{
    // replaced later when real function defined
    console.warn('showWorkoutDetail placeholder invoked before definition');
  };
}

// ADVANCED ANALYTICS FUNCTIONS (Plotly extras)

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

// Expose other utilities if needed
window.shareWorkout = window.shareWorkout || shareWorkout;
