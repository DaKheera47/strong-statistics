// Utility functions
async function fetchJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(await r.text());
  return r.json();
}

function formatDate(dateStr) {
  return new Date(dateStr).toLocaleDateString('en-US', { 
    month: 'short', 
    day: 'numeric' 
  });
}

// Color palette for consistency
const colors = {
  primary: '#2c3e50',
  secondary: '#3498db', 
  success: '#27ae60',
  warning: '#f39c12',
  danger: '#e74c3c',
  info: '#8e44ad',
  light: '#95a5a6',
  dark: '#2c3e50'
};

const exerciseColors = [
  '#3498db', '#e74c3c', '#27ae60', '#f39c12', '#8e44ad', '#16a085'
];

// Progressive Overload Chart - Most important chart
async function loadProgressiveOverload() {
  try {
    const data = await fetchJSON('/api/progressive-overload');
    
    if (!data || Object.keys(data).length === 0) {
      Plotly.newPlot('progressiveOverloadChart', [], {
        title: 'No strength progression data available',
        font: { size: 14 }
      });
      return;
    }

    const traces = [];
    let colorIndex = 0;

    // Create traces for each exercise - focus on estimated 1RM
    Object.entries(data).forEach(([exercise, sessions]) => {
      if (sessions.length > 0) {
        traces.push({
          x: sessions.map(s => s.date),
          y: sessions.map(s => s.estimated_1rm),
          type: 'scatter',
          mode: 'lines+markers',
          name: exercise,
          line: { width: 3 },
          marker: { size: 8 },
          hovertemplate: `<b>${exercise}</b><br>` +
                        `Date: %{x}<br>` +
                        `Est. 1RM: %{y} kg<br>` +
                        `<extra></extra>`
        });
      }
      colorIndex++;
    });

    const layout = {
      title: {
        text: 'Estimated 1-Rep Max Progression',
        font: { size: 18 }
      },
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { 
        title: 'Estimated 1RM (kg)',
        zeroline: false
      },
      hovermode: 'x unified',
      legend: { 
        orientation: 'h',
        y: -0.2
      },
      margin: { t: 60, b: 100 }
    };

    Plotly.newPlot('progressiveOverloadChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading progressive overload:', error);
  }
}

// Volume Progression Chart
async function loadVolumeProgression() {
  try {
    const data = await fetchJSON('/api/volume-progression');
    
    if (!data || data.length === 0) return;

    const traces = [
      {
        x: data.map(d => d.date),
        y: data.map(d => d.volume),
        type: 'bar',
        name: 'Session Volume',
        marker: { color: colors.secondary, opacity: 0.7 },
        hovertemplate: 'Date: %{x}<br>Volume: %{y} kg<extra></extra>'
      },
      {
        x: data.map(d => d.date),
        y: data.map(d => d.volume_7day_avg),
        type: 'scatter',
        mode: 'lines',
        name: '7-Day Average',
        line: { width: 3, color: colors.danger },
        hovertemplate: 'Date: %{x}<br>7-Day Avg: %{y:.1f} kg<extra></extra>'
      }
    ];

    const layout = {
      title: 'Training Volume Over Time',
      xaxis: { 
        title: 'Date',
        tickangle: -45,
        type: 'date'
      },
      yaxis: { title: 'Volume (kg)' },
      hovermode: 'x unified',
      legend: { orientation: 'h', y: -0.15 }
    };

    Plotly.newPlot('volumeProgressionChart', traces, layout, { 
      responsive: true,
      displayModeBar: false 
    });

  } catch (error) {
    console.error('Error loading volume progression:', error);
  }
}

// Personal Records Timeline
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
async function loadExercisesList() {
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

// Load metadata
async function refreshMeta() {
  try {
    const health = await fetchJSON('/health');
    if (health.last_ingested_at) {
      const lastIngested = new Date(health.last_ingested_at).toLocaleString();
      document.getElementById('lastIngested').textContent = `Last update: ${lastIngested}`;
    }
  } catch (error) {
    console.error('Error loading metadata:', error);
  }
}

// Initialize dashboard
(async function init() {
  console.log('Initializing improved lifting dashboard...');
  
  try {
    // Load all charts in parallel for better performance
    await Promise.all([
      loadProgressiveOverload(),
      loadVolumeProgression(),
      loadPersonalRecords(),
      loadTrainingConsistency(),
      loadStrengthBalance(),
      loadExercisesList(),
      refreshMeta()
    ]);
    
    // Initialize exercise analysis with empty state
    loadExerciseAnalysis(null);
    
    console.log('Dashboard loaded successfully!');
  } catch (error) {
    console.error('Error initializing dashboard:', error);
  }
})();
