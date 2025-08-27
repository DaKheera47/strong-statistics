// Strong-inspired Analytics Dashboard

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

// Initialize all charts when page loads
document.addEventListener('DOMContentLoaded', function() {
    console.log('Initializing improved lifting dashboard...');
    
    // Load original charts
    console.log('Loading original charts...');
    loadProgressiveOverload();
    loadVolumeProgression(); 
    loadPersonalRecords();
    loadTrainingConsistency();
    loadStrengthBalance();
    loadExerciseOptions();
    
    // Load Strong-inspired analytics
    console.log('Loading Strong-inspired analytics...');
    loadPersonalRecordsTable();
    loadTrainingCalendar();
    loadMuscleGroupBalance();
    loadBodyMeasurements();
    loadTrainingStreak();
    
    // Load metadata
    console.log('Loading metadata...');
    loadLastIngested();
    
    console.log('Dashboard loaded successfully!');
});

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
      hovertemplate: 'Date: %{x}<br>Exercises: %{y}<br>Volume: %{customdata.volume} kg<br>Sets: %{customdata.sets}<extra></extra>',
      customdata: data.map(d => ({ volume: d.total_volume, sets: d.total_sets }))
    };

    const layout = {
      title: '📅 Training Calendar - Workout Intensity',
      xaxis: { title: 'Date', type: 'date' },
      yaxis: { title: 'Exercises Performed' },
      showlegend: false
    };

    Plotly.newPlot('calendarChart', [trace], layout, { responsive: true, displayModeBar: false });

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
