let chartInstance = null;
let currentRawData = null;
let cachedBackgroundDataset = null;

const loadBtn = document.getElementById('load-btn');
const carSelect = document.getElementById('car-select');
const trackSelect = document.getElementById('track-select');
const startSlider = document.getElementById('time-start');
const startVal = document.getElementById('start-val');
const windowSizeInput = document.getElementById('window-size');
const modelToggles = document.getElementById('model-toggles');
const checkboxList = document.getElementById('checkbox-list');

const zoomInBtn = document.getElementById('zoom-in');
const zoomOutBtn = document.getElementById('zoom-out');
const resetZoomBtn = document.getElementById('reset-zoom');

function initChart() {
    const ctx = document.getElementById('trajectoryChart').getContext('2d');
    chartInstance = new Chart(ctx, {
        type: 'scatter',
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 0 },
            elements: {
                point: { radius: 0 },
                line: { borderWidth: 2, tension: 0 }
            },
            showLine: true,
            scales: {
                x: {
                    type: 'linear', position: 'bottom',
                    grid: { color: '#f0f0f0' },
                    ticks: { color: '#737373' }
                },
                y: {
                    type: 'linear',
                    grid: { color: '#f0f0f0' },
                    ticks: { color: '#737373' }
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false },
                zoom: {
                    pan: {
                        enabled: true,
                        mode: 'xy'
                    },
                    zoom: {
                        wheel: { enabled: false },
                        pinch: { enabled: false },
                        mode: 'xy'
                    }
                }
            }
        }
    });
}

function integrateTrajectory(vx, vy, yaw_rate, dt, x0, y0, phi0) {
    const n = vx.length;
    const path = [];
    let curr_x = x0;
    let curr_y = y0;
    let curr_phi = phi0;
    
    path.push({x: curr_x, y: curr_y});
    
    for (let i = 0; i < n; i++) {
        if (isNaN(vx[i]) || isNaN(vy[i]) || isNaN(yaw_rate[i])) break;
        
        let next_phi = curr_phi + yaw_rate[i] * dt;
        let next_x = curr_x + (vx[i] * Math.cos(curr_phi) - vy[i] * Math.sin(curr_phi)) * dt;
        let next_y = curr_y + (vx[i] * Math.sin(curr_phi) + vy[i] * Math.cos(curr_phi)) * dt;
        
        let dx = next_x - curr_x;
        let dy = next_y - curr_y;
        
        // Break early if integration blows up (star artifacting)
        if (Math.abs(dx) > 100 || Math.abs(dy) > 100 || Math.abs(vx[i]) > 200 || Math.abs(vy[i]) > 200) {
            break;
        }

        path.push({x: next_x, y: next_y});
        curr_x = next_x; curr_y = next_y; curr_phi = next_phi;
    }
    return path;
}

function updateChartSlice() {
    if (!currentRawData) return;

    let startPercent = parseInt(startSlider.value);
    let size = parseFloat(windowSizeInput.value);
    if(isNaN(size) || size <= 0) size = 2; // Default fallback to 2%
    
    let endPercent = Math.min(100, startPercent + size);

    startVal.textContent = startPercent + '%';

    const datasets = [];
    
    // Total dataset bounds based *strictly* on physics model length (not raw CSV)
    const max_N = currentRawData.ground_truth.vx.length + 5;
    
    // Safety clamp just in case sizes exceed the map
    const globalStart = Math.floor((startPercent / 100) * max_N);
    let potentialEnd = Math.floor((endPercent / 100) * max_N);
    const globalEnd = Math.min(max_N, potentialEnd);
    
    // 1. Draw the entire GPS Track as a background map
    if (currentRawData.gps.x.length > 0) {
        if (!cachedBackgroundDataset) {
            const fullGpsData = [];
            for(let i = 0; i < currentRawData.gps.x.length; i++) {
                fullGpsData.push({x: currentRawData.gps.x[i], y: currentRawData.gps.y[i]});
            }
            cachedBackgroundDataset = {
                label: 'Track Map Layout',
                data: fullGpsData,
                borderColor: '#f2f2f2',
                borderWidth: 10,
                pointRadius: 0,
                order: 10
            };
        }
        datasets.push(cachedBackgroundDataset);
        
        const activeGpsData = [];
        for(let i = globalStart; i < globalEnd; i++) {
            if(i < currentRawData.gps.x.length) {
                activeGpsData.push({x: currentRawData.gps.x[i], y: currentRawData.gps.y[i]});
            }
        }
        datasets.push({
            label: 'Active GPS Region',
            data: activeGpsData,
            borderColor: '#d4d4d4',
            borderWidth: 10,
            pointRadius: 0,
            order: 5
        });
    }

    // Models start 5 indexes behind GPS due to horizon delay
    const m_start = Math.max(0, globalStart - 5);
    const m_end = Math.max(0, globalEnd - 5);
    
    let x0 = 0.0, y0 = 0.0, phi0 = 0.0;
    if (currentRawData.gps.x.length > 0) {
        let exactIdx = m_start + 5;
        if (exactIdx >= currentRawData.gps.x.length) exactIdx = currentRawData.gps.x.length - 1;
        x0 = currentRawData.gps.x[exactIdx];
        y0 = currentRawData.gps.y[exactIdx];
        phi0 = currentRawData.gps.phi[exactIdx];
    }
    
    // Get Toggle states
    const checkboxes = document.querySelectorAll('.model-cb');
    const checkedNames = new Set();
    checkboxes.forEach(cb => {
        if (cb.checked) checkedNames.add(cb.value);
    });

    if (checkedNames.has(currentRawData.ground_truth.name)) {
        let gt_vx = currentRawData.ground_truth.vx.slice(m_start, m_end);
        let gt_vy = currentRawData.ground_truth.vy.slice(m_start, m_end);
        let gt_yaw = currentRawData.ground_truth.yaw_rate.slice(m_start, m_end);
        datasets.push({
            label: currentRawData.ground_truth.name,
            data: integrateTrajectory(gt_vx, gt_vy, gt_yaw, 0.04, x0, y0, phi0),
            borderColor: '#111',
            borderDash: [5, 5],
            borderWidth: 2,
            order: 2
        });
    }

    currentRawData.models.forEach(m => {
        if (checkedNames.has(m.name)) {
            let m_vx = m.vx.slice(m_start, m_end);
            let m_vy = m.vy.slice(m_start, m_end);
            let m_yaw = m.yaw_rate.slice(m_start, m_end);
            datasets.push({
                label: m.name,
                data: integrateTrajectory(m_vx, m_vy, m_yaw, 0.04, x0, y0, phi0),
                borderColor: m.color,
                borderWidth: 2,
                order: 1
            });
        }
    });

    chartInstance.data.datasets = datasets;
    chartInstance.update('none');
}

function buildCheckboxes(data) {
    checkboxList.innerHTML = '';
    
    const entries = [{name: data.ground_truth.name, color: '#111'}, ...data.models];
    
    entries.forEach(m => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        div.innerHTML = `
            <input type="checkbox" class="model-cb" value="${m.name}" checked>
            <span style="display:inline-block; width:10px; height:10px; background:${m.color}; border-radius:2px;"></span>
            <span>${m.name}</span>
        `;
        checkboxList.appendChild(div);
    });
    
    document.querySelectorAll('.model-cb').forEach(cb => {
        cb.addEventListener('change', updateChartSlice);
    });
    
    modelToggles.style.display = 'flex';
}

async function fetchTrajectories() {
    const car = carSelect.value;
    const track = trackSelect.value;
    loadBtn.textContent = "loading...";
    
    try {
        cachedBackgroundDataset = null; // Clear cache on load
        
        const response = await fetch(`/api/trajectories?car=${car}&track=${track}`);
        if (!response.ok) throw new Error("Failed to fetch");
        currentRawData = await response.json();
        
        let allX = currentRawData.gps.x.length > 0 ? currentRawData.gps.x : currentRawData.ground_truth.vx; 
        if (currentRawData.gps.x.length > 0) {
            let minX = Math.min(...currentRawData.gps.x);
            let maxX = Math.max(...currentRawData.gps.x);
            let minY = Math.min(...currentRawData.gps.y);
            let maxY = Math.max(...currentRawData.gps.y);
            let spanX = maxX - minX;
            let spanY = maxY - minY;
            
            chartInstance.options.scales.x.min = minX - spanX * 0.1;
            chartInstance.options.scales.x.max = maxX + spanX * 0.1;
            chartInstance.options.scales.y.min = minY - spanY * 0.1;
            chartInstance.options.scales.y.max = maxY + spanY * 0.1;
        } else {
            // ETHZ Fallback without bounds
            chartInstance.options.scales.x.min = -2;
            chartInstance.options.scales.x.max = 2;
            chartInstance.options.scales.y.min = -2;
            chartInstance.options.scales.y.max = 2;
        }
        chartInstance.update();
        
        buildCheckboxes(currentRawData);
        updateChartSlice();
    } catch (err) {
        console.error(err);
        alert("Error loading trajectories.");
    } finally {
        loadBtn.textContent = "run simulation";
    }
}

startSlider.addEventListener('input', updateChartSlice);
windowSizeInput.addEventListener('input', updateChartSlice);
loadBtn.addEventListener('click', fetchTrajectories);

zoomInBtn.addEventListener('click', () => chartInstance.zoom(1.2));
zoomOutBtn.addEventListener('click', () => chartInstance.zoom(0.8));
resetZoomBtn.addEventListener('click', () => chartInstance.resetZoom());

window.onload = initChart;
