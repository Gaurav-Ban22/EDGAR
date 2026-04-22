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

/**
 * Integrate body-frame velocities into global XY trajectory.
 * 
 * FIX: Uses GPS heading (phi) from the aligned poses array at each step
 * instead of integrating yaw_rate. This eliminates cumulative heading
 * drift which was causing 50-83m position error over 120s.
 * 
 * @param {number[]} vx - longitudinal velocity (m/s)
 * @param {number[]} vy - lateral velocity (m/s)  
 * @param {number[]} yaw_rate - yaw rate (rad/s) - unused now, kept for API compat
 * @param {number} dt - timestep (s)
 * @param {number} x0 - initial x position
 * @param {number} y0 - initial y position
 * @param {number} phi0 - initial heading
 * @param {number[]} gps_phi - GPS heading at each step (from aligned poses array)
 * @param {number} gps_phi_offset - index offset into gps_phi array
 */
function integrateTrajectory(vx, vy, yaw_rate, dt, x0, y0, phi0, gps_phi, gps_phi_offset) {
    const n = vx.length;
    const path = [];
    let curr_x = x0;
    let curr_y = y0;
    let curr_phi = phi0;
    const useGpsPhi = gps_phi && gps_phi.length > 0;
    
    path.push({x: curr_x, y: curr_y});
    
    for (let i = 0; i < n; i++) {
        if (isNaN(vx[i]) || isNaN(vy[i]) || isNaN(yaw_rate[i])) break;
        
        // Use GPS heading if available — this prevents cumulative yaw drift
        if (useGpsPhi) {
            let phiIdx = gps_phi_offset + i;
            if (phiIdx >= 0 && phiIdx < gps_phi.length) {
                curr_phi = gps_phi[phiIdx];
            } else {
                // Fall back to yaw_rate integration if past GPS data
                curr_phi = curr_phi + yaw_rate[i] * dt;
            }
        } else {
            curr_phi = curr_phi + yaw_rate[i] * dt;
        }

        let next_x = curr_x + (vx[i] * Math.cos(curr_phi) - vy[i] * Math.sin(curr_phi)) * dt;
        let next_y = curr_y + (vx[i] * Math.sin(curr_phi) + vy[i] * Math.cos(curr_phi)) * dt;
        
        let dx = next_x - curr_x;
        let dy = next_y - curr_y;
        
        // Break early if integration blows up (star artifacting)
        if (Math.abs(dx) > 100 || Math.abs(dy) > 100 || Math.abs(vx[i]) > 200 || Math.abs(vy[i]) > 200) {
            break;
        }

        path.push({x: next_x, y: next_y});
        curr_x = next_x; curr_y = next_y;
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
    
    // The horizon offset: NPZ labels[i] corresponds to poses[i + horizon]
    // where horizon = 5. So model/GT sample i maps to GPS pose at i + 5.
    const HORIZON = 5;
    
    // Total dataset bounds based on model/GT length
    const gt_len = currentRawData.ground_truth.vx.length;
    
    const globalStart = Math.floor((startPercent / 100) * gt_len);
    let potentialEnd = Math.floor((endPercent / 100) * gt_len);
    const globalEnd = Math.min(gt_len, potentialEnd);
    
    // The GPS poses are aligned with the NPZ data.
    // poses[i + HORIZON] corresponds to model/GT sample i.
    // So for model sample globalStart, the GPS position is at gps index (globalStart + HORIZON).
    
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
        
        // Active GPS region — use aligned indices
        const gpsActiveStart = globalStart + HORIZON;
        const gpsActiveEnd = Math.min(currentRawData.gps.x.length, globalEnd + HORIZON);
        const activeGpsData = [];
        for(let i = gpsActiveStart; i < gpsActiveEnd; i++) {
            activeGpsData.push({x: currentRawData.gps.x[i], y: currentRawData.gps.y[i]});
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

    // Initial position from GPS at the aligned index
    let x0 = 0.0, y0 = 0.0, phi0 = 0.0;
    const gpsStartIdx = globalStart + HORIZON;
    if (currentRawData.gps.x.length > 0 && gpsStartIdx < currentRawData.gps.x.length) {
        x0 = currentRawData.gps.x[gpsStartIdx];
        y0 = currentRawData.gps.y[gpsStartIdx];
        phi0 = currentRawData.gps.phi[gpsStartIdx];
    }
    
    // Get Toggle states
    const checkboxes = document.querySelectorAll('.model-cb');
    const checkedNames = new Set();
    checkboxes.forEach(cb => {
        if (cb.checked) checkedNames.add(cb.value);
    });

    if (checkedNames.has(currentRawData.ground_truth.name)) {
        let gt_vx = currentRawData.ground_truth.vx.slice(globalStart, globalEnd);
        let gt_vy = currentRawData.ground_truth.vy.slice(globalStart, globalEnd);
        let gt_yaw = currentRawData.ground_truth.yaw_rate.slice(globalStart, globalEnd);
        datasets.push({
            label: currentRawData.ground_truth.name,
            data: integrateTrajectory(gt_vx, gt_vy, gt_yaw, 0.04, x0, y0, phi0,
                                      currentRawData.gps.phi, gpsStartIdx),
            borderColor: '#111',
            borderDash: [5, 5],
            borderWidth: 2,
            order: 2
        });
    }

    const forceGpsPhi = document.getElementById('force-gps-phi').checked;

    currentRawData.models.forEach(m => {
        if (checkedNames.has(m.name)) {
            let m_vx = m.vx.slice(globalStart, globalEnd);
            let m_vy = m.vy.slice(globalStart, globalEnd);
            let m_yaw = m.yaw_rate.slice(globalStart, globalEnd);
            
            let modelRefPhi = forceGpsPhi ? currentRawData.gps.phi : null;
            let modelRefIdx = forceGpsPhi ? gpsStartIdx : 0;
            
            datasets.push({
                label: m.name,
                data: integrateTrajectory(m_vx, m_vy, m_yaw, 0.04, x0, y0, phi0,
                                          modelRefPhi, modelRefIdx),
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
        const isTarget = ['Ground Truth (IMU)', 'Hybrid PCNN+PINN'].includes(m.name);
        const checkedAttr = isTarget ? 'checked' : '';
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        div.innerHTML = `
            <input type="checkbox" class="model-cb" value="${m.name}" ${checkedAttr}>
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
document.getElementById('force-gps-phi').addEventListener('change', updateChartSlice);
loadBtn.addEventListener('click', fetchTrajectories);

zoomInBtn.addEventListener('click', () => chartInstance.zoom(1.2));
zoomOutBtn.addEventListener('click', () => chartInstance.zoom(0.8));
resetZoomBtn.addEventListener('click', () => chartInstance.resetZoom());

window.onload = initChart;
