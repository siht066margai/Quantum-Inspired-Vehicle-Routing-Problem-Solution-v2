// Customer Color Palette: C1 Blue, C2 Orange, C3 Green, C4 Purple, C5 Pink, etc.
const CUSTOMER_PALETTE = [
    { name: "Blue", hex: "#3b82f6", border: "#2563eb", glow: "rgba(59, 130, 246, 0.45)" },
    { name: "Orange", hex: "#f97316", border: "#ea580c", glow: "rgba(249, 115, 22, 0.45)" },
    { name: "Green", hex: "#10b981", border: "#059669", glow: "rgba(16, 185, 129, 0.45)" },
    { name: "Purple", hex: "#a855f7", border: "#9333ea", glow: "rgba(168, 85, 247, 0.45)" },
    { name: "Pink", hex: "#ec4899", border: "#db2777", glow: "rgba(236, 72, 153, 0.45)" },
    { name: "Cyan", hex: "#06b6d4", border: "#0891b2", glow: "rgba(6, 182, 212, 0.45)" },
    { name: "Yellow", hex: "#eab308", border: "#ca8a04", glow: "rgba(234, 179, 8, 0.45)" },
    { name: "Red", hex: "#f43f5e", border: "#e11d48", glow: "rgba(244, 63, 94, 0.45)" },
    { name: "Teal", hex: "#14b8a6", border: "#0d9488", glow: "rgba(20, 184, 166, 0.45)" },
    { name: "Indigo", hex: "#6366f1", border: "#4f46e5", glow: "rgba(99, 102, 241, 0.45)" },
];

class TrafficApp {
    constructor() {
        this.networkData = null;
        this.ws = null;
        this.canvas = document.getElementById('traffic-map');
        this.ctx = this.canvas.getContext('2d');

        // Viewport transform (pan & zoom)
        this.zoom = 1.0;
        this.panX = 0;
        this.panY = 0;
        this.isDragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;

        // Interaction state
        this.pickMode = null; // 'origin' | 'c1' | 'c2' ... | null
        this.customerCount = 3; // Default 3 customer orders
        this.hoveredItem = null;

        // Fleet configuration (initially 0, user enters values)
        this.fleetCount = 0;
        this.vehicleCapacity = 0.0;
        this.ordersList = [];

        // Live simulation state
        this.simRunning = false;
        this.simPaused = false;
        this.simTime = 0.0;
        this.vehicles = [];
        this.congestedEdges = new Map();
        this.activeRoute = null;
        this.activeIncidents = [];
        this.lastComparisonData = null;

        this.initUI();
        this.fetchNetworkData();
        this.connectWebSocket();
        this.loadBenchmarkHistory();
    }

    initUI() {
        // Window resize
        window.addEventListener('resize', () => this.resizeCanvas());
        this.resizeCanvas();

        // Header View Nav Tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                const targetView = e.currentTarget.dataset.view;
                this.switchView(targetView, e.currentTarget);
            });
        });

        // SIM Control buttons
        document.getElementById('btn-start').addEventListener('click', () => this.sendSimCmd('start'));
        document.getElementById('btn-pause').addEventListener('click', () => this.sendSimCmd('pause'));
        document.getElementById('btn-step').addEventListener('click', () => this.sendSimCmd('step'));
        document.getElementById('btn-stop').addEventListener('click', () => this.sendSimCmd('stop'));

        // Customer & Fleet inputs
        this.renderCustomerControls();
        const btnAdd = document.getElementById('btn-add-customer');
        if (btnAdd) btnAdd.addEventListener('click', () => this.addCustomer());
        const btnRemove = document.getElementById('btn-remove-customer');
        if (btnRemove) btnRemove.addEventListener('click', () => this.removeCustomer());
        const btnGen = document.getElementById('btn-generate-orders');
        if (btnGen) btnGen.addEventListener('click', () => this.generateSyntheticOrders());

        const fleetCountInput = document.getElementById('fleet-count-input');
        if (fleetCountInput) {
            fleetCountInput.addEventListener('input', (e) => {
                this.fleetCount = parseInt(e.target.value, 10) || 0;
                this.updateVRPBadge();
                this.invalidatePreviousSolution();
            });
            fleetCountInput.addEventListener('change', (e) => {
                this.fleetCount = parseInt(e.target.value, 10) || 0;
                this.updateVRPBadge();
                this.invalidatePreviousSolution();
            });
        }
        const capInput = document.getElementById('vehicle-cap-input');
        if (capInput) {
            capInput.addEventListener('input', (e) => {
                this.vehicleCapacity = parseFloat(e.target.value) || 0.0;
                this.invalidatePreviousSolution();
            });
            capInput.addEventListener('change', (e) => {
                this.vehicleCapacity = parseFloat(e.target.value) || 0.0;
                this.invalidatePreviousSolution();
            });
        }

        const origSelect = document.getElementById('origin-select');
        if (origSelect) {
            origSelect.addEventListener('change', () => this.invalidatePreviousSolution());
        }

        // Pick buttons in toolbar
        document.getElementById('btn-pick-origin').addEventListener('click', (e) => this.togglePickMode('origin', e.currentTarget));
        const btnPickCust = document.getElementById('btn-pick-customer');
        if (btnPickCust) btnPickCust.addEventListener('click', (e) => this.togglePickMode('c1', e.currentTarget));
        document.getElementById('btn-reset-view').addEventListener('click', () => this.fitBounds());

        // Solver buttons
        document.getElementById('btn-qpso-route').addEventListener('click', () => this.solveVRP('qpso'));
        const btnAlns = document.getElementById('btn-alns-route');
        if (btnAlns) btnAlns.addEventListener('click', () => this.solveVRP('alns'));
        const btnHgs = document.getElementById('btn-hgs-route');
        if (btnHgs) btnHgs.addEventListener('click', () => this.solveVRP('hgs'));
        document.getElementById('btn-calc-route').addEventListener('click', () => this.solveVRP('dijkstra'));
        document.getElementById('btn-qaoa-route').addEventListener('click', () => this.solveVRP('qaoa'));
        document.getElementById('btn-compare-route').addEventListener('click', () => this.solveVRP('compare'));

        // Research View / Proofs Button & Tabs
        const btnOpenAnalysis = document.getElementById('btn-open-analysis');
        if (btnOpenAnalysis) {
            btnOpenAnalysis.addEventListener('click', () => {
                this.switchView('research', document.getElementById('tab-btn-research'));
            });
        }

        document.querySelectorAll('.analysis-tab').forEach(button => {
            button.addEventListener('click', () => this.activateAnalysisTab(button.dataset.analysisTab));
        });

        // Benchmark Hub
        const btnBenchmark = document.getElementById('btn-run-full-benchmark');
        if (btnBenchmark) {
            btnBenchmark.addEventListener('click', () => this.runBenchmarkSuite());
        }

        const btnScalability = document.getElementById('btn-run-scalability');
        if (btnScalability) {
            btnScalability.addEventListener('click', () => this.runScalabilitySweep());
        }

        const btnRefreshHistory = document.getElementById('btn-refresh-history');
        if (btnRefreshHistory) {
            btnRefreshHistory.addEventListener('click', () => this.loadBenchmarkHistory());
        }

        const btnProjectSolution = document.getElementById('btn-project-solution');
        if (btnProjectSolution) {
            btnProjectSolution.addEventListener('click', () => this.projectSolutionOnMap());
        }

        const selectInspectSolution = document.getElementById('inspect-solution-select');
        if (selectInspectSolution) {
            selectInspectSolution.addEventListener('change', () => this.updateInspectedSolutionDetails());
        }

        // Incident buttons
        document.getElementById('btn-inject-incident').addEventListener('click', () => this.injectIncident());
        document.getElementById('btn-clear-incident').addEventListener('click', () => this.clearIncident());

        // Canvas Pan / Zoom / Click
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', () => this.onMouseUp());
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
    }

    switchView(viewName, activeTabEl) {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.view-panel').forEach(v => v.classList.remove('active'));

        if (activeTabEl) {
            activeTabEl.classList.add('active');
        } else {
            const btn = document.getElementById(`tab-btn-${viewName}`);
            if (btn) btn.classList.add('active');
        }

        const targetPanel = document.getElementById(`view-${viewName}`);
        if (targetPanel) {
            targetPanel.classList.add('active');
        }

        if (viewName === 'research') {
            this.renderResearchView();
        } else if (viewName === 'benchmark') {
            this.loadBenchmarkHistory();
        }
    }

    activateAnalysisTab(tabId) {
        document.querySelectorAll('.analysis-tab').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.analysisTab === tabId);
        });
        document.querySelectorAll('.analysis-panel').forEach(panel => {
            panel.classList.toggle('active', panel.dataset.analysisPanel === tabId);
        });
    }

    invalidatePreviousSolution() {
        this.activeRoute = null;
        this.lastComparisonData = null;
        const resElem = document.getElementById('route-results');
        if (resElem) resElem.classList.add('hidden');
        const frList = document.getElementById('fleet-routes-list');
        if (frList) frList.classList.add('hidden');
        const btnAnalysis = document.getElementById('btn-open-analysis');
        if (btnAnalysis) btnAnalysis.classList.add('hidden');
        this.draw();
    }

    renderCustomerControls() {
        const container = document.getElementById('customer-selects-container');
        if (!container) return;
        container.innerHTML = '';

        for (let i = 1; i <= this.customerCount; i++) {
            const col = CUSTOMER_PALETTE[(i - 1) % CUSTOMER_PALETTE.length];
            const div = document.createElement('div');
            div.className = 'form-group customer-group';
            div.setAttribute('data-dest-index', i);
            div.style.borderLeft = `3px solid ${col.hex}`;
            div.style.paddingLeft = '8px';
            div.style.marginBottom = '8px';
            div.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <label for="dest${i}-select" style="margin: 0; display: flex; align-items: center; gap: 6px;">
                        <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: ${col.hex}; box-shadow: 0 0 5px ${col.glow};"></span>
                        Order ${i} (<strong style="color: ${col.hex}">C${i}</strong>):
                    </label>
                    <button type="button" class="btn btn-outline btn-sm btn-pick-cust" data-cust-index="${i}" style="padding: 1px 6px; font-size: 10px; border-color: ${col.hex}; color: ${col.hex};">📍 Pick C${i}</button>
                </div>
                <select id="dest${i}-select" class="form-control customer-select"></select>
                <div style="display: flex; align-items: center; justify-content: space-between; margin-top: 4px; padding: 2px 4px; background: rgba(255,255,255,0.02); border-radius: 4px;">
                    <span style="font-size: 11px; color: var(--text-secondary);">Demand / Packages:</span>
                    <input type="number" id="dest${i}-demand" class="form-control" value="1" min="1" max="100" style="width: 70px; padding: 2px 6px; font-size: 11px; height: 24px;">
                </div>
            `;
            container.appendChild(div);

            const demInp = div.querySelector(`#dest${i}-demand`);
            if (demInp) {
                demInp.addEventListener('input', () => this.invalidatePreviousSolution());
                demInp.addEventListener('change', () => this.invalidatePreviousSolution());
            }
            const selEl = div.querySelector(`#dest${i}-select`);
            if (selEl) {
                selEl.addEventListener('change', () => this.invalidatePreviousSolution());
            }
        }

        container.querySelectorAll('.btn-pick-cust').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(e.currentTarget.dataset.custIndex, 10);
                this.togglePickMode(`c${idx}`, e.currentTarget);
            });
        });

        if (this.networkData) {
            this.populateSelects();
        }
        this.updateVRPBadge();
    }

    addCustomer() {
        if (this.customerCount < 15) {
            this.customerCount++;
            this.invalidatePreviousSolution();
            this.renderCustomerControls();
            setTimeout(() => {
                const newElem = document.getElementById(`dest${this.customerCount}-select`);
                if (newElem) {
                    newElem.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            }, 50);
        }
    }

    removeCustomer() {
        if (this.customerCount > 1) {
            this.customerCount--;
            this.invalidatePreviousSolution();
            this.renderCustomerControls();
        }
    }

    async generateSyntheticOrders() {
        try {
            const res = await fetch('/api/orders/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ count: this.customerCount })
            });
            const data = await res.json();
            if (data.success && data.orders) {
                this.ordersList = data.orders;
                data.orders.forEach((ord, idx) => {
                    const sel = document.getElementById(`dest${idx+1}-select`);
                    if (sel) sel.value = ord.node_id;
                    const demInp = document.getElementById(`dest${idx+1}-demand`);
                    if (demInp) demInp.value = ord.demand;
                });
                this.invalidatePreviousSolution();
                this.draw();
            }
        } catch (e) {
            console.error("Failed to generate synthetic orders:", e);
        }
    }

    updateVRPBadge() {
        const badge = document.getElementById('vrp-scenario-badge');
        if (badge) {
            badge.innerText = `1 Depot + ${this.customerCount} Orders + ${this.fleetCount} Vehicles`;
        }
    }

    async fetchNetworkData() {
        try {
            const response = await fetch('/api/network');
            this.networkData = await response.json();
            this.populateSelects();
            this.fitBounds();
            this.draw();
        } catch (e) {
            console.error("Failed to fetch network topology:", e);
        }
    }

    populateSelects() {
        if (!this.networkData || !this.networkData.nodes) return;
        const nodes = Object.keys(this.networkData.nodes).sort();

        const origSelect = document.getElementById('origin-select');
        const prevOrig = origSelect ? origSelect.value : null;
        if (origSelect) {
            origSelect.innerHTML = '';
            nodes.forEach(n => {
                const opt = document.createElement('option');
                opt.value = n;
                opt.textContent = `Node ${n}`;
                origSelect.appendChild(opt);
            });
            origSelect.value = prevOrig && nodes.includes(prevOrig) ? prevOrig : nodes[0];
        }

        for (let i = 1; i <= this.customerCount; i++) {
            const destSelect = document.getElementById(`dest${i}-select`);
            if (destSelect) {
                const prevVal = destSelect.value;
                destSelect.innerHTML = '';
                nodes.forEach(n => {
                    const opt = document.createElement('option');
                    opt.value = n;
                    opt.textContent = `Node ${n}`;
                    destSelect.appendChild(opt);
                });
                if (prevVal && nodes.includes(prevVal)) {
                    destSelect.value = prevVal;
                } else {
                    const defaultIndex = Math.min(i * 15, nodes.length - 1);
                    destSelect.value = nodes[defaultIndex];
                }
            }
        }

        const incSelect = document.getElementById('incident-edge-select');
        if (incSelect && this.networkData.edges) {
            incSelect.innerHTML = '';
            this.networkData.edges.forEach(e => {
                const opt = document.createElement('option');
                opt.value = e.id;
                opt.textContent = `Edge ${e.id} (${e.from} ➔ ${e.to}, ${Math.round(e.length)}m)`;
                incSelect.appendChild(opt);
            });
        }
    }

    async solveVRP(algorithm) {
        const fleetInput = parseInt(document.getElementById('fleet-count-input').value || '0', 10);
        const capInput = parseFloat(document.getElementById('vehicle-cap-input').value || '0');

        if (fleetInput <= 0 || capInput <= 0) {
            alert("Please specify Number of Vehicles (X > 0) and Capacity per Vehicle (Cv > 0) in the Fleet Vehicles Setup section before calculating routes.");
            return;
        }

        this.fleetCount = fleetInput;
        this.vehicleCapacity = capInput;

        const originNode = document.getElementById('origin-select').value;
        const destNodes = [];
        const customerDemands = {};
        const orders = [];
        for (let i = 1; i <= this.customerCount; i++) {
            const sel = document.getElementById(`dest${i}-select`);
            const demandInput = document.getElementById(`dest${i}-demand`);
            const demandVal = demandInput ? (parseFloat(demandInput.value) || 1.0) : 1.0;
            if (sel && sel.value) {
                destNodes.push(sel.value);
                customerDemands[sel.value] = demandVal;
                orders.push({
                    order_id: `C${i}`,
                    node_id: sel.value,
                    demand: demandVal,
                });
            }
        }

        // Button loading indicators
        const btnMap = {
            'qpso': { id: 'btn-qpso-route', loading: '⏳ QPSO Quantum Swarm Optimizing...' },
            'alns': { id: 'btn-alns-route', loading: '⏳ ALNS Metaheuristic Optimizing...' },
            'hgs': { id: 'btn-hgs-route', loading: '⏳ HGS Metaheuristic Optimizing...' },
            'qaoa': { id: 'btn-qaoa-route', loading: '⏳ QAOA Quantum Computing...' },
            'dijkstra': { id: 'btn-calc-route', loading: '⏳ Dijkstra Calculating...' },
            'compare': { id: 'btn-compare-route', loading: '⚡ Comparing QPSO, ALNS, HGS, QAOA...' },
        };
        const targetBtnInfo = btnMap[algorithm];
        const targetBtn = targetBtnInfo ? document.getElementById(targetBtnInfo.id) : null;
        let originalText = '';
        if (targetBtn) {
            originalText = targetBtn.innerHTML;
            targetBtn.innerHTML = targetBtnInfo.loading;
            targetBtn.disabled = true;
        }

        const payload = {
            origin_node: originNode,
            destination_nodes: destNodes,
            num_vehicles: this.fleetCount,
            vehicle_capacity: this.vehicleCapacity,
            customer_demands: customerDemands,
            orders: orders,
            algorithm: algorithm,
            alpha: 1.0,
            beta: 0.0,
            gamma: 0.0,
        };

        try {
            const res = await fetch(`/api/vrp/${algorithm}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data) {
                this.activeRoute = data;
                this.lastComparisonData = data;
                this.updateRouteMetrics(data);
                const btnAnalysis = document.getElementById('btn-open-analysis');
                if (btnAnalysis) btnAnalysis.classList.remove('hidden');
                this.draw();
            }
        } catch (e) {
            console.error(`Failed to solve VRP with ${algorithm}:`, e);
        } finally {
            if (targetBtn) {
                targetBtn.innerHTML = originalText;
                targetBtn.disabled = false;
            }
        }
    }

    updateRouteMetrics(data) {
        const container = document.getElementById('route-results');
        if (!container) return;
        container.classList.remove('hidden');

        const statusElem = document.getElementById('r-status');
        if (data.status === 'unsupported_size') {
            statusElem.innerHTML = `<span style="color: #f59e0b;">⚠️ ${data.algorithm} • Unsupported Size (N > 4)</span>`;
        } else if (data.status === 'infeasible' || data.feasible === false) {
            statusElem.innerHTML = `<span style="color: #ef4444;">❌ ${data.algorithm} • Mathematically Infeasible</span>`;
        } else {
            statusElem.innerHTML = `<span style="color: #10b981;">✅ ${data.algorithm} • Feasible Fleet Solution</span>`;
        }

        document.getElementById('r-timestamp').innerText = `t = ${(data.snapshot_timestamp || 0.0).toFixed(1)} s`;

        const fleetRoutes = data.fleet_routes || [];
        const numV = fleetRoutes.length || data.problem?.num_vehicles || this.fleetCount;
        const capV = data.problem?.vehicle_capacity || this.vehicleCapacity;
        document.getElementById('r-fleet-setup-val').innerText = `${numV} Vehicles (Capacity ${capV})`;

        const rawTime = data.total_travel_time || 0.0;
        const rawDist = data.total_distance || 0.0;
        const timeFormatted = rawTime >= 60 ? `${(rawTime / 60.0).toFixed(1)} min (${rawTime.toFixed(1)} s)` : `${rawTime.toFixed(1)} s`;
        const distFormatted = rawDist >= 1000 ? `${(rawDist / 1000.0).toFixed(2)} km (${rawDist.toFixed(1)} m)` : `${rawDist.toFixed(1)} m`;

        document.getElementById('r-tt').innerText = timeFormatted;
        document.getElementById('r-dist').innerText = distFormatted;
        document.getElementById('r-speed').innerText = `${(data.average_speed_ms || 0.0).toFixed(1)} m/s`;
        document.getElementById('r-comp').innerText = `${(data.computation_time_ms || 0.0).toFixed(2)} ms`;

        // Map node IDs to human labels and color info (Depot O, C1 Blue, C2 Orange, C3 Green...)
        const nodeLabels = {};
        const nodeToCust = {};
        const originVal = document.getElementById('origin-select')?.value;
        if (originVal) nodeLabels[originVal] = `Depot O`;

        for (let i = 1; i <= this.customerCount; i++) {
            const sel = document.getElementById(`dest${i}-select`);
            if (sel && sel.value) {
                const col = CUSTOMER_PALETTE[(i - 1) % CUSTOMER_PALETTE.length];
                nodeLabels[sel.value] = `Customer C${i}`;
                nodeToCust[sel.value] = {
                    index: i,
                    label: `C${i}`,
                    name: `Customer C${i}`,
                    color: col.hex,
                    border: col.border,
                    glow: col.glow,
                    colorName: col.name,
                };
            }
        }

        // Render fleet routes list dynamically with step-by-step guidance
        const frContainer = document.getElementById('fleet-routes-container');
        const frList = document.getElementById('fleet-routes-list');
        if (frContainer && frList) {
            if (data.error && (!fleetRoutes || fleetRoutes.length === 0)) {
                frList.classList.remove('hidden');
                frContainer.innerHTML = `
                    <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; padding: 12px; border-radius: 6px; font-size: 12px; line-height: 1.5;">
                        <strong>⚠️ Constraint / Feasibility Violation:</strong><br>${data.error}
                    </div>
                `;
            } else if (fleetRoutes.length > 0) {
                frList.classList.remove('hidden');
                let errorBanner = '';
                if (data.error && data.status === 'infeasible') {
                    errorBanner = `
                        <div style="background: rgba(239, 68, 68, 0.15); border: 1px solid #ef4444; color: #fca5a5; padding: 10px; border-radius: 6px; font-size: 12px; margin-bottom: 10px; line-height: 1.4;">
                            <strong>⛔ Infeasible Problem:</strong> ${data.error}
                        </div>
                    `;
                } else if (data.status === 'unsupported_size') {
                    errorBanner = `
                        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fde68a; padding: 10px; border-radius: 6px; font-size: 12px; margin-bottom: 10px; line-height: 1.4;">
                            <strong>⚡ Classical Simulation Limit:</strong> ${data.error}
                        </div>
                    `;
                }

                frContainer.innerHTML = errorBanner + fleetRoutes.map(fr => {
                    const assignedList = fr.assigned_orders || [];
                    const seq = fr.visit_sequence || [];
                    const isActive = fr.is_active && assignedList.length > 0;
                    const vColor = fr.color || '#3b82f6';
                    const capUtil = fr.capacity_utilization_pct || (fr.capacity > 0 ? ((fr.load_used / fr.capacity) * 100).toFixed(1) : 0);

                    if (!isActive) {
                        return `
                            <div class="fleet-route-item" style="border-left: 4px solid #4b5563; padding: 10px; margin-bottom: 8px; background: #0b0f17; border-radius: 6px; opacity: 0.85;">
                                <div class="v-head" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-weight: 700;">
                                    <span style="color: #9ca3af; font-size: 12px;">🚚 ${fr.vehicle_id}</span>
                                    <span class="badge" style="background: #374151; color: #9ca3af; font-size: 10px; padding: 2px 6px; border-radius: 3px;">Standby • Load: 0 / ${fr.capacity} pkgs (0% util)</span>
                                </div>
                                <div style="font-size: 11px; color: #9ca3af; line-height: 1.4; background: #111827; padding: 6px 8px; border-radius: 4px; border: 1px dashed #374151;">
                                    ⏸ <strong>Status: Standby / Unused</strong> (0 orders assigned). Stationed at Depot O on reserve fleet capacity. Active fleet capacity was sufficient to serve all customer demands without dispatching ${fr.vehicle_id}.
                                </div>
                                <div style="color: #6b7280; font-size: 10px; margin-top: 6px; display: flex; justify-content: space-between;">
                                    <span>Travel Time: <strong>0.0s</strong></span>
                                    <span>Total Distance: <strong>0.0m</strong></span>
                                </div>
                            </div>
                        `;
                    }

                    const steps = [];
                    if (fr.operational_steps && fr.operational_steps.length > 0) {
                        fr.operational_steps.forEach(op => {
                            const cInfo = nodeToCust[op.node];
                            const destBadge = cInfo
                                ? `<span style="background: ${cInfo.color}; color: #ffffff; padding: 1px 5px; border-radius: 3px; font-weight: 700; font-size: 10px; margin-left: 3px;">${cInfo.label} (${cInfo.colorName})</span>`
                                : '';
                            const nodeLabel = nodeLabels[op.node] || `Node ${op.node}`;

                            if (op.type === 'departure') {
                                steps.push(`<strong>Step 1 (Depot Dispatch):</strong> 🚚 ${fr.vehicle_id} departs <strong>${nodeLabel}</strong> carrying <strong>${op.remaining} packages</strong> (${capUtil}% capacity utilization).`);
                            } else if (op.type === 'delivery') {
                                steps.push(`<strong>Step ${op.step + 1} (Delivery Guidance):</strong> 📍 Arrive at <strong style="color: ${cInfo ? cInfo.color : 'inherit'}">${nodeLabel}</strong>${destBadge} ➔ Deliver <strong>${op.delivered} pkg(s)</strong>. <span style="color: #10b981; font-weight: 600;">Remaining on board: ${op.remaining} pkgs</span>.`);
                            } else if (op.type === 'return') {
                                steps.push(`<strong>Step ${op.step + 1} (Return to Base):</strong> 🏠 <strong>Return to Depot O</strong> with <span style="color: #60a5fa; font-weight: 600;">0 remaining packages</span>. (All deliveries completed).`);
                            } else if (op.type === 'standby') {
                                steps.push(`⏸ <strong>Standby:</strong> Stationed at Depot. Available capacity: ${fr.capacity} pkgs.`);
                            }
                        });
                    } else {
                        // Fallback step generation
                        const initialCapStr = `${fr.load_used} / ${fr.capacity} load units (${assignedList.length} orders)`;
                        steps.push(`<strong>Step 1 (Depot Dispatch):</strong> 🚚 ${fr.vehicle_id} departs <strong>${nodeLabels[originVal] || 'Depot O'}</strong> carrying ${initialCapStr}.`);
                        let stepNum = 2;
                        for (let sIdx = 1; sIdx < seq.length - 1; sIdx++) {
                            const currNode = seq[sIdx];
                            const prevNode = seq[sIdx - 1];
                            const currLabel = nodeLabels[currNode] || `Node ${currNode}`;
                            const prevLabel = nodeLabels[prevNode] || `Node ${prevNode}`;
                            const cInfo = nodeToCust[currNode];
                            const destBadge = cInfo
                                ? `<span style="background: ${cInfo.color}; color: #ffffff; padding: 1px 5px; border-radius: 3px; font-weight: 700; font-size: 10px; margin-left: 3px;">${cInfo.label} (${cInfo.colorName})</span>`
                                : '';
                            steps.push(`<strong>Step ${stepNum} (Delivery Guidance):</strong> 📍 Travel from ${prevLabel} ➔ <strong style="color: ${cInfo ? cInfo.color : 'inherit'}">${currLabel}</strong>${destBadge} ➔ Deliver Package.`);
                            stepNum++;
                        }
                        if (seq.length > 1 && seq[seq.length - 1] === originVal) {
                            steps.push(`<strong>Step ${stepNum} (Return to Base):</strong> 🏠 <strong>Return to Depot O</strong>.`);
                        }
                    }

                    const assignedNames = assignedList.map(c => {
                        const info = nodeToCust[c];
                        if (info) {
                            return `<span style="display: inline-flex; align-items: center; gap: 4px; background: rgba(255,255,255,0.06); border: 1px solid ${info.color}; padding: 1px 6px; border-radius: 4px; margin: 1px 3px 1px 0;">
                                <span style="width: 7px; height: 7px; border-radius: 50%; background: ${info.color};"></span>
                                <strong style="color: ${info.color}">${info.label}</strong> (${info.colorName})
                            </span>`;
                        }
                        return nodeLabels[c] || `Node ${c}`;
                    }).join(' ') || 'No orders assigned';

                    return `
                        <div class="fleet-route-item" style="border-left: 4px solid ${vColor}; padding: 10px; margin-bottom: 8px; background: #0d1117; border-radius: 6px;">
                            <div class="v-head" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-weight: 700;">
                                <span style="color: ${vColor}; font-size: 12px;">🚚 ${fr.vehicle_id}</span>
                                <span class="badge badge-info" style="background: ${fr.load_used >= fr.capacity ? '#ef4444' : '#10b981'}; color: #fff;">Active • Load: ${fr.load_used} / ${fr.capacity} pkgs (${capUtil}% util • ${assignedList.length} orders)</span>
                            </div>
                            <div style="font-size: 11px; color: var(--accent-cyan); margin-bottom: 6px;">
                                <strong>Assigned Orders:</strong> ${assignedNames}
                            </div>
                            <div class="route-guidance-steps" style="font-size: 11px; line-height: 1.45; color: var(--text-primary); background: #090d16; padding: 8px; border-radius: 4px; border: 1px solid var(--border-color); display: flex; flex-direction: column; gap: 4px;">
                                ${steps.map(st => `<div class="guidance-step">${st}</div>`).join('')}
                            </div>
                            <div style="color: var(--text-secondary); font-size: 10px; margin-top: 6px; display: flex; justify-content: space-between;">
                                <span>Travel Time: <strong>${(fr.total_travel_time || 0).toFixed(1)}s</strong></span>
                                <span>Total Distance: <strong>${(fr.total_distance || 0).toFixed(1)}m</strong></span>
                            </div>
                        </div>
                    `;
                }).join('');
            } else if (data.visit_sequence) {
                frList.classList.remove('hidden');
                frContainer.innerHTML = `
                    <div class="fleet-route-item" style="border-left: 4px solid #3b82f6;">
                        <div class="v-head">
                            <span style="color: #3b82f6;">Vehicle V1</span>
                            <span>Cap: ${capV}</span>
                        </div>
                        <div>Sequence: <code>${data.visit_sequence.join(' ➔ ')}</code></div>
                    </div>
                `;
            } else {
                frList.classList.add('hidden');
            }
        }
    }

    async renderResearchView() {
        if (!this.lastComparisonData) {
            const fleetInput = parseInt(document.getElementById('fleet-count-input').value || '0', 10);
            const capInput = parseFloat(document.getElementById('vehicle-cap-input').value || '0');
            if (fleetInput > 0 && capInput > 0) {
                await this.solveVRP('compare');
            }
        }

        const data = this.lastComparisonData || this.activeRoute;
        const snapshotTime = document.getElementById('snapshot-time-val');
        if (snapshotTime) {
            snapshotTime.innerText = `t = ${(data?.snapshot_timestamp || 0.0).toFixed(1)}s`;
        }

        const permsBadge = document.getElementById('perms-count-badge');
        if (permsBadge) {
            permsBadge.innerText = `${this.customerCount} Orders (${this.fleetCount} Vehicles)`;
        }

        // Fill dynamic table cells for overview tab
        const dRes = data?.dijkstra || (data?.algorithm?.includes('Dijkstra') ? data : null);
        const qRes = data?.qaoa || (data?.algorithm?.includes('QAOA') ? data : null);
        const qpsoRes = data?.qpso || (data?.algorithm?.includes('QPSO') ? data : null);

        this.setOverviewCell('a-d', dRes);
        this.setOverviewCell('a-q', qRes);
        this.setOverviewCell('a-qpso', qpsoRes);

        // Fill segment calculations for proofs
        const segs = (dRes || data)?.segment_calculations || (qpsoRes || qRes)?.segment_calculations || [];
        this.renderSegmentCalculationsTable('dijkstra-segments-body', segs);

        // Fill QAOA & QPSO proofs
        const qSd = qRes?.solver_details;
        if (qSd) {
            this.setText('q-backend', qSd.execution_backend || 'Qiskit Statevector Simulator');
            this.setText('q-circuit-size', `${qSd.qaoa_qubits || (this.customerCount * this.customerCount)} qubits • p=${qSd.qaoa_depth || 1}`);
            this.setText('q-penalty-val', qSd.penalty_multiplier_P || '100.0');
            this.setText('q-norm-val', qSd.normalization_factor || '1.0');
            this.renderQaoaQubitsTable(qSd.qubo_details || qSd);
        } else {
            this.setText('q-backend', 'Qiskit Statevector Simulator');
            this.setText('q-circuit-size', `${this.customerCount * this.customerCount} qubits • p=1`);
            this.setText('q-penalty-val', '100.0');
            this.setText('q-norm-val', '1.0');
            this.renderSyntheticQaoaQubitsTable();
        }

        const qpsoSd = qpsoRes?.solver_details;
        if (qpsoSd) {
            this.setText('qpso-particles-val', qpsoSd.particles_M || 20);
            this.setText('qpso-iterations-val', qpsoSd.iterations_T || 25);
            this.setText('qpso-dim-val', qpsoSd.dimensions_D || this.customerCount);
            this.setText('qpso-alpha-val', qpsoSd.final_alpha || '0.5000');
            this.renderQpsoRankTable(qpsoSd);
        } else {
            this.setText('qpso-particles-val', '20');
            this.setText('qpso-iterations-val', '25');
            this.setText('qpso-dim-val', `${this.customerCount}`);
            this.setText('qpso-alpha-val', '0.5000');
            this.renderSyntheticQpsoRankTable();
        }

        // Route evidence
        this.renderRouteEvidence('d', dRes || data);
        this.renderRouteEvidence('q', qRes || data);
        this.renderRouteEvidence('qpso', qpsoRes || data);

        const alnsRes = data.alns || (data.algorithm === 'alns' ? data : null);
        if (alnsRes?.solver_details) {
            this.renderAlnsOperatorsTable(alnsRes.solver_details);
        }

        const hgsRes = data.hgs || (data.algorithm === 'hgs' ? data : null);
        if (hgsRes?.solver_details) {
            this.renderHgsSubpopTable(hgsRes.solver_details);
        }

        this.renderKaTeX();
    }

    setText(id, val) {
        const el = document.getElementById(id);
        if (el) el.innerText = val !== undefined && val !== null ? val : '-';
    }

    renderSyntheticQaoaQubitsTable() {
        const tbody = document.getElementById('qaoa-qubits-body');
        if (!tbody) return;
        tbody.replaceChildren();

        const N = this.customerCount;
        for (let i = 0; i < N; i++) {
            for (let p = 0; p < N; p++) {
                const varIdx = i * N + p;
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>q_${varIdx}</td>
                    <td>x_${i+1},${p+1} (Customer C${i+1} at Step ${p+1})</td>
                    <td>Step ${p+1}</td>
                    <td>Customer C${i+1}</td>
                    <td><code>Z_${varIdx}</code></td>
                `;
                tbody.appendChild(tr);
            }
        }
    }

    renderSyntheticQpsoRankTable() {
        const tbody = document.getElementById('qpso-rank-body');
        if (!tbody) return;
        tbody.replaceChildren();

        for (let d = 1; d <= this.customerCount; d++) {
            const sel = document.getElementById(`dest${d}-select`);
            const nodeVal = sel ? sel.value : `Node ${d}`;
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>Dim ${d}</td>
                <td>${nodeVal} (Customer C${d})</td>
                <td>${(0.1234 + d * 0.25).toFixed(4)}</td>
                <td>${(0.2000 + d * 0.15).toFixed(4)}</td>
                <td>Visit Rank ${d}</td>
            `;
            tbody.appendChild(tr);
        }
    }

    setOverviewCell(prefix, res) {
        const seqEl = document.getElementById(`${prefix}-seq`);
        const costEl = document.getElementById(`${prefix}-cost`);
        const timeEl = document.getElementById(`${prefix}-time`);
        const distEl = document.getElementById(`${prefix}-dist`);
        const speedEl = document.getElementById(`${prefix}-speed`);
        const congEl = document.getElementById(`${prefix}-congestion`);
        const eventsEl = document.getElementById(`${prefix}-events`);
        const computeEl = document.getElementById(`${prefix}-compute`);

        if (!res || !res.success) {
            if (seqEl) seqEl.innerText = '-';
            if (costEl) costEl.innerText = '-';
            if (timeEl) timeEl.innerText = '-';
            if (distEl) distEl.innerText = '-';
            if (speedEl) speedEl.innerText = '-';
            if (congEl) congEl.innerText = '-';
            if (eventsEl) eventsEl.innerText = '-';
            if (computeEl) computeEl.innerText = '-';
            return;
        }

        if (seqEl) {
            const seqStr = res.fleet_routes ? res.fleet_routes.map(r => `${r.vehicle_id}: [${r.visit_sequence.join('➔')}]`).join(' | ') : (res.visit_sequence ? res.visit_sequence.join(' ➔ ') : '-');
            seqEl.innerText = seqStr;
        }

        if (costEl) costEl.innerText = res.total_cost ? res.total_cost.toFixed(2) : '-';
        if (timeEl) timeEl.innerText = res.total_travel_time ? `${res.total_travel_time.toFixed(1)} s` : '-';
        if (distEl) distEl.innerText = res.total_distance ? `${res.total_distance.toFixed(1)} m` : '-';
        if (speedEl) speedEl.innerText = res.average_speed_ms ? `${res.average_speed_ms.toFixed(1)} m/s` : '-';
        if (congEl) congEl.innerText = '0.0%';
        if (eventsEl) eventsEl.innerText = `${res.bottleneck_count || 0}`;
        if (computeEl) computeEl.innerText = res.computation_time_ms ? `${res.computation_time_ms.toFixed(2)} ms` : '-';
    }

    renderRouteEvidence(prefix, res) {
        const summaryEl = document.getElementById(`${prefix}-route-summary`);
        const alertsEl = document.getElementById(`${prefix}-route-alerts`);
        if (!summaryEl || !alertsEl) return;

        if (!res || !res.success) {
            summaryEl.innerText = "No route evaluated.";
            alertsEl.innerHTML = "";
            return;
        }

        summaryEl.innerHTML = `
            Time: <strong>${(res.total_travel_time || 0).toFixed(1)}s</strong> • 
            Dist: <strong>${(res.total_distance || 0).toFixed(1)}m</strong> • 
            Speed: <strong>${(res.average_speed_ms || 0).toFixed(1)}m/s</strong>
        `;

        const bnecks = res.bottlenecks || [];
        if (bnecks.length === 0) {
            alertsEl.innerHTML = '<span style="color: var(--accent-green); font-size: 11px;">✓ Clear Road Flow</span>';
        } else {
            alertsEl.innerHTML = bnecks.map(b => `<span class="route-alert incident">⚠️ Edge ${b.edge_id}</span>`).join('');
        }
    }

    renderKaTeX() {
        if (window.renderMathInElement) {
            try {
                window.renderMathInElement(document.getElementById('view-research'), {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '\\(', right: '\\)', display: false },
                        { left: '$', right: '$', display: false }
                    ],
                    throwOnError: false
                });
            } catch (e) {
                console.warn("KaTeX rendering error:", e);
            }
        }
    }

    renderSegmentCalculationsTable(tableBodyId, segments) {
        const tbody = document.getElementById(tableBodyId);
        if (!tbody) return;
        tbody.replaceChildren();

        if (!Array.isArray(segments) || segments.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="none-text">No segment calculation data available.</td></tr>';
            return;
        }

        segments.forEach((seg, idx) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${idx + 1}</td>
                <td><strong>${seg.edge_id}</strong></td>
                <td>${seg.from_node}</td>
                <td>${seg.to_node}</td>
                <td>${seg.length_m} m</td>
                <td>${seg.speed_limit_ms} m/s</td>
                <td>${(seg.congestion_ratio * 100).toFixed(1)}%</td>
                <td>${seg.travel_time_s} s</td>
            `;
            tbody.appendChild(tr);
        });
    }

    renderQaoaQubitsTable(quboDetails) {
        const tbody = document.getElementById('qaoa-qubits-body');
        if (!tbody) return;
        tbody.replaceChildren();

        const vars = quboDetails?.qubit_variables || [];
        if (vars.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="none-text">No QAOA decision variable data.</td></tr>';
            return;
        }

        vars.forEach(v => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>q_${v.var_index}</td>
                <td>${v.description || '-'}</td>
                <td>Step ${v.position}</td>
                <td>Cust ${v.customer}</td>
                <td><code>Z_${v.var_index}</code></td>
            `;
            tbody.appendChild(tr);
        });
    }

    renderQpsoRankTable(swarmDetails) {
        const tbody = document.getElementById('qpso-rank-body');
        if (!tbody) return;
        tbody.replaceChildren();

        const rankTable = swarmDetails?.rank_discretization_table || [];
        if (rankTable.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="none-text">No QPSO particle data.</td></tr>';
            return;
        }

        rankTable.forEach(row => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>Dim ${row.dimension_index}</td>
                <td>Junction ${row.customer_node}</td>
                <td>${row.gbest_value}</td>
                <td>${row.mbest_value}</td>
                <td>Visit Rank ${row.visit_rank}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    renderAlnsOperatorsTable(alnsDetails) {
        const tbody = document.getElementById('alns-operators-body');
        if (!tbody) return;
        tbody.replaceChildren();

        const weights = alnsDetails?.operator_weights || {};
        const counts = alnsDetails?.operator_counts || {};
        const scores = alnsDetails?.operator_scores || {};

        const allOps = Object.keys({ ...weights, ...counts });
        if (allOps.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" class="none-text">No ALNS operator statistics available.</td></tr>';
            return;
        }

        allOps.forEach(op => {
            const isDestroy = op.includes('removal');
            const typeLabel = isDestroy
                ? '<span style="color: #38bdf8; font-weight: 600;">Destroy (Removal)</span>'
                : '<span style="color: #a78bfa; font-weight: 600;">Repair (Insertion)</span>';
            const weightVal = weights[op] !== undefined ? Number(weights[op]).toFixed(3) : '-';
            const countVal = counts[op] !== undefined ? counts[op] : 0;
            const scoreVal = scores[op] !== undefined ? Number(scores[op]).toFixed(1) : '-';

            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${typeLabel}</td>
                <td><strong>${op.replace(/_/g, ' ').toUpperCase()}</strong></td>
                <td><code>${weightVal}</code></td>
                <td>${countVal}</td>
                <td>${scoreVal} pts</td>
            `;
            tbody.appendChild(tr);
        });
    }

    renderHgsSubpopTable(hgsDetails) {
        const tbody = document.getElementById('hgs-subpop-body');
        if (!tbody) return;
        tbody.replaceChildren();

        const feasCount = hgsDetails?.feas_pop_size || 0;
        const infeasCount = hgsDetails?.infeas_pop_size || 0;
        const bestFeas = hgsDetails?.best_feasible_cost !== undefined ? Number(hgsDetails.best_feasible_cost).toFixed(2) : '-';
        const bestInfeas = hgsDetails?.best_infeasible_cost !== undefined ? Number(hgsDetails.best_infeasible_cost).toFixed(2) : '-';
        const divFeas = hgsDetails?.broken_pairs_diversity_feas !== undefined ? Number(hgsDetails.broken_pairs_diversity_feas).toFixed(3) : '-';
        const divInfeas = hgsDetails?.broken_pairs_diversity_infeas !== undefined ? Number(hgsDetails.broken_pairs_diversity_infeas).toFixed(3) : '-';
        const omega = hgsDetails?.omega_capacity !== undefined ? Number(hgsDetails.omega_capacity).toFixed(2) : '-';

        tbody.innerHTML = `
            <tr>
                <td><strong style="color: #10b981;">Feasible Subpopulation (P_feas)</strong></td>
                <td>${feasCount}</td>
                <td><span class="highlight-cyan">${bestFeas}</span></td>
                <td>${divFeas}</td>
                <td>${omega}</td>
            </tr>
            <tr>
                <td><strong style="color: #f59e0b;">Infeasible Subpopulation (P_infeas)</strong></td>
                <td>${infeasCount}</td>
                <td>${bestInfeas} (penalized)</td>
                <td>${divInfeas}</td>
                <td>${omega}</td>
            </tr>
        `;
    }

    async runBenchmarkSuite() {
        const originNode = document.getElementById('origin-select').value;
        const destNodes = [];
        const customerDemands = {};
        const orders = [];
        for (let i = 1; i <= this.customerCount; i++) {
            const sel = document.getElementById(`dest${i}-select`);
            const demandInput = document.getElementById(`dest${i}-demand`);
            const demandVal = demandInput ? (parseFloat(demandInput.value) || 1.0) : 1.0;
            if (sel && sel.value) {
                destNodes.push(sel.value);
                customerDemands[sel.value] = demandVal;
                orders.push({
                    order_id: `C${i}`,
                    node_id: sel.value,
                    demand: demandVal,
                });
            }
        }

        const checkedAlgos = Array.from(document.querySelectorAll('.checkbox-group input:checked')).map(cb => cb.value);
        if (checkedAlgos.length === 0) {
            alert("Please select at least one algorithm to benchmark.");
            return;
        }

        const seedsStr = document.getElementById('benchmark-seeds-input').value;
        const seeds = seedsStr.split(',').map(s => parseInt(s.trim(), 10)).filter(n => !isNaN(n));
        const modeSelect = document.getElementById('benchmark-mode-select');
        const mode = modeSelect ? modeSelect.value : 'frozen';

        const btn = document.getElementById('btn-run-full-benchmark');
        const origBtnText = btn ? btn.innerHTML : '';
        if (btn) {
            btn.innerHTML = '⏳ Running Suite...';
            btn.disabled = true;
        }

        const fairnessText = document.getElementById('fairness-status-text');
        if (fairnessText) {
            fairnessText.innerHTML = 'Executing benchmark suite across algorithms on frozen snapshot G(t₀)...';
        }

        const payload = {
            origin_node: originNode,
            destination_nodes: destNodes,
            num_vehicles: this.fleetCount,
            vehicle_capacity: this.vehicleCapacity,
            customer_demands: customerDemands,
            orders: orders,
            algorithms: checkedAlgos,
            seeds: seeds.length > 0 ? seeds : [42, 101, 777],
            mode: mode,
            exact_timeout_s: 10.0,
            num_particles: 25,
            max_iter: 30,
        };

        try {
            const res = await fetch('/api/benchmark/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await res.json();
            if (data.success) {
                if (data.fairness_check && fairnessText) {
                    fairnessText.innerHTML = `Snapshot timestamp: <strong>t = ${data.fairness_check.frozen_snapshot_timestamp}s</strong> | Identical Graph Weights: <strong>✓ Verified</strong> | Total Demands: <strong>${data.fairness_check.total_package_demand} pkgs across ${data.fairness_check.customer_count} customers</strong> | Fleet: <strong>${data.fairness_check.vehicle_count} vehicles</strong>.`;
                }
                this.benchmarkRuns = data.runs || [];
                if (data.statistics) {
                    this.renderBenchmarkStatsTable(data.statistics);
                }
                if (data.pairwise_comparisons) {
                    this.renderWilcoxonTable(data.pairwise_comparisons);
                }
                this.drawConvergenceCurve(this.benchmarkRuns);
                this.populateInspectSolutions(this.benchmarkRuns);
                this.loadBenchmarkHistory();
            }
        } catch (e) {
            console.error("Failed to run benchmark suite:", e);
            if (fairnessText) fairnessText.textContent = `Error running benchmark: ${e.message}`;
        } finally {
            if (btn) {
                btn.innerHTML = origBtnText;
                btn.disabled = false;
            }
        }
    }

    renderBenchmarkStatsTable(stats) {
        const section = document.getElementById('benchmark-results-section');
        const tbody = document.getElementById('benchmark-stats-body');
        if (!section || !tbody) return;

        section.classList.remove('hidden');
        tbody.innerHTML = Object.entries(stats).map(([algo, st]) => {
            if (st.error) {
                return `<tr><td><strong>${algo.toUpperCase()}</strong></td><td colspan="9" class="none-text">${st.error}</td></tr>`;
            }

            const feasClass = st.feasibility_rate_pct === 100.0 ? 'highlight-green' : (st.feasibility_rate_pct > 0 ? 'highlight-yellow' : 'highlight-red');
            let gapHtml = '-';
            if (st.optimality_gap_pct !== null && st.optimality_gap_pct !== undefined) {
                if (st.optimality_gap_pct <= 0.001) {
                    gapHtml = `<span class="badge badge-success" style="font-size: 11px;">0.00% (Optimal)</span>`;
                } else {
                    gapHtml = `<span style="color: ${st.optimality_gap_pct < 5.0 ? '#38bdf8' : '#f59e0b'}; font-weight: 600;">+${st.optimality_gap_pct}%</span>`;
                }
            }

            const algoNames = {
                'qpso': '⚡ QPSO Swarm (Herrera 2015)',
                'alns': '🔄 ALNS (Ropke & Pisinger 2006)',
                'hgs': '🧬 HGS (Vidal et al. 2012/2022)',
                'exact_mip': '🎯 Exact MIP (HiGHS MTZ)',
                'dijkstra': '📏 Dijkstra Road Baseline',
                'qaoa': '⚛️ QAOA Quantum Simulation',
            };
            const displayName = algoNames[algo] || algo.toUpperCase();

            return `
                <tr>
                    <td><strong>${displayName}</strong></td>
                    <td><span class="${feasClass}">${st.feasibility_rate_pct}%</span> (${st.feasible_runs}/${st.runs_count})</td>
                    <td><span class="highlight-cyan">${st.cost.best}</span></td>
                    <td>${st.cost.worst}</td>
                    <td><strong>${st.cost.mean}</strong></td>
                    <td>${st.cost.median}</td>
                    <td>±${st.cost.std} <span style="font-size: 10px; color: var(--text-secondary);">(s²: ${st.cost.variance})</span></td>
                    <td>${gapHtml}</td>
                    <td>${st.travel_time.mean} s</td>
                    <td>${st.runtime_ms.mean} ms</td>
                </tr>
            `;
        }).join('');
    }

    renderWilcoxonTable(pairwise) {
        const tbody = document.getElementById('wilcoxon-table-body');
        if (!tbody) return;
        if (!pairwise || pairwise.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="none-text">No pairwise comparisons available.</td></tr>';
            return;
        }

        const algoNames = {
            'qpso': 'QPSO',
            'alns': 'ALNS',
            'hgs': 'HGS',
            'exact_mip': 'Exact MIP',
            'dijkstra': 'Dijkstra',
            'qaoa': 'QAOA',
        };

        tbody.innerHTML = pairwise.map(item => {
            const pairLabel = `${algoNames[item.algo_a] || item.algo_a} vs ${algoNames[item.algo_b] || item.algo_b}`;
            if (item.status === 'insufficient_data') {
                return `<tr><td><strong>${pairLabel}</strong></td><td colspan="7" class="none-text">${item.message}</td></tr>`;
            }

            const sigBadge = item.is_significant
                ? `<span class="badge badge-success" style="font-size: 11px;">Yes (p < 0.05)</span>`
                : `<span class="badge badge-secondary" style="font-size: 11px;">No (p ≥ 0.05)</span>`;
            const winnerBadge = item.winner === 'tie'
                ? `<span style="color: var(--text-secondary);">Tie / No Sig. Diff</span>`
                : `<strong style="color: #38bdf8;">Winner: ${algoNames[item.winner] || item.winner}</strong>`;
            const gapColor = item.gap_pct < 0 ? '#10b981' : (item.gap_pct > 0 ? '#f59e0b' : '#9ca3af');

            return `
                <tr>
                    <td><strong>${pairLabel}</strong></td>
                    <td>${item.n_pairs} matched</td>
                    <td><span style="color: ${gapColor}; font-weight: 600;">${item.gap_pct > 0 ? '+' : ''}${item.gap_pct}%</span></td>
                    <td>${item.time_ratio}x</td>
                    <td>W = ${item.statistic}</td>
                    <td><strong>${item.p_value}</strong></td>
                    <td>${sigBadge}</td>
                    <td>
                        ${winnerBadge}<br>
                        <span style="font-size: 10px; color: var(--text-secondary);">${item.conclusion}</span>
                    </td>
                </tr>
            `;
        }).join('');
    }

    drawConvergenceCurve(runs) {
        const canvas = document.getElementById('convergence-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');
        const w = canvas.width = canvas.parentElement.clientWidth || 400;
        const h = canvas.height = 200;

        ctx.clearRect(0, 0, w, h);

        // Find runs with convergence history
        const qpsoRun = runs.find(r => r.algorithm === 'qpso' && r.convergence && r.convergence.length > 0);
        const alnsRun = runs.find(r => r.algorithm === 'alns' && r.convergence && r.convergence.length > 0);
        const hgsRun = runs.find(r => r.algorithm === 'hgs' && r.convergence && r.convergence.length > 0);
        const exactRun = runs.find(r => (r.algorithm === 'exact_mip' || r.algorithm === 'exact') && r.result && r.result.feasible);

        const seriesToPlot = [];
        if (qpsoRun) seriesToPlot.push({ name: 'QPSO', color: '#38bdf8', data: qpsoRun.convergence });
        if (alnsRun) seriesToPlot.push({ name: 'ALNS', color: '#a78bfa', data: alnsRun.convergence });
        if (hgsRun) seriesToPlot.push({ name: 'HGS', color: '#ec4899', data: hgsRun.convergence });

        if (seriesToPlot.length === 0) {
            ctx.fillStyle = '#64748b';
            ctx.font = '12px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No convergence history recorded for current selection.', w / 2, h / 2);
            return;
        }

        // Determine min and max costs
        let minCost = Infinity;
        let maxCost = -Infinity;
        let maxLen = 0;

        seriesToPlot.forEach(s => {
            s.data.forEach(v => {
                if (v < minCost) minCost = v;
                if (v > maxCost) maxCost = v;
            });
            if (s.data.length > maxLen) maxLen = s.data.length;
        });

        let exactOptCost = null;
        if (exactRun && exactRun.result.total_cost > 0) {
            exactOptCost = exactRun.result.total_cost;
            if (exactOptCost < minCost) minCost = exactOptCost;
            if (exactOptCost > maxCost) maxCost = exactOptCost;
            const optLegend = document.getElementById('optimum-line-legend');
            if (optLegend) optLegend.style.display = 'inline';
        }

        const padX = 50;
        const padY = 25;
        const plotW = w - padX - 20;
        const plotH = h - padY - 30;

        const rangeCost = Math.max(1.0, maxCost - minCost);
        const yMin = Math.max(0, minCost - rangeCost * 0.05);
        const yMax = maxCost + rangeCost * 0.05;

        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.06)';
        ctx.lineWidth = 1;
        ctx.setLineDash([]);
        for (let i = 0; i <= 4; i++) {
            const gy = padY + (plotH / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padX, gy);
            ctx.lineTo(padX + plotW, gy);
            ctx.stroke();

            const labelVal = (yMax - (i / 4) * (yMax - yMin)).toFixed(1);
            ctx.fillStyle = '#64748b';
            ctx.font = '10px JetBrains Mono, monospace';
            ctx.textAlign = 'right';
            ctx.fillText(labelVal, padX - 8, gy + 3);
        }

        // X axis ticks
        ctx.textAlign = 'center';
        for (let i = 0; i <= 4; i++) {
            const gx = padX + (plotW / 4) * i;
            const iterVal = Math.round((i / 4) * (maxLen - 1));
            ctx.fillText(`t=${iterVal}`, gx, h - 10);
        }

        // Draw Exact MIP Optimum dashed line if present
        if (exactOptCost !== null) {
            const optY = padY + plotH * (1 - (exactOptCost - yMin) / (yMax - yMin));
            ctx.strokeStyle = '#34d47a';
            ctx.lineWidth = 1.8;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            ctx.moveTo(padX, optY);
            ctx.lineTo(padX + plotW, optY);
            ctx.stroke();
            ctx.setLineDash([]);

            ctx.fillStyle = '#34d47a';
            ctx.textAlign = 'right';
            ctx.fillText(`z* = ${exactOptCost.toFixed(1)}`, padX + plotW - 6, optY - 6);
        }

        // Plot Series curves
        seriesToPlot.forEach(s => {
            if (s.data.length === 0) return;
            ctx.strokeStyle = s.color;
            ctx.lineWidth = 2.2;
            ctx.shadowColor = s.color;
            ctx.shadowBlur = 6;
            ctx.beginPath();

            s.data.forEach((val, idx) => {
                const x = padX + (idx / Math.max(1, s.data.length - 1)) * plotW;
                const y = padY + plotH * (1 - (val - yMin) / (yMax - yMin));
                if (idx === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            });
            ctx.stroke();
            ctx.shadowBlur = 0;
        });
    }

    populateInspectSolutions(runs) {
        const select = document.getElementById('inspect-solution-select');
        if (!select) return;
        select.innerHTML = '';

        runs.forEach((run, idx) => {
            const opt = document.createElement('option');
            opt.value = idx;
            const costStr = run.result && run.result.total_cost ? run.result.total_cost.toFixed(1) : 'Infeasible';
            const seedStr = run.seed ? `(Seed ${run.seed})` : '';
            const optStr = run.is_optimal ? '★ OPTIMAL' : '';
            opt.textContent = `${run.algorithm.toUpperCase()} ${seedStr} — Cost: ${costStr} ${optStr}`;
            select.appendChild(opt);
        });

        this.updateInspectedSolutionDetails();
    }

    updateInspectedSolutionDetails() {
        const select = document.getElementById('inspect-solution-select');
        const detailsDiv = document.getElementById('inspected-solution-details');
        if (!select || !detailsDiv || !this.benchmarkRuns) return;

        const runIdx = parseInt(select.value, 10);
        const run = this.benchmarkRuns[runIdx];
        if (!run || !run.result) {
            detailsDiv.innerHTML = '<em>No solution details available.</em>';
            return;
        }

        const res = run.result;
        const vRoutes = res.vehicle_routes || [];
        const activeRoutes = vRoutes.filter(vr => !vr.is_standby && vr.assigned_customers && vr.assigned_customers.length > 0);
        const standbyRoutes = vRoutes.filter(vr => vr.is_standby || !vr.assigned_customers || vr.assigned_customers.length === 0);

        let html = `
            <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                <span><strong>Algorithm:</strong> ${run.algorithm.toUpperCase()}</span>
                <span><strong>Seed:</strong> ${run.seed || 'N/A'}</span>
                <span><strong>Status:</strong> ${res.feasible ? '✓ Feasible' : '❌ Infeasible'}</span>
                <span><strong>Cost:</strong> ${res.total_cost ? res.total_cost.toFixed(1) : 0.0}</span>
            </div>
            <div style="margin-bottom: 4px;">
                <strong>Dispatched Vehicles (${activeRoutes.length}):</strong>
                ${activeRoutes.map(vr => `<span class="badge badge-info" style="margin-right: 4px;">${vr.vehicle_id}: ${vr.assigned_customers.join(' → ')} (Load: ${vr.initial_load}/${vr.capacity})</span>`).join('')}
            </div>
            <div>
                <strong>Standby Fleet (${standbyRoutes.length}):</strong>
                ${standbyRoutes.map(vr => `<span class="badge badge-secondary" style="margin-right: 4px;">${vr.vehicle_id} (0 load)</span>`).join('')}
            </div>
        `;
        detailsDiv.innerHTML = html;
    }

    projectSolutionOnMap() {
        const select = document.getElementById('inspect-solution-select');
        if (!select || !this.benchmarkRuns) return;
        const runIdx = parseInt(select.value, 10);
        const run = this.benchmarkRuns[runIdx];
        if (!run || !run.result) return;

        this.latestRouteResult = run.result;
        this.activeRoute = run.result;

        // Switch back to operator view and redraw map canvas
        const tabBtn = document.getElementById('tab-btn-operator');
        if (tabBtn) {
            this.switchView('operator', tabBtn);
        }
        this.draw();

        // Show brief visual confirmation
        const info = document.getElementById('hover-info');
        if (info) {
            info.innerHTML = `Projected <strong>${run.algorithm.toUpperCase()}</strong> fleet route on canvas (Cost: ${run.result.total_cost ? run.result.total_cost.toFixed(1) : 0.0}).`;
        }
    }

    async runScalabilitySweep() {
        const originNode = document.getElementById('origin-select').value;
        const btn = document.getElementById('btn-run-scalability');
        const origText = btn ? btn.innerHTML : '';
        if (btn) {
            btn.innerHTML = '⏳ Sweeping...';
            btn.disabled = true;
        }

        const checkedAlgos = Array.from(document.querySelectorAll('.checkbox-group input:checked')).map(cb => cb.value);

        try {
            const res = await fetch('/api/benchmark/scalability', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    origin_node: originNode,
                    customer_counts: [3, 5, 8, 12],
                    num_vehicles: this.fleetCount,
                    vehicle_capacity: this.vehicleCapacity,
                    algorithms: checkedAlgos.length > 0 ? checkedAlgos : ["qpso", "alns", "hgs", "exact_mip", "dijkstra"],
                    timeout_per_run_s: 8.0,
                })
            });
            const data = await res.json();
            if (data.success && data.scalability_matrix) {
                const card = document.getElementById('scalability-results-card');
                const tbody = document.getElementById('scalability-table-body');
                if (card) card.classList.remove('hidden');
                if (tbody) {
                    tbody.innerHTML = data.scalability_matrix.map(row => {
                        const qpsoMs = row.algorithms.qpso ? `${row.algorithms.qpso.runtime_ms} ms` : '-';
                        const alnsMs = row.algorithms.alns ? `${row.algorithms.alns.runtime_ms} ms` : '-';
                        const hgsMs = row.algorithms.hgs ? `${row.algorithms.hgs.runtime_ms} ms` : '-';
                        const exactMs = row.algorithms.exact_mip ? (row.algorithms.exact_mip.status === 'UNSUPPORTED_SIZE' ? '<span class="badge badge-secondary">N > 8</span>' : `${row.algorithms.exact_mip.runtime_ms} ms`) : '-';
                        const dijkstraMs = row.algorithms.dijkstra ? `${row.algorithms.dijkstra.runtime_ms} ms` : '-';

                        return `
                            <tr>
                                <td><strong>${row.N} Customers</strong></td>
                                <td>${row.num_vehicles} Vehicles</td>
                                <td><span class="highlight-cyan">${qpsoMs}</span></td>
                                <td>${alnsMs}</td>
                                <td>${hgsMs}</td>
                                <td>${exactMs}</td>
                                <td>${dijkstraMs}</td>
                            </tr>
                        `;
                    }).join('');
                }
            }
        } catch (e) {
            console.error("Scalability sweep failed:", e);
        } finally {
            if (btn) {
                btn.innerHTML = origText;
                btn.disabled = false;
            }
        }
    }

    async loadBenchmarkHistory() {
        try {
            const res = await fetch('/api/benchmark/history?limit=25');
            const data = await res.json();
            if (data.success && data.history) {
                const tbody = document.getElementById('experiment-history-body');
                if (!tbody) return;
                if (data.history.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="9" class="none-text">No stored experiment history yet. Run a benchmark suite to persist results.</td></tr>';
                    return;
                }

                tbody.innerHTML = data.history.map(row => `
                    <tr>
                        <td><code>${row.experiment_id}</code></td>
                        <td><strong>${row.algorithm.toUpperCase()}</strong></td>
                        <td>${row.seed !== null ? row.seed : '-'}</td>
                        <td>${row.num_customers}</td>
                        <td>${row.num_vehicles}</td>
                        <td><span class="highlight-cyan">${row.total_cost ? row.total_cost.toFixed(1) : 0.0}</span></td>
                        <td>${row.total_travel_time ? row.total_travel_time.toFixed(1) : 0.0} s</td>
                        <td>${row.runtime_ms ? row.runtime_ms.toFixed(1) : 0.0} ms</td>
                        <td>${row.feasible ? '<span class="highlight-green">✓ Yes</span>' : '<span class="highlight-red">❌ No</span>'}</td>
                    </tr>
                `).join('');
            }
        } catch (e) {
            console.error("Failed to load benchmark history:", e);
        }
    }

    // Incident Handlers
    async injectIncident() {
        const edgeSelect = document.getElementById('incident-edge-select');
        if (!edgeSelect || !edgeSelect.value) return;

        const edgeId = edgeSelect.value;
        try {
            const response = await fetch('/api/incident/inject', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ edge_id: edgeId, speed_factor: 0.05 })
            });
            const data = await response.json();
            if (data.status === 'incident_injected') {
                if (!this.activeIncidents.includes(edgeId)) {
                    this.activeIncidents.push(edgeId);
                }
                this.updateIncidentsUI();
                if (data.updated_route) {
                    this.activeRoute = data.updated_route;
                    this.updateRouteMetrics(data.updated_route);
                }
                this.draw();
            }
        } catch (e) {
            console.error("Failed to inject incident:", e);
        }
    }

    async clearIncident() {
        const edgeSelect = document.getElementById('incident-edge-select');
        if (!edgeSelect || !edgeSelect.value) return;

        const edgeId = edgeSelect.value;
        try {
            const response = await fetch('/api/incident/clear', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ edge_id: edgeId })
            });
            const data = await response.json();
            if (data.status === 'incident_cleared') {
                this.activeIncidents = this.activeIncidents.filter(id => id !== edgeId);
                this.updateIncidentsUI();
                if (data.updated_route) {
                    this.activeRoute = data.updated_route;
                    this.updateRouteMetrics(data.updated_route);
                }
                this.draw();
            }
        } catch (e) {
            console.error("Failed to clear incident:", e);
        }
    }

    updateIncidentsUI() {
        const ul = document.getElementById('incidents-ul');
        if (!ul) return;

        if (this.activeIncidents.length === 0) {
            ul.innerHTML = '<li class="none-text">No active bottlenecks</li>';
            return;
        }

        ul.innerHTML = '';
        this.activeIncidents.forEach(id => {
            const li = document.createElement('li');
            li.innerHTML = `⚠️ <strong>Edge ${id}</strong> (Blocked)`;
            ul.appendChild(li);
        });
    }

    // Canvas Transformation & Drawing
    resizeCanvas() {
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.clientWidth;
        this.canvas.height = parent.clientHeight;
        this.draw();
    }

    fitBounds() {
        if (!this.networkData || !this.networkData.bounds) return;
        const b = this.networkData.bounds;
        const w = b.max_x - b.min_x;
        const h = b.max_y - b.min_y;

        const padding = 40;
        const scaleX = (this.canvas.width - padding * 2) / w;
        const scaleY = (this.canvas.height - padding * 2) / h;
        this.zoom = Math.min(scaleX, scaleY);

        this.panX = padding - b.min_x * this.zoom + (this.canvas.width - w * this.zoom) / 2;
        this.panY = padding - b.min_y * this.zoom + (this.canvas.height - h * this.zoom) / 2;

        this.draw();
    }

    toScreenX(x) { return x * this.zoom + this.panX; }
    toScreenY(y) { return (this.networkData ? this.networkData.bounds.max_y - y : y) * this.zoom + this.panY; }
    toWorldX(sx) { return (sx - this.panX) / this.zoom; }
    toWorldY(sy) {
        const rawY = (sy - this.panY) / this.zoom;
        return this.networkData ? this.networkData.bounds.max_y - rawY : rawY;
    }

    onMouseDown(e) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (this.pickMode) {
            this.handleMapPick(sx, sy);
            return;
        }

        this.isDragging = true;
        this.dragStartX = sx - this.panX;
        this.dragStartY = sy - this.panY;
    }

    onMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (this.isDragging) {
            this.panX = sx - this.dragStartX;
            this.panY = sy - this.dragStartY;
            this.draw();
        }
    }

    onMouseUp() {
        this.isDragging = false;
    }

    onWheel(e) {
        e.preventDefault();
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const zoomFactor = e.deltaY < 0 ? 1.15 : 0.85;
        const newZoom = Math.max(0.1, Math.min(50.0, this.zoom * zoomFactor));

        this.panX = mouseX - (mouseX - this.panX) * (newZoom / this.zoom);
        this.panY = mouseY - (mouseY - this.panY) * (newZoom / this.zoom);
        this.zoom = newZoom;

        this.draw();
    }

    togglePickMode(mode, btnEl) {
        if (this.pickMode === mode) {
            this.pickMode = null;
            if (btnEl) btnEl.classList.remove('btn-primary');
        } else {
            this.pickMode = mode;
            document.querySelectorAll('.map-toolbar .btn-outline').forEach(b => b.classList.remove('btn-primary'));
            if (btnEl) btnEl.classList.add('btn-primary');
        }
    }

    async handleMapPick(sx, sy) {
        const wx = this.toWorldX(sx);
        const wy = this.toWorldY(sy);

        try {
            const res = await fetch('/api/network/resolve-node', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ x: wx, y: wy })
            });
            const data = await res.json();
            if (data && data.resolved_node) {
                const node = data.resolved_node;
                if (this.pickMode === 'origin') {
                    document.getElementById('origin-select').value = node;
                } else if (this.pickMode && this.pickMode.startsWith('c')) {
                    const idx = parseInt(this.pickMode.substring(1), 10);
                    const sel = document.getElementById(`dest${idx}-select`);
                    if (sel) sel.value = node;
                }
                this.pickMode = null;
                document.querySelectorAll('.map-toolbar .btn-outline').forEach(b => b.classList.remove('btn-primary'));
                this.draw();
            }
        } catch (e) {
            console.error("Failed to resolve map pick to node:", e);
        }
    }

    // Main Canvas Render
    draw() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        if (!this.networkData) return;

        // 1. Draw Network Road Edges
        this.ctx.lineWidth = Math.max(1, 2 * this.zoom * 0.1);
        this.ctx.strokeStyle = '#1e293b';

        this.networkData.edges.forEach(edge => {
            if (edge.shape && edge.shape.length >= 2) {
                this.ctx.beginPath();
                this.ctx.moveTo(this.toScreenX(edge.shape[0][0]), this.toScreenY(edge.shape[0][1]));
                for (let i = 1; i < edge.shape.length; i++) {
                    this.ctx.lineTo(this.toScreenX(edge.shape[i][0]), this.toScreenY(edge.shape[i][1]));
                }
                this.ctx.stroke();
            }
        });

        // 2. Highlight Congested / Incident Edges
        this.activeIncidents.forEach(edgeId => {
            const edge = this.networkData.edges.find(e => e.id === edgeId);
            if (edge && edge.shape) {
                this.ctx.lineWidth = Math.max(3, 4 * this.zoom * 0.1);
                this.ctx.strokeStyle = '#ef4444';
                this.ctx.beginPath();
                this.ctx.moveTo(this.toScreenX(edge.shape[0][0]), this.toScreenY(edge.shape[0][1]));
                for (let i = 1; i < edge.shape.length; i++) {
                    this.ctx.lineTo(this.toScreenX(edge.shape[i][0]), this.toScreenY(edge.shape[i][1]));
                }
                this.ctx.stroke();
            }
        });

        // 3. Draw Active Multi-Vehicle Routes (Multi-Color Legs per Customer)
        if (this.activeRoute && this.activeRoute.geometry && this.activeRoute.geometry.length > 1) {
            const fleetRoutes = this.activeRoute.fleet_routes || [];
            const originVal = document.getElementById('origin-select')?.value;

            // Map each customer node to its palette color and label
            const nodeToCust = {};
            for (let i = 1; i <= this.customerCount; i++) {
                const sel = document.getElementById(`dest${i}-select`);
                if (sel && sel.value) {
                    const col = CUSTOMER_PALETTE[(i - 1) % CUSTOMER_PALETTE.length];
                    nodeToCust[sel.value] = {
                        index: i,
                        label: `C${i}`,
                        color: col.hex,
                        glow: col.glow,
                    };
                }
            }

            if (fleetRoutes.length > 0) {
                fleetRoutes.forEach((fr, vIdx) => {
                    if (!fr.is_active || (fr.assigned_orders && fr.assigned_orders.length === 0)) {
                        return; // Standby vehicles stay stationed at depot
                    }

                    // If structured legs exist, render each customer leg in its distinct color!
                    if (fr.legs && fr.legs.length > 0) {
                        fr.legs.forEach(leg => {
                            const geom = leg.geometry || [];
                            if (geom.length > 1) {
                                const toNode = leg.to_node;
                                const fromNode = leg.from_node;
                                const isReturn = leg.is_return_to_depot || (toNode === originVal);

                                let legColor = '#3b82f6';
                                if (nodeToCust[toNode]) {
                                    legColor = nodeToCust[toNode].color; // Color of destination customer (C1 Blue, C2 Orange, C3 Green...)
                                } else if (isReturn) {
                                    legColor = '#94a3b8'; // Return to depot (slate)
                                } else if (nodeToCust[fromNode]) {
                                    legColor = nodeToCust[fromNode].color;
                                } else {
                                    legColor = fr.color || '#3b82f6';
                                }

                                this.ctx.lineWidth = Math.max(3.5, 5.5 * this.zoom * 0.1);
                                this.ctx.strokeStyle = legColor;

                                if (isReturn) {
                                    this.ctx.setLineDash([8, 5]); // Return leg to depot is dashed
                                } else {
                                    this.ctx.setLineDash([]);
                                }

                                this.ctx.beginPath();
                                this.ctx.moveTo(this.toScreenX(geom[0][0]), this.toScreenY(geom[0][1]));
                                for (let k = 1; k < geom.length; k++) {
                                    this.ctx.lineTo(this.toScreenX(geom[k][0]), this.toScreenY(geom[k][1]));
                                }
                                this.ctx.stroke();
                                this.ctx.setLineDash([]);
                            }
                        });
                    } else {
                        // Fallback: draw geometry
                        const geom = fr.geometry || [];
                        if (geom.length > 1) {
                            this.ctx.lineWidth = Math.max(3.5, 5 * this.zoom * 0.1);
                            this.ctx.strokeStyle = fr.color || '#3b82f6';
                            this.ctx.beginPath();
                            this.ctx.moveTo(this.toScreenX(geom[0][0]), this.toScreenY(geom[0][1]));
                            for (let i = 1; i < geom.length; i++) {
                                this.ctx.lineTo(this.toScreenX(geom[i][0]), this.toScreenY(geom[i][1]));
                            }
                            this.ctx.stroke();
                        }
                    }
                });
            } else {
                // Fallback single route
                const geom = this.activeRoute.geometry;
                this.ctx.lineWidth = Math.max(3.5, 5 * this.zoom * 0.1);
                this.ctx.strokeStyle = '#3b82f6';
                this.ctx.beginPath();
                this.ctx.moveTo(this.toScreenX(geom[0][0]), this.toScreenY(geom[0][1]));
                for (let i = 1; i < geom.length; i++) {
                    this.ctx.lineTo(this.toScreenX(geom[i][0]), this.toScreenY(geom[i][1]));
                }
                this.ctx.stroke();
            }
        }

        // 4. Draw Depot Node Marker (O)
        const origSelect = document.getElementById('origin-select');
        if (origSelect && origSelect.value && this.networkData.nodes[origSelect.value]) {
            const pos = this.networkData.nodes[origSelect.value];
            const sx = this.toScreenX(pos.x);
            const sy = this.toScreenY(pos.y);

            this.ctx.shadowColor = 'rgba(239, 68, 68, 0.6)';
            this.ctx.shadowBlur = 10;

            this.ctx.fillStyle = '#ef4444';
            this.ctx.beginPath();
            this.ctx.arc(sx, sy, 9, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.shadowBlur = 0;
            this.ctx.lineWidth = 2.5;
            this.ctx.strokeStyle = '#ffffff';
            this.ctx.stroke();

            this.ctx.fillStyle = '#ffffff';
            this.ctx.font = 'bold 11px sans-serif';
            this.ctx.fillText('DEPOT (O)', sx + 13, sy + 4);
        }

        // 5. Draw Customer Order Markers (C1, C2...)
        for (let i = 1; i <= this.customerCount; i++) {
            const destSelect = document.getElementById(`dest${i}-select`);
            if (destSelect && destSelect.value && this.networkData.nodes[destSelect.value]) {
                const pos = this.networkData.nodes[destSelect.value];
                const sx = this.toScreenX(pos.x);
                const sy = this.toScreenY(pos.y);
                const col = CUSTOMER_PALETTE[(i - 1) % CUSTOMER_PALETTE.length];

                this.ctx.shadowColor = col.glow;
                this.ctx.shadowBlur = 10;

                this.ctx.fillStyle = col.hex;
                this.ctx.beginPath();
                this.ctx.arc(sx, sy, 8, 0, Math.PI * 2);
                this.ctx.fill();

                this.ctx.shadowBlur = 0;
                this.ctx.lineWidth = 2.5;
                this.ctx.strokeStyle = '#ffffff';
                this.ctx.stroke();

                this.ctx.fillStyle = '#ffffff';
                this.ctx.font = 'bold 11px sans-serif';
                this.ctx.fillText(`C${i}`, sx + 12, sy + 4);
            }
        }

        // 6. Draw Live Vehicles from SUMO
        if (this.vehicles && this.vehicles.length > 0) {
            this.vehicles.forEach(v => {
                const sx = this.toScreenX(v.x);
                const sy = this.toScreenY(v.y);

                this.ctx.fillStyle = '#f59e0b';
                this.ctx.beginPath();
                this.ctx.arc(sx, sy, 4, 0, Math.PI * 2);
                this.ctx.fill();
            });
        }

        // 7. Draw Visual Legend on Map Canvas
        if (this.activeRoute && this.activeRoute.geometry && this.activeRoute.geometry.length > 1) {
            this.drawRouteLegend();
        }
    }

    drawRouteLegend() {
        const boxWidth = 175;
        const itemHeight = 20;
        const totalItems = 2 + Math.min(this.customerCount, 6);
        const boxHeight = 28 + totalItems * itemHeight;
        const bx = this.canvas.width - boxWidth - 15;
        const by = 15;

        this.ctx.save();
        this.ctx.fillStyle = "rgba(13, 17, 23, 0.88)";
        this.ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        if (this.ctx.roundRect) {
            this.ctx.roundRect(bx, by, boxWidth, boxHeight, 6);
        } else {
            this.ctx.rect(bx, by, boxWidth, boxHeight);
        }
        this.ctx.fill();
        this.ctx.stroke();

        this.ctx.fillStyle = "#94a3b8";
        this.ctx.font = "bold 10px sans-serif";
        this.ctx.fillText("ROUTE LEGS & STOPS", bx + 10, by + 18);

        let cy = by + 34;

        // Depot
        this.ctx.fillStyle = "#ef4444";
        this.ctx.beginPath();
        this.ctx.arc(bx + 16, cy - 3, 5, 0, Math.PI * 2);
        this.ctx.fill();
        this.ctx.fillStyle = "#ffffff";
        this.ctx.font = "10px sans-serif";
        this.ctx.fillText("Depot (Origin)", bx + 28, cy);
        cy += itemHeight;

        // Customers C1, C2, ...
        for (let i = 1; i <= Math.min(this.customerCount, 6); i++) {
            const col = CUSTOMER_PALETTE[(i - 1) % CUSTOMER_PALETTE.length];
            this.ctx.strokeStyle = col.hex;
            this.ctx.lineWidth = 3;
            this.ctx.beginPath();
            this.ctx.moveTo(bx + 10, cy - 3);
            this.ctx.lineTo(bx + 22, cy - 3);
            this.ctx.stroke();

            this.ctx.fillStyle = col.hex;
            this.ctx.beginPath();
            this.ctx.arc(bx + 16, cy - 3, 4, 0, Math.PI * 2);
            this.ctx.fill();

            this.ctx.fillStyle = "#ffffff";
            this.ctx.font = "10px sans-serif";
            this.ctx.fillText(`C${i} Leg (${col.name})`, bx + 28, cy);
            cy += itemHeight;
        }

        // Return Leg
        this.ctx.strokeStyle = "#94a3b8";
        this.ctx.lineWidth = 2.5;
        this.ctx.setLineDash([4, 3]);
        this.ctx.beginPath();
        this.ctx.moveTo(bx + 10, cy - 3);
        this.ctx.lineTo(bx + 22, cy - 3);
        this.ctx.stroke();
        this.ctx.setLineDash([]);

        this.ctx.fillStyle = "#94a3b8";
        this.ctx.font = "10px sans-serif";
        this.ctx.fillText("Return to Depot", bx + 28, cy);

        this.ctx.restore();
    }

    // WebSocket Connection
    connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        this.ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

        this.ws.onopen = () => {
            const ind = document.getElementById('ws-indicator');
            const txt = document.getElementById('ws-status-text');
            if (ind) ind.className = 'dot online';
            if (txt) txt.innerText = 'Connected';
        };

        this.ws.onmessage = (event) => {
            try {
                const msg = JSON.parse(event.data);
                if (msg.type === 'sim_update') {
                    this.onSimUpdate(msg);
                }
            } catch (e) {
                console.error("Failed to parse WS message:", e);
            }
        };

        this.ws.onclose = () => {
            const ind = document.getElementById('ws-indicator');
            const txt = document.getElementById('ws-status-text');
            if (ind) ind.className = 'dot offline';
            if (txt) txt.innerText = 'Disconnected';
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }

    sendSimCmd(cmd) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({ command: cmd, gui: true }));
        }
    }

    onSimUpdate(data) {
        this.simRunning = data.running;
        this.simPaused = data.paused;
        this.simTime = data.sim_time;
        this.vehicles = data.vehicles || [];
        this.activeIncidents = data.active_incidents || [];

        document.getElementById('sim-clock').innerText = `${this.simTime.toFixed(1)}s`;
        document.getElementById('stat-active').innerText = data.active_vehicles || 0;
        document.getElementById('stat-arrived').innerText = data.arrived_vehicles || 0;
        document.getElementById('stat-congested').innerText = (data.congested_edges || []).length;
        document.getElementById('stat-incidents').innerText = this.activeIncidents.length;

        const simInd = document.getElementById('sim-indicator');
        const simTxt = document.getElementById('sim-status-text');
        if (simInd && simTxt) {
            if (this.simRunning) {
                simInd.className = this.simPaused ? 'dot stopped' : 'dot running';
                simTxt.innerText = this.simPaused ? 'SUMO Paused' : 'SUMO Running';
            } else {
                simInd.className = 'dot stopped';
                simTxt.innerText = 'SUMO Idle';
            }
        }

        document.getElementById('btn-start').disabled = this.simRunning;
        document.getElementById('btn-pause').disabled = !this.simRunning;
        document.getElementById('btn-step').disabled = !this.simRunning;
        document.getElementById('btn-stop').disabled = !this.simRunning;

        if (data.active_route) {
            this.activeRoute = data.active_route;
            this.updateRouteMetrics(data.active_route);
        }

        if (data.events && Array.isArray(data.events)) {
            this.renderSimEvents(data.events);
        }

        this.draw();
    }

    renderSimEvents(events) {
        const listElem = document.getElementById('sim-events-list');
        const badgeElem = document.getElementById('events-count-badge');
        if (!listElem) return;
        if (badgeElem) badgeElem.innerText = `${events.length} Events`;

        if (!events || events.length === 0) {
            listElem.innerHTML = '<li style="color: #6b7280; font-style: italic;">No fleet events recorded yet. Start simulation and solve fleet routing to view live execution events.</li>';
            return;
        }

        const typeColors = {
            'VEHICLE_DEPARTED': '#38bdf8',
            'VEHICLE_ARRIVED_AT_CUSTOMER': '#f59e0b',
            'DELIVERY_COMPLETED': '#10b981',
            'LOAD_CHANGED': '#a78bfa',
            'VEHICLE_RETURNED_TO_DEPOT': '#34d47a',
            'INCIDENT_INJECTED': '#ef4444',
            'INCIDENT_CLEARED': '#06b6d4',
            'REROUTE_TRIGGERED': '#f97316',
        };

        listElem.innerHTML = events.slice().reverse().map(ev => {
            const col = typeColors[ev.type] || '#9ca3af';
            return `
                <li style="margin-bottom: 5px; line-height: 1.35; display: flex; align-items: flex-start; gap: 6px;">
                    <span style="color: #64748b; font-family: 'JetBrains Mono', monospace; font-size: 10px; flex-shrink: 0;">[t=${(ev.timestamp || 0).toFixed(1)}s]</span>
                    <span style="background: ${col}22; color: ${col}; border: 1px solid ${col}44; border-radius: 3px; padding: 0 4px; font-size: 9px; font-weight: 700; flex-shrink: 0;">${ev.type}</span>
                    <span style="color: #e2e8f0; font-size: 11px;">${ev.message}</span>
                </li>
            `;
        }).join('');
    }
}

// Initialize application on DOM load
window.addEventListener('DOMContentLoaded', () => {
    window.app = new TrafficApp();
});
