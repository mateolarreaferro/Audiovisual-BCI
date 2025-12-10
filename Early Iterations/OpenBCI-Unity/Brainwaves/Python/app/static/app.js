/**
 * EEG Terrain Visualizer - Web UI JavaScript
 * Handles real-time updates, controls, and API communication
 */

class EegControlPanel {
    constructor() {
        this.updateInterval = null;
        this.isStreaming = false;
        
        this.initializeElements();
        this.bindEvents();
        this.startStatusUpdates();
        
        console.log('EEG Control Panel initialized');
    }
    
    initializeElements() {
        // Control elements
        this.elements = {
            // Status
            statusIndicator: document.getElementById('status-indicator'),
            packetRate: document.getElementById('packet-rate'),
            
            // Controls
            modeSelect: document.getElementById('mode-select'),
            unityIp: document.getElementById('unity-ip'),
            unityPort: document.getElementById('unity-port'),
            startBtn: document.getElementById('start-btn'),
            stopBtn: document.getElementById('stop-btn'),
            
            // Generator settings
            generatorType: document.getElementById('generator-type'),
            frequency: document.getElementById('frequency'),
            amplitude: document.getElementById('amplitude'),
            noiseLevel: document.getElementById('noise-level'),
            freqValue: document.getElementById('freq-value'),
            ampValue: document.getElementById('amp-value'),
            noiseValue: document.getElementById('noise-value'),
            
            // Triggers
            calmTrigger: document.getElementById('calm-trigger'),
            engagedTrigger: document.getElementById('engaged-trigger'),
            
            // EEG Values
            alphaValue: document.getElementById('alpha-value'),
            betaValue: document.getElementById('beta-value'),
            thetaValue: document.getElementById('theta-value'),
            deltaValue: document.getElementById('delta-value'),
            arousalValue: document.getElementById('arousal-value'),
            alphaBar: document.getElementById('alpha-bar'),
            betaBar: document.getElementById('beta-bar'),
            thetaBar: document.getElementById('theta-bar'),
            deltaBar: document.getElementById('delta-bar'),
            arousalBar: document.getElementById('arousal-bar'),
            
            // State indicators
            calmIndicator: document.getElementById('calm-indicator'),
            engagedIndicator: document.getElementById('engaged-indicator'),
            
            // Statistics
            packetsSent: document.getElementById('packets-sent'),
            sendRate: document.getElementById('send-rate'),
            uptime: document.getElementById('uptime'),
            errorCount: document.getElementById('error-count'),
            
            // Manual controls
            manualAlpha: document.getElementById('manual-alpha'),
            manualBeta: document.getElementById('manual-beta'),
            manualTheta: document.getElementById('manual-theta'),
            manualDelta: document.getElementById('manual-delta'),
            manualAlphaVal: document.getElementById('manual-alpha-val'),
            manualBetaVal: document.getElementById('manual-beta-val'),
            manualThetaVal: document.getElementById('manual-theta-val'),
            manualDeltaVal: document.getElementById('manual-delta-val'),
            sendTest: document.getElementById('send-test'),
            
            // Error log
            errorLog: document.getElementById('error-log'),
            
            // Demo settings
            demoSettings: document.getElementById('demo-settings')
        };
        
        // Check if elements exist
        Object.entries(this.elements).forEach(([key, element]) => {
            if (!element) {
                console.warn(`Element not found: ${key}`);
            }
        });
    }
    
    bindEvents() {
        // Control buttons
        if (this.elements.startBtn) {
            this.elements.startBtn.addEventListener('click', () => this.startStreaming());
        }
        
        if (this.elements.stopBtn) {
            this.elements.stopBtn.addEventListener('click', () => this.stopStreaming());
        }
        
        // Configuration changes
        if (this.elements.modeSelect) {
            this.elements.modeSelect.addEventListener('change', () => this.updateConfig());
        }
        
        if (this.elements.unityIp) {
            this.elements.unityIp.addEventListener('change', () => this.updateConfig());
        }
        
        if (this.elements.unityPort) {
            this.elements.unityPort.addEventListener('change', () => this.updateConfig());
        }
        
        // Generator settings
        [this.elements.generatorType, this.elements.frequency, 
         this.elements.amplitude, this.elements.noiseLevel].forEach(element => {
            if (element) {
                element.addEventListener('input', () => this.updateGeneratorSettings());
            }
        });
        
        // Manual trigger buttons
        if (this.elements.calmTrigger) {
            this.elements.calmTrigger.addEventListener('click', () => this.triggerState('calm'));
        }
        
        if (this.elements.engagedTrigger) {
            this.elements.engagedTrigger.addEventListener('click', () => this.triggerState('engaged'));
        }
        
        // Manual control sliders
        [this.elements.manualAlpha, this.elements.manualBeta,
         this.elements.manualTheta, this.elements.manualDelta].forEach(element => {
            if (element) {
                element.addEventListener('input', () => this.updateManualValues());
            }
        });
        
        if (this.elements.sendTest) {
            this.elements.sendTest.addEventListener('click', () => this.sendTestSignal());
        }
        
        // Update slider value displays
        this.updateSliderDisplays();
    }
    
    async updateConfig() {
        const config = {
            mode: this.elements.modeSelect?.value || 'demo',
            unity_ip: this.elements.unityIp?.value || '127.0.0.1',
            unity_port: parseInt(this.elements.unityPort?.value) || 7777
        };
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (!result.success) {
                this.showError(`Config update failed: ${result.error}`);
            }
            
            // Show/hide demo settings based on mode
            if (this.elements.demoSettings) {
                this.elements.demoSettings.style.display = 
                    config.mode === 'demo' ? 'block' : 'none';
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    async updateGeneratorSettings() {
        const config = {
            generator_type: this.elements.generatorType?.value || 'sine',
            frequency: parseFloat(this.elements.frequency?.value) || 0.5,
            amplitude: parseFloat(this.elements.amplitude?.value) || 0.4,
            noise_level: parseFloat(this.elements.noiseLevel?.value) || 0.1
        };
        
        // Update display values
        if (this.elements.freqValue) {
            this.elements.freqValue.textContent = config.frequency.toFixed(1);
        }
        if (this.elements.ampValue) {
            this.elements.ampValue.textContent = config.amplitude.toFixed(2);
        }
        if (this.elements.noiseValue) {
            this.elements.noiseValue.textContent = config.noise_level.toFixed(2);
        }
        
        try {
            const response = await fetch('/api/config', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(config)
            });
            
            const result = await response.json();
            
            if (!result.success) {
                this.showError(`Generator config failed: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    updateManualValues() {
        const values = ['Alpha', 'Beta', 'Theta', 'Delta'];
        values.forEach(value => {
            const slider = this.elements[`manual${value}`];
            const display = this.elements[`manual${value}Val`];
            if (slider && display) {
                display.textContent = parseFloat(slider.value).toFixed(2);
            }
        });
    }
    
    updateSliderDisplays() {
        // Update generator slider displays
        ['frequency', 'amplitude', 'noiseLevel'].forEach(param => {
            const slider = this.elements[param];
            if (slider) {
                slider.dispatchEvent(new Event('input'));
            }
        });
        
        // Update manual control displays
        this.updateManualValues();
    }
    
    async startStreaming() {
        try {
            this.setLoadingState(true);
            
            const response = await fetch('/api/start', {
                method: 'POST',
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isStreaming = true;
                this.updateControlStates();
                this.showSuccess('Streaming started');
            } else {
                this.showError(`Failed to start: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        } finally {
            this.setLoadingState(false);
        }
    }
    
    async stopStreaming() {
        try {
            this.setLoadingState(true);
            
            const response = await fetch('/api/stop', {
                method: 'POST',
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.isStreaming = false;
                this.updateControlStates();
                this.showSuccess('Streaming stopped');
            } else {
                this.showError(`Failed to stop: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        } finally {
            this.setLoadingState(false);
        }
    }
    
    async triggerState(state) {
        try {
            const response = await fetch(`/api/trigger/${state}`, {
                method: 'POST',
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(`${state} trigger sent`);
            } else {
                this.showError(`Trigger failed: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    async sendTestSignal() {
        const signal = {
            alpha: parseFloat(this.elements.manualAlpha?.value) || 0.5,
            beta: parseFloat(this.elements.manualBeta?.value) || 0.5,
            theta: parseFloat(this.elements.manualTheta?.value) || 0.5,
            delta: parseFloat(this.elements.manualDelta?.value) || 0.5,
            arousal: 0.5,
            calm: 0,
            engaged: 0
        };
        
        try {
            const response = await fetch('/api/test_signal', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(signal)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('Test signal sent');
            } else {
                this.showError(`Test failed: ${result.error}`);
            }
            
        } catch (error) {
            this.showError(`Network error: ${error.message}`);
        }
    }
    
    startStatusUpdates() {
        this.updateInterval = setInterval(() => {
            this.updateStatus();
        }, 100); // Update every 100ms
    }
    
    async updateStatus() {
        try {
            const response = await fetch('/api/status');
            const result = await response.json();
            
            if (result.success) {
                this.updateUI(result.data);
            }
            
        } catch (error) {
            // Silently fail for status updates to avoid spam
            if (this.elements.statusIndicator) {
                this.elements.statusIndicator.textContent = 'DISCONNECTED';
                this.elements.statusIndicator.className = 'badge bg-danger';
            }
        }
    }
    
    updateUI(data) {
        const config = data.config;
        const stats = data.stats;
        const running = data.running;
        
        // Update status indicator
        if (this.elements.statusIndicator) {
            if (running && stats.send_rate > 0) {
                this.elements.statusIndicator.textContent = 'STREAMING';
                this.elements.statusIndicator.className = 'badge bg-success';
            } else if (running) {
                this.elements.statusIndicator.textContent = 'STARTING';
                this.elements.statusIndicator.className = 'badge bg-warning';
            } else {
                this.elements.statusIndicator.textContent = 'STOPPED';
                this.elements.statusIndicator.className = 'badge bg-secondary';
            }
        }
        
        // Update packet rate
        if (this.elements.packetRate) {
            this.elements.packetRate.textContent = `${stats.send_rate.toFixed(1)} Hz`;
        }
        
        // Update streaming state
        this.isStreaming = running;
        this.updateControlStates();
        
        // Update statistics
        if (this.elements.packetsSent) {
            this.elements.packetsSent.textContent = stats.packets_sent;
        }
        if (this.elements.sendRate) {
            this.elements.sendRate.textContent = stats.send_rate.toFixed(1);
        }
        if (this.elements.uptime) {
            this.elements.uptime.textContent = Math.floor(stats.uptime_seconds);
        }
        if (this.elements.errorCount) {
            this.elements.errorCount.textContent = stats.error_count;
        }
        
        // Update error log
        if (data.recent_errors && this.elements.errorLog) {
            const errorHtml = data.recent_errors.length > 0
                ? data.recent_errors.map(error => `<div>${error}</div>`).join('')
                : 'No errors';
            this.elements.errorLog.innerHTML = errorHtml;
            this.elements.errorLog.scrollTop = this.elements.errorLog.scrollHeight;
        }
        
        // Update form values
        this.updateFormValues(config);
    }
    
    updateFormValues(config) {
        // Only update if not focused (to avoid interrupting user input)
        if (this.elements.modeSelect && document.activeElement !== this.elements.modeSelect) {
            this.elements.modeSelect.value = config.mode;
        }
        
        if (this.elements.unityIp && document.activeElement !== this.elements.unityIp) {
            this.elements.unityIp.value = config.unity_ip;
        }
        
        if (this.elements.unityPort && document.activeElement !== this.elements.unityPort) {
            this.elements.unityPort.value = config.unity_port;
        }
        
        // Update demo settings visibility
        if (this.elements.demoSettings) {
            this.elements.demoSettings.style.display = 
                config.mode === 'demo' ? 'block' : 'none';
        }
    }
    
    updateControlStates() {
        if (this.elements.startBtn) {
            this.elements.startBtn.disabled = this.isStreaming;
        }
        
        if (this.elements.stopBtn) {
            this.elements.stopBtn.disabled = !this.isStreaming;
        }
        
        // Enable/disable trigger buttons
        [this.elements.calmTrigger, this.elements.engagedTrigger, 
         this.elements.sendTest].forEach(btn => {
            if (btn) {
                btn.disabled = !this.isStreaming;
            }
        });
    }
    
    setLoadingState(loading) {
        const buttons = [this.elements.startBtn, this.elements.stopBtn];
        buttons.forEach(btn => {
            if (btn) {
                btn.disabled = loading;
                if (loading) {
                    btn.classList.add('loading');
                } else {
                    btn.classList.remove('loading');
                }
            }
        });
    }
    
    showSuccess(message) {
        // Simple success feedback (could be enhanced with toast notifications)
        console.log('Success:', message);
    }
    
    showError(message) {
        // Simple error feedback (could be enhanced with toast notifications)
        console.error('Error:', message);
        
        // Add to error log
        if (this.elements.errorLog) {
            const timestamp = new Date().toLocaleTimeString();
            const errorDiv = document.createElement('div');
            errorDiv.className = 'text-danger';
            errorDiv.textContent = `[${timestamp}] ${message}`;
            this.elements.errorLog.appendChild(errorDiv);
            this.elements.errorLog.scrollTop = this.elements.errorLog.scrollHeight;
        }
    }
    
    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
    }
}

// Initialize the control panel when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.eegPanel = new EegControlPanel();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (window.eegPanel) {
        window.eegPanel.destroy();
    }
});