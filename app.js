// Enabling Strict Mode
"use strict";

// Base API Constant
const API_BASE_URL = 'https://student-economic-pressure-map.onrender.com';

// Caching DOM Elements
// 1. Form & Controls
const form = document.getElementById('prediction-form');
const internetCostSelect = document.getElementById('internet-cost');
const deviceStrainSelect = document.getElementById('device-strain');
const workMotivationSelect = document.getElementById('work-motivation');
const institutionalImprovementSelect = document.getElementById('institutional-improvement');
const submitBtn = document.getElementById('submit-btn');

// 2. Loading State
const loadingIndicator = document.getElementById('loading-indicator');
const coldStartNotice = document.getElementById('cold-start-notice');

// 3. Result Presentation
const resultsSection = document.getElementById('results-section');
const predictedCategory = document.getElementById('predicted-category');
const primaryBadge = document.querySelector('.primary-prediction-badge');
const probLowVal = document.getElementById('prob-low-val');
const probModVal = document.getElementById('prob-mod-val');
const probHighVal = document.getElementById('prob-high-val');
const probLowFill = document.getElementById('prob-low-fill');
const probModFill = document.getElementById('prob-mod-fill');
const probHighFill = document.getElementById('prob-high-fill');

// 4. Uncertainty & Logic
const uncertaintyBadge = document.getElementById('uncertainty-badge');
const ruleTraceText = document.getElementById('rule-trace-text');
const resetBtn = document.getElementById('reset-btn');

// Pre-warming Frontend To Avoid Cold-Start on Render
async function preWarmBackend() {
  try {
    await fetch(`${API_BASE_URL}/health`, {
      method: "GET",
      headers: { "Content-Type": "application/json" }
    });
    console.debug("Backend pre-warming ping successful.");
  } catch (err) {
    console.debug("Backend pre-warming ping dispatched.");
  }
}

document.addEventListener("DOMContentLoaded", preWarmBackend);

/// Timer reference holder
let serverWakeTimer = null;

// Centralized UI State Manager
function toggleLoadingState(isLoading) {
    if (isLoading) {
        submitBtn.disabled = true;
        submitBtn.textContent = "Processing...";
        loadingIndicator.classList.remove('hidden');
        resultsSection.classList.add('hidden');

        serverWakeTimer = setTimeout(() => {
            coldStartNotice.classList.remove('hidden');
        }, 2500);

    } else {
        if (serverWakeTimer) {
            clearTimeout(serverWakeTimer);
            serverWakeTimer = null;
        }

        loadingIndicator.classList.add('hidden');
        coldStartNotice.classList.add('hidden');
        submitBtn.disabled = false;
        submitBtn.textContent = "Generate Cohort Estimate";
    }
}

// Asynchronous API Handler with Defensive Hardening
async function performServerTask(payload) {
    toggleLoadingState(true);

    const controller = new AbortController();
    const timeoutDuration = 60000; // 60 seconds max timeout for cold starts
    const timeoutId = setTimeout(() => controller.abort(), timeoutDuration);

    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
            signal: controller.signal
        });

        if (!response.ok) {
            let errorDetail = `Service error (${response.status})`;
            try {
                const errorData = await response.json();
                if (errorData.detail) {
                    errorDetail = typeof errorData.detail === 'string' 
                        ? errorData.detail 
                        : JSON.stringify(errorData.detail);
                }
            } catch (_) {
                // Non-JSON response body (e.g. 502 Bad Gateway HTML)
            }
            throw new Error(errorDetail);
        }

        const data = await response.json();
        renderResults(data);

    } catch (error) {
        console.error('Inference request failed:', error);
        
        if (error.name === 'AbortError') {
            alert('Request timed out. The backend service may still be initializing. Please wait a moment and try again.');
        } else {
            alert(`Unable to generate cohort estimate: ${error.message}`);
        }
    } finally {
        clearTimeout(timeoutId);
        toggleLoadingState(false);
    }
}

// Triggering the process on Form Submission
form.addEventListener('submit', function(event) {
    event.preventDefault();

    const formData = {
        internet_cost: internetCostSelect.value,
        device_access: deviceStrainSelect.value,
        work_reason: workMotivationSelect.value,
        education_improvement: institutionalImprovementSelect.value
    };

    const isEmpty = Object.values(formData).some(value => value === "");
    if (isEmpty) {
        alert("Please complete all four questionnaire fields.");
        return;
    }

    performServerTask(formData);
});

// Results Renderer Function
function renderResults(data) {

    // Primary Category Text & Accent Color
    predictedCategory.textContent = data.predicted_category;
    if (primaryBadge) {
        primaryBadge.classList.remove('tone-low', 'tone-mod', 'tone-high');
        if (data.predicted_category === 'Low') primaryBadge.classList.add('tone-low');
        else if (data.predicted_category === 'Moderate') primaryBadge.classList.add('tone-mod');
        else if (data.predicted_category === 'High') primaryBadge.classList.add('tone-high');        
    }

    // Probability Values and Animated Fill Widths
    const lowPct = Math.round(data.probabilities.Low * 100);
    const modPct = Math.round(data.probabilities.Moderate * 100);
    const highPct = Math.round(data.probabilities.High * 100);

    probLowVal.textContent = `${lowPct}%`;
    probModVal.textContent = `${modPct}%`;
    probHighVal.textContent = `${highPct}%`;

    probLowFill.style.width = `${lowPct}%`;
    probModFill.style.width = `${modPct}%`;
    probHighFill.style.width = `${highPct}%`;

    // Uncertainty Alert Badge
    if (data.is_uncertain) {
        uncertaintyBadge.classList.remove('hidden');
    } else {
        uncertaintyBadge.classList.add('hidden');
    }

    // Rule-Tracing Explanation
    ruleTraceText.textContent = data.rule_trace;

    // Reveal the Results Section
    resultsSection.classList.remove('hidden');
    resultsSection.scrollIntoView({behavior: 'smooth', block: 'start'});
}

// Reset Button Logic
resetBtn.addEventListener('click', function(event) {
    event.preventDefault();

    form.reset();
    resultsSection.classList.add('hidden');
    window.scrollTo({top: 0, behavior: 'smooth'});
});

