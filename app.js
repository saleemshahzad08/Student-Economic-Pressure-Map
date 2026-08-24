// Enabling Strict Mode
"use strict";

// Base API Constant
const API_BASE_URL = 'https://student-economic-pressure-api.onrender.com';

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

form.addEventListener('submit', function(event) {
    event.preventDefault();
    const formData = {
        internet_cost: internetCostSelect.value,
        device_access: deviceStrainSelect.value,
        work_reason: workMotivationSelect.value,
        education_improvement: institutionalImprovementSelect.value
    }
    const isEmpty = Object.values(formData).some(value => value === "");
    if (isEmpty) {
        alert("Please fill all the fields");
        console.log("At least one field in the form is empty.")
        return;
    }
    console.log('Submitting Profile:', formData)
})