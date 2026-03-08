function initMapping() {
    const treatmentCol = document.getElementById("treatment-col");
    const valueCol = document.getElementById("value-col");
    const replicateCol = document.getElementById("replicate-col");
    const btnAnalyze = document.getElementById("btn-analyze");

    // Enable/disable analyze button
    function checkReady() {
        btnAnalyze.disabled = !(treatmentCol.value && valueCol.value && replicateCol.value);
    }
    treatmentCol.addEventListener("change", checkReady);
    valueCol.addEventListener("change", checkReady);
    replicateCol.addEventListener("change", checkReady);
}

function populateColumns(columns) {
    const selects = ["treatment-col", "value-col", "replicate-col"];
    selects.forEach((id) => {
        const sel = document.getElementById(id);
        sel.innerHTML = '<option value="">Select column...</option>';
        columns.forEach((col) => {
            const opt = document.createElement("option");
            opt.value = col;
            opt.textContent = col;
            sel.appendChild(opt);
        });
    });
}

function showDataPreview(preview, columns) {
    const container = document.getElementById("data-preview-container");
    const head = document.getElementById("preview-head");
    const body = document.getElementById("preview-body");

    head.innerHTML = "<tr>" + columns.map((c) => `<th>${c}</th>`).join("") + "</tr>";
    body.innerHTML = preview
        .map((row) => "<tr>" + columns.map((c) => `<td>${row[c] ?? ""}</td>`).join("") + "</tr>")
        .join("");

    container.classList.remove("hidden");
}

function getAnalysisParams(sessionId) {
    const downsampleMode = document.querySelector('input[name="downsample"]:checked').value;
    let downsampleValue = null;
    if (downsampleMode === "percentage") downsampleValue = parseFloat(document.getElementById("downsample-pct").value);
    if (downsampleMode === "number") downsampleValue = parseFloat(document.getElementById("downsample-num").value);

    return {
        session_id: sessionId,
        treatment_col: document.getElementById("treatment-col").value,
        value_col: document.getElementById("value-col").value,
        replicate_col: document.getElementById("replicate-col").value,
        paired: document.getElementById("paired-toggle").checked,
        font_size: parseInt(document.getElementById("font-size").value),
        color_map: document.getElementById("color-map").value,
        template: document.getElementById("plot-style").value,
        marker_size: parseInt(document.getElementById("marker-size").value),
        downsample_mode: downsampleMode,
        downsample_value: downsampleValue,
        log_scale: document.getElementById("log-scale").checked,
        show_stars: document.getElementById("show-stars").checked,
        show_axes: document.getElementById("show-axes").checked,
    };
}
