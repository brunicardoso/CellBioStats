function displayStats(statsText) {
    document.getElementById("stats-text").textContent = statsText;
}

function displayKsInfo(ksInfo) {
    const indicator = document.getElementById("ks-indicator");
    const dot = document.getElementById("ks-dot");
    const text = document.getElementById("ks-text");

    if (!ksInfo) {
        indicator.classList.add("hidden");
        return;
    }

    indicator.classList.remove("hidden");
    if (ksInfo.warning) {
        indicator.className = "mt-4 flex items-center gap-2 text-sm px-3 py-2 rounded-lg bg-amber-50 border border-amber-200";
        dot.className = "inline-block w-2.5 h-2.5 rounded-full bg-amber-400";
        text.className = "text-amber-700";
    } else {
        indicator.className = "mt-4 flex items-center gap-2 text-sm px-3 py-2 rounded-lg bg-green-50 border border-green-200";
        dot.className = "inline-block w-2.5 h-2.5 rounded-full bg-green-400";
        text.className = "text-green-700";
    }
    text.textContent = "Downsampling: " + ksInfo.summary;
}

function initResults(getSessionId, getParams) {
    const fontSizeInput = document.getElementById("font-size");
    const fontSizeVal = document.getElementById("font-size-val");
    const markerSizeInput = document.getElementById("marker-size");
    const markerSizeVal = document.getElementById("marker-size-val");
    const downsampleRadios = document.querySelectorAll('input[name="downsample"]');
    const downsamplePct = document.getElementById("downsample-pct");
    const downsampleNum = document.getElementById("downsample-num");
    const replotSpinner = document.getElementById("replot-spinner");

    // Slider labels
    fontSizeInput.addEventListener("input", () => { fontSizeVal.textContent = fontSizeInput.value; });
    markerSizeInput.addEventListener("input", () => { markerSizeVal.textContent = markerSizeInput.value; });

    // Downsample toggles
    downsampleRadios.forEach((radio) => {
        radio.addEventListener("change", () => {
            downsamplePct.disabled = radio.value !== "percentage";
            downsampleNum.disabled = radio.value !== "number";
        });
    });

    // Load color maps
    API.getColorMaps().then((maps) => {
        const colorMapSelect = document.getElementById("color-map");
        maps.forEach((name) => {
            const opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            if (name === "Plotly") opt.selected = true;
            colorMapSelect.appendChild(opt);
        });
    });

    // Debounced replot
    let replotTimer = null;
    function scheduleReplot() {
        // Only replot if results section is visible
        const resultsSection = document.getElementById("section-results");
        if (resultsSection.classList.contains("section-hidden")) return;

        clearTimeout(replotTimer);
        replotTimer = setTimeout(async () => {
            replotSpinner.classList.remove("hidden");
            try {
                const params = getParams();
                const result = await API.analyze(params);
                renderPlot(result.plot_json);
                displayStats(result.stats_text);
                displayKsInfo(result.ks_info);
            } catch (err) {
                showError(err.message);
            } finally {
                replotSpinner.classList.add("hidden");
            }
        }, 400);
    }

    // Bind replot to all customization controls
    fontSizeInput.addEventListener("input", scheduleReplot);
    markerSizeInput.addEventListener("input", scheduleReplot);
    document.getElementById("color-map").addEventListener("change", scheduleReplot);
    document.getElementById("plot-style").addEventListener("change", scheduleReplot);
    document.getElementById("log-scale").addEventListener("change", scheduleReplot);
    document.getElementById("show-stars").addEventListener("change", scheduleReplot);
    document.getElementById("show-axes").addEventListener("change", scheduleReplot);
    downsampleRadios.forEach((radio) => radio.addEventListener("change", scheduleReplot));
    downsamplePct.addEventListener("input", scheduleReplot);
    downsampleNum.addEventListener("input", scheduleReplot);

    // Download handlers
    document.getElementById("btn-download-stats").addEventListener("click", async () => {
        try {
            await API.downloadStats(getSessionId());
        } catch (err) {
            showError(err.message);
        }
    });

    document.getElementById("btn-download-plot").addEventListener("click", async () => {
        const spinner = document.getElementById("plot-download-spinner");
        spinner.classList.remove("hidden");
        try {
            const params = getParams();
            params.format = document.getElementById("plot-format").value;
            params.scale = parseInt(document.getElementById("plot-resolution").value);
            await API.downloadPlot(params);
        } catch (err) {
            showError(err.message);
        } finally {
            spinner.classList.add("hidden");
        }
    });
}
