const App = (() => {
    let sessionId = null;
    let uploadData = null;

    function showSection(id) {
        ["section-upload", "section-configure", "section-results"].forEach((s) => {
            const el = document.getElementById(s);
            if (s === id) {
                el.style.display = "";
                // Trigger reflow then animate in
                requestAnimationFrame(() => {
                    el.className = "section-visible " + (el.dataset.extraClass || "");
                    if (s === "section-upload") {
                        el.classList.add("min-h-screen", "flex", "flex-col", "items-center", "justify-center", "px-4");
                    } else {
                        el.classList.add("px-4", "py-16");
                    }
                });
            } else {
                el.className = "section-hidden";
            }
        });
        window.scrollTo({ top: 0, behavior: "smooth" });
    }

    function onUploadSuccess(data) {
        sessionId = data.session_id;
        uploadData = data;
        document.getElementById("file-info").textContent = `${data.filename} - ${data.rows} rows`;
        populateColumns(data.columns);
        showDataPreview(data.preview, data.columns);
        showSection("section-configure");
    }

    function getCurrentParams() {
        return getAnalysisParams(sessionId);
    }

    async function runAnalysis() {
        const spinner = document.getElementById("analyze-spinner");
        const btn = document.getElementById("btn-analyze");
        spinner.classList.remove("hidden");
        btn.disabled = true;

        try {
            const params = getCurrentParams();
            const result = await API.analyze(params);
            renderPlot(result.plot_json);
            displayStats(result.stats_text);
            displayKsInfo(result.ks_info);
            showSection("section-results");
        } catch (err) {
            showError(err.message);
        } finally {
            spinner.classList.add("hidden");
            btn.disabled = false;
        }
    }

    function init() {
        initUpload();
        initMapping();
        initResults(() => sessionId, getCurrentParams);

        document.getElementById("btn-analyze").addEventListener("click", runAnalysis);
        document.getElementById("btn-back-upload").addEventListener("click", () => showSection("section-upload"));
        document.getElementById("btn-back-configure").addEventListener("click", () => showSection("section-configure"));
    }

    document.addEventListener("DOMContentLoaded", init);

    return { onUploadSuccess };
})();

function showError(msg) {
    const toast = document.getElementById("error-toast");
    toast.textContent = msg;
    toast.classList.remove("hidden");
    setTimeout(() => toast.classList.add("hidden"), 5000);
}
