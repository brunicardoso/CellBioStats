const API = {
    async upload(file) {
        const form = new FormData();
        form.append("file", file);
        const res = await fetch("/api/upload", { method: "POST", body: form });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Upload failed");
        }
        return res.json();
    },

    async analyze(params) {
        const res = await fetch("/api/analyze", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({ detail: res.statusText }));
            throw new Error(err.detail || "Analysis failed");
        }
        return res.json();
    },

    async downloadStats(sessionId) {
        const res = await fetch(`/api/download/stats/${sessionId}`);
        if (!res.ok) throw new Error("Download failed");
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition") || "";
        const match = disposition.match(/filename="(.+)"/);
        const filename = match ? match[1] : "stats.txt";
        triggerDownload(blob, filename);
    },

    async downloadPlot(params) {
        const res = await fetch("/api/download/plot", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(params),
        });
        if (!res.ok) throw new Error("Plot download failed");
        const blob = await res.blob();
        const disposition = res.headers.get("Content-Disposition") || "";
        const match = disposition.match(/filename="(.+)"/);
        const filename = match ? match[1] : `plot.${params.format}`;
        triggerDownload(blob, filename);
    },

    async getColorMaps() {
        const res = await fetch("/api/color-maps");
        if (!res.ok) return ["Plotly"];
        const data = await res.json();
        return data.color_maps;
    },
};

function triggerDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}
