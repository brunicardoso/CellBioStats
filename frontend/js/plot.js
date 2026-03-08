function renderPlot(plotJson) {
    const container = document.getElementById("plot-container");
    const figure = JSON.parse(plotJson);
    // Make plot responsive
    figure.layout = figure.layout || {};
    figure.layout.autosize = true;
    delete figure.layout.width;
    Plotly.newPlot(container, figure.data, figure.layout, {
        responsive: true,
        displayModeBar: true,
        modeBarButtonsToRemove: ["lasso2d", "select2d"],
    });
}
