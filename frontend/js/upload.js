function initUpload() {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const feedback = document.getElementById("upload-feedback");
    const spinner = document.getElementById("upload-spinner");

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files[0]) handleFile(fileInput.files[0]);
    });

    async function handleFile(file) {
        feedback.classList.add("hidden");
        spinner.classList.remove("hidden");

        try {
            const data = await API.upload(file);
            spinner.classList.add("hidden");
            App.onUploadSuccess(data);
        } catch (err) {
            spinner.classList.add("hidden");
            feedback.textContent = err.message;
            feedback.className = "mt-4 text-sm text-red-600";
            feedback.classList.remove("hidden");
        }
    }
}
