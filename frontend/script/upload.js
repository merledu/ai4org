//Define Maximum File Size as 50MB
    const MAX_FILE_SIZE = 50 * 1024 * 1024;

    //Define Upload Chunk Size as 1MB
    const CHUNK_SIZE = 1024 * 1024;

    function arrayBufferToBase64(buffer) {
      let binary = "";
      const bytes = new Uint8Array(buffer);
      const len = bytes.byteLength;

      for(let i = 0; i < len; i++) {
        binary += String.fromCharCode(bytes[i]);
      }

      return btoa(binary);
    }

    function resetUploadUI() {
      const progressContainer = document.querySelector(".progress-container");
      const progressBar = document.getElementById("progressBar");
      const progressText = document.getElementById("progressText");
      const doneMessage = document.getElementById("doneMessage");

      if (progressContainer) {
        progressContainer.style.display = "none";
      }
      if (progressBar) {
        progressBar.style.width = "0%";
      }
      if (progressText) {
        progressText.textContent = "0%";
      }
      if (doneMessage) {
        doneMessage.style.display = "none";
      }
    }

    async function uploadFile() {
      const fileInput = document.getElementById("fileInput");

      if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
        alert("Please select a file first.");
        return;
      }

      const file = fileInput.files[0];

      if(file.size > MAX_FILE_SIZE) {
        alert("File is too large. Maximum allowed size is 50 MB.");
        return;
      }

      const progressContainer = document.querySelector(".progress-container");
      const progressBar = document.getElementById("progressBar");
      const progressText = document.getElementById("progressText");
      const doneMessage = document.getElementById("doneMessage");

      progressContainer.style.display = "block";
      progressBar.style.width = "0%";
      progressText.textContent = "0%";
      doneMessage.style.display = "none";

      let offset = 0;
      let chunkIndex = 0;

      if (!window.pywebview || !window.pywebview.api || typeof window.pywebview.api.save_file_chunk !== "function") {
        alert("Upload service is not available. Please try again.");
        resetUploadUI();
        return;
      }

      while(offset < file.size) {
        const chunk = file.slice(offset, offset + CHUNK_SIZE);
        const arrayBuffer = await chunk.arrayBuffer();

        const base64Chunk = arrayBufferToBase64(arrayBuffer);
        const result = await window.pywebview.api.save_file_chunk(
          file.name,
          base64Chunk,
          chunkIndex,
          offset + CHUNK_SIZE >= file.size
        );
        if(result !== "success") {
          alert("Upload failed: " + result);
          resetUploadUI();
          return;
        }

        offset = Math.min(offset + CHUNK_SIZE, file.size);
        chunkIndex++;

        const progress = Math.round(Math.min((offset/file.size)*100, 100));
        progressBar.style.width = progress + "%";
        progressText.textContent = progress + "%";
      }
      doneMessage.style.display = "block";
      setTimeout(() => { window.location.href = "chat.html"; }, 1000);
    }

    function goToChat() {
      window.location.href = "chat.html";
    }
