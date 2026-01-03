// File Upload Handlers
class FileUploader {
    constructor() {
        this.setupImageUpload();
        this.setupDataUpload();
    }

    setupImageUpload() {
        const uploadZone = document.getElementById('image-upload-zone');
        const fileInput = document.getElementById('image-file-input');
        const browseBtn = document.getElementById('browse-image');
        const previewContainer = document.getElementById('image-preview-container');
        const preview = document.getElementById('image-preview');

        // Browse button click
        browseBtn.addEventListener('click', () => fileInput.click());

        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleImageFile(file, preview, previewContainer, uploadZone);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#00d4ff';
            uploadZone.style.background = 'rgba(0, 212, 255, 0.1)';
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
            uploadZone.style.background = '';

            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                this.handleImageFile(file, preview, previewContainer, uploadZone);
            } else {
                showToast('Please upload an image file', 'error');
            }
        });
    }

    handleImageFile(file, preview, previewContainer, uploadZone) {
        const reader = new FileReader();
        reader.onload = (e) => {
            preview.src = e.target.result;
            uploadZone.classList.add('hidden');
            previewContainer.classList.remove('hidden');

            // Store file for later use
            window.currentImageFile = file;
        };
        reader.readAsDataURL(file);
    }

    setupDataUpload() {
        const uploadZone = document.getElementById('data-upload-zone');
        const fileInput = document.getElementById('data-file-input');
        const browseBtn = document.getElementById('browse-data');
        const configContainer = document.getElementById('data-config-container');

        // Browse button click
        browseBtn.addEventListener('click', () => fileInput.click());

        // File input change
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                this.handleDataFile(file, configContainer, uploadZone);
            }
        });

        // Drag and drop
        uploadZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '#00d4ff';
        });

        uploadZone.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';
        });

        uploadZone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadZone.style.borderColor = '';

            const file = e.dataTransfer.files[0];
            if (file && file.name.endsWith('.csv')) {
                this.handleDataFile(file, configContainer, uploadZone);
            } else {
                showToast('Please upload a CSV file', 'error');
            }
        });
    }

    handleDataFile(file, configContainer, uploadZone) {
        uploadZone.classList.add('hidden');
        configContainer.classList.remove('hidden');

        // Store file for later use
        window.currentDataFile = file;

        showToast(`File "${file.name}" loaded successfully`, 'success');
    }
}

// Initialize file uploader
let fileUploader;

document.addEventListener('DOMContentLoaded', () => {
    fileUploader = new FileUploader();
});
