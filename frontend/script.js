document.getElementById('upload-form').addEventListener('submit', function (event) {
    event.preventDefault(); // Prevent the form from submitting the traditional way

    let fileInput = document.getElementById('file-input');
    let formData = new FormData();

    if (fileInput.files.length === 0) {
        alert('Please upload a file first.');
        return;
    }

    formData.append('video', fileInput.files[0]);

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerHTML = `<p>Error: ${data.error}</p>`;
        } else {
            document.getElementById('result').innerHTML = `<p>${data.message}</p>`;
            // If you want to remove the video after detection, clear the file input
            fileInput.value = '';
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `<p>Error during detection.</p>`;
    });
});
