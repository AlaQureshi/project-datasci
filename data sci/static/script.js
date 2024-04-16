document.addEventListener('DOMContentLoaded', function () {
    document.getElementById('data_scaling_button').addEventListener('click', function() {
        runScript('data_scaling');
    });
    document.getElementById('data_visualization_button').addEventListener('click', function() {
        runScript('data_visualization');
    });
    document.getElementById('descriptive_statistics_button').addEventListener('click', function() {
        runScript('descriptive_statistics');
    });
    document.getElementById('supervised_learning_button').addEventListener('click', function() {
        runScript('supervised_learning');
    });
    document.getElementById('unsupervised_learning_button').addEventListener('click', function() {
        runScript('unsupervised_learning');
    });
});

function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert('Please select a file to upload first.');
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    fetch('/upload_file', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
      .then(data => {
        if (data.error) {
            alert(data.error);
        } else {
            document.getElementById('output').innerText = 'File uploaded successfully: ' + JSON.stringify(data, null, 2);
        }
    }).catch(error => console.error('Error:', error));
}

function runScript(scriptName) {
    const formData = new FormData();
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length === 0) {
        alert('Please upload a file first.');
        return;
    }
    formData.append('file', fileInput.files[0]);
    formData.append('script', scriptName);

    fetch('/run_script', {
        method: 'POST',
        body: formData
    }).then(response => response.json())
      .then(data => {
        if (data.error) {
            alert(data.error);
        } else if (scriptName === 'data_visualization') {
            // Assume the backend returns the name of the plot
            requestVisualization(data.plotName);
        } else {
            document.getElementById('output').innerText = JSON.stringify(data, null, 2);
        }
    }).catch(error => console.error('Error:', error));
}

function requestVisualization(plotName) {
    const plotUrl = `/get_plot/${plotName}`;
    const imageElement = document.createElement('img');
    imageElement.src = plotUrl;
    document.getElementById('output').appendChild(imageElement);
}

