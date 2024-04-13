from flask import Flask, request, jsonify, render_template, render_template_string
import os
import subprocess
from scripts.eda import execute_full_eda
from scripts.descriptive_statistics import execute_full_statistics
from scripts.data_wrangling import execute_full_wrangling
from scripts.data_visualization import execute_full_visualization
from scripts.lda import execute_full_lda  
from scripts.pca import execute_full_pca
from scripts.supervised_learning import execute_full_supervised_learning  
from scripts.unsupervised_learning import execute_full_unsupervised_learning  
from scripts.data_scaling import execute_full_scaling

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    return jsonify({'message': 'File uploaded successfully'})

@app.route('/run_script', methods=['POST'])
def run_script():
    file = request.files['file']
    script = request.form.get('script')
    method = request.form.get('method', 'standard')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        if script == 'data_scaling':
            scaled_data, error = execute_full_scaling(filepath, method)
            output = "Data scaling completed successfully."
            scaled_data_json = scaled_data.to_json(orient='records')  # Serialize DataFrame to JSON
            return jsonify({'output': output, 'scaled_data': scaled_data_json})
        elif script == 'data_visualization':
            result, message = execute_full_visualization(filepath)
        elif script == 'data_wrangling':
            processed_data, error = execute_full_wrangling(filepath)
            return jsonify({'data': processed_data})
        elif script == 'descriptive_statistics':
            output = execute_full_statistics(filepath)
        elif script == 'eda':
            output = execute_full_eda(filepath)
        elif script == 'lda':
            lda_plot = execute_full_lda(filepath, target_column, n_components)
            output = "LDA plot created successfully."
        elif script == 'pca':
            pca_plot, principal_df = execute_full_pca(filepath, n_components)
            output = "PCA plot created successfully."
        elif script == 'supervised_learning':
            results = execute_full_supervised_learning(filepath, target_column, model_type)
            output = f"Model trained successfully with results: {results}"
        elif script == 'unsupervised_learning':
            results = execute_full_unsupervised_learning(filepath, file_type)
            output = f"Unsupervised learning applied successfully with results: {results}"
        return jsonify({'output': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        os.remove(filepath)  # Clean up the file after processing


if __name__ == '__main__':
    app.run(debug=True)
