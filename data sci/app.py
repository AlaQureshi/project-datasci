from flask import Flask, request, jsonify, render_template, render_template_string, send_from_directory
import os
import subprocess
from scripts.descriptive_statistics import execute_full_statistics
from scripts.Dvis import process_and_plot_data
from scripts.super import execute_full_supervised_learning 
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
    target_column = request.form.get('Income')
    n_components = request.form.get('n_components')
    model_type = request.form.get('knn_regression')
    x_column = request.form.get('Age')
    y_column = request.form.get('Salary')
    file_type = request.form.get('csv')
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        if script == 'data_scaling':
            scaled_data, error = execute_full_scaling(filepath, method)
            output = "Data scaling completed successfully."
            scaled_data_json = scaled_data.to_json(orient='records')  # Serialize DataFrame to JSON
            return jsonify({'output': output, 'scaled_data': scaled_data_json})
        
        elif script == 'data_visualization':
            output = process_and_plot_data(filepath, x_column, y_column)

        elif script == 'descriptive_statistics':
            output = execute_full_statistics(filepath)

        elif script == 'supervised_learning':
            results = execute_full_supervised_learning(filepath, 'Income', 'linear_regression', is_continuous=True)
            output = f"Model trained successfully with results: {results}"

        elif script == 'unsupervised_learning':
            results = execute_full_unsupervised_learning(filepath, 'csv')
            output = f"Unsupervised learning applied successfully with results: {results}"

        return jsonify({'Results': output})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    finally:
        os.remove(filepath)  # Clean up the file after processing


if __name__ == '__main__':
    app.run(debug=True)
