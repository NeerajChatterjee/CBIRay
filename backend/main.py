from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from lbp.utils import retrieve_similar_images_lbp, load_lbp_database_features
from cnn.utils import retrieve_similar_images_vgg, load_cnn_features_and_model
from combined.utils import retrieve_similar_images_combined, load_combined_features_and_model

UPLOAD_FOLDER = './static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*", "expose_headers": "*"}})

isLoad = True

if isLoad:
    load_lbp_database_features()
    load_cnn_features_and_model()
    load_combined_features_and_model()
    isLoad = False


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/search", methods=["POST"])
def search():
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files or 'model' not in request.form or 'numberOfImages' not in request.form:
            return Response("{'message': 'Bad request', 'status_code': 400}", status=400, mimetype='application/json')
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return Response("{'message': 'Bad request', 'status_code': 400}", status=400, mimetype='application/json')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            file.save(file_path)

            if request.form.get('model') == 'lbp':
                images_count = request.form.get('numberOfImages')

                [images_filenames, images_values, images_classification] =\
                    retrieve_similar_images_lbp(file_path, images_count)

                if len(images_filenames) == 0 or len(images_values) == 0 or len(images_classification) == 0:
                    return Response("{'message': 'Server internal error', 'status_code': 500}", status=500,
                                    mimetype='application/json')

                response = {
                    'message': 'Successful',
                    'status_code': 200,
                    'files': list(images_filenames),
                    'similarityValues': list(images_values),
                    'classifications': list(images_classification)
                }
                return jsonify(response), 200

            elif request.form.get('model') == 'vgg16':
                images_count = request.form.get('numberOfImages')

                [images_filenames, images_values, images_classification] = \
                    retrieve_similar_images_vgg(file_path, images_count)

                if len(images_filenames) == 0 or len(images_values) == 0 or len(images_classification) == 0:
                    return Response("{'message': 'Server internal error', 'status_code': 500}", status=500,
                                    mimetype='application/json')

                response = {
                    'message': 'Successful',
                    'status_code': 200,
                    'files': list(images_filenames),
                    'similarityValues': list(images_values),
                    'classifications': list(images_classification)
                }
                return jsonify(response), 200
            elif request.form.get('model') == 'combined':
                images_count = request.form.get('numberOfImages')

                [images_filenames, images_values, images_classification] = \
                    retrieve_similar_images_combined(file_path, images_count)

                if len(images_filenames) == 0 or len(images_values) == 0 or len(images_classification) == 0:
                    return Response("{'message': 'Server internal error', 'status_code': 500}", status=500,
                                    mimetype='application/json')

                response = {
                    'message': 'Successful',
                    'status_code': 200,
                    'files': list(images_filenames),
                    'similarityValues': list(images_values),
                    'classifications': list(images_classification)
                }
                return jsonify(response), 200
            else:
                return Response("{'message': 'Model not supported', 'status_code': 400}", status=400,
                                mimetype='application/json')


app.run(debug=True, port=5001)
