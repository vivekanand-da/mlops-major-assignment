from flask import Flask, request, render_template_string
import joblib
from PIL import Image
import numpy as np

# Load the saved model
model = joblib.load("savedmodel.pth")

app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<html>
  <head>
    <title>Olivetti Face Classifier</title>
  </head>
  <body>
    <h1>Olivetti Face Classifier</h1>
    <p>Upload a grayscale face image (64x64) to get the predicted class.</p>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file" accept="image/*">
      <button type="submit">Predict</button>
    </form>
    {% if prediction is not none %}
      <h2>Predicted Class: {{ prediction }}</h2>
    {% endif %}
  </body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file:
            prediction = "No file uploaded"
        else:
            # Open the image, convert to grayscale, resize to 64x64
            img = Image.open(file).convert("L").resize((64, 64))
            img_array = np.array(img, dtype=np.float32)

            # Normalize to 0–1 range similar to Olivetti faces
            img_array /= 255.0

            # Flatten to shape (1, 4096)
            img_flat = img_array.reshape(1, -1)

            # Predict class
            pred = model.predict(img_flat)[0]
            prediction = int(pred)

    return render_template_string(HTML_PAGE, prediction=prediction)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)