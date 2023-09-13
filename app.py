from flask import Flask, render_template, request
from PIL import Image
import base64
import cv2
import numpy as np

from model_final import processing_image

app = Flask(__name__)
#This line creates an instance of the Flask class. The __name__ parameter is passed to allow Flask to determine the root path for the application.

# Function to convert a DataFrame to an HTML table
def df_to_html(dataframe):
    return dataframe.to_html(classes="table table-bordered table-hover", header=True, index=False, escape=False)

# Function to convert a NumPy array to a base64-encoded image
def numpy_array_to_base64_image(numpy_array):
    _, buffer = cv2.imencode('.png', numpy_array)
    return base64.b64encode(buffer).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
#This is a decorator that associates the / URL with the index function below. 
# It specifies that the function should be called when a user accesses the root URL (/). 
# The methods parameter specifies that this route can handle both GET and POST requests.
def index():
    if request.method == "POST":
        desired_pattern = (request.form.get('pattern_type'))
        count = (request.form.get('count'))
        fabric_choice = (request.form.get('fabric_choice'))       
        uploaded_file = request.files["file"]

        if uploaded_file.filename != "":
            image = Image.open(uploaded_file) #Opens the processed image using the PIL Image.open() method.
            cv2Image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
            processedData = processing_image(cv2Image, desired_pattern, count)
            
            # Convert the NumPy array to a base64-encoded image
            image_base64 = numpy_array_to_base64_image(processedData['image_with_gridlines'])
        

            # Create an HTML table from the color_table DataFrame
            color_table_html = df_to_html(processedData['color_table'])
    

            return render_template("index.html", image_base64=image_base64, color_table_html=color_table_html)   

    return render_template("index.html")

#This line is executed when a GET request is made to the root URL or after processing a POST request. It renders the "index.html" template without any additional context.

if __name__ == "__main__":
    app.run(debug=True)
    # from waitress import serve
    # serve(app, host="0.0.0.0", port=8080)

    

#This block checks if the script is being run directly (not imported as a module). 
#If it's being run directly, the app.run() function starts the Flask development server. The debug=True parameter enables debugging mode.



# commands for storing code in github

# git init => only once. For initializing git version control in project
# git add . => to add changes to the staging area
# git commit -m "some message" => to create a new version with recent staged changes
# git push origin master => to upload version and changes to github