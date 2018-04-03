import numpy as np
import flask
import io
import sys
from squad.demo_prepro import prepro
from basic.demo_cli import Demo

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)

class MachineComprehend:
	def __init__(self):
		self.model = Demo()

	def answer_question(self, paragraph, question):
		pq_prepro = prepro(paragraph, question)
		answer = self.model.run(pq_prepro)
		if len(answer) == 0:
			return None
		return answer


# ## Ask Away!
@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the view
    data = {"success": False, "predictions": []}

    file_name = 'The-Future-Computed_2.8.18.txt'
    mc = MachineComprehend()
    p = open(file_name, 'r')
    paragraph = []
    for line in p:
        paragraph.append(line.rstrip())
    paragraph = ". ".join(paragraph)
    #question = (flask.request.files["question"].read()).decode("utf-8")
    question = (flask.request.data).decode("utf-8")
    print("**********************************************************")
    print(question)
    y_output = mc.answer_question(paragraph, question)
    print(y_output)
    data["predictions"].append(str(y_output))
    
    #indicate that the request was a success
    data["success"] = True
    #return the data dictionary as a JSON response
    return flask.jsonify(data)


if __name__== "__main__":
    print(("* Loading Keras model and Flask starting server..."
		"please wait until server has fully started"))

    app.run(host='0.0.0.0') # Ignore, Development server



