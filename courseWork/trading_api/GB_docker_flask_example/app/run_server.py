# USAGE
# Start the server:
# 	python run_front_server.py
# Submit a request via Python:
#	python simple_request.py
# import the necessary packages
import json
import logging
import os
from logging.handlers import RotatingFileHandler
import flask
import pandas as pd
from time import strftime
import dill
dill._dill._reverse_typemap['ClassType'] = type

#  request payload example
# {
#   "open": 1528.96,
#   "high": 1529.4,
#   "low": 1527.57,
#   "close": 1528.66,
#   "EMA_4_ema4o": 1529.72009850086,
#   "EMA_5_ema5c": 1529.48590620679,
#   "spread": 170,
#   "tick_volume": 3112,
#   "CCI_14_0.015": -95.3727110857222,
#   "RSI_14": 53.8782078975101,
#   "MOM_14": 6.06000000000017,
#   "STOCHRSIk_14_14_3_3": 15.6530333057249,
#   "STOCHRSId_14_14_3_3": 30.295676250451
# }


# initialize our Flask application and the model
app = flask.Flask(__name__)
model = None

handler = RotatingFileHandler(filename='app.log', maxBytes=100000, backupCount=10)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(handler)


def load_model(model_path):
	# load the pre-trained model
	global dmodel
	with open(model_path, 'rb') as f:
		dmodel = dill.load(f)
	print(dmodel)

modelpath = "/app/app/models/catboost_pipeline.dill"
modelpath = "../../../models/catboost_pipeline.dill" # TODO change according to docker volume (above)
load_model(modelpath)


@app.route("/", methods=["GET"])
def general():
	return """Welcome to trading long prediction process. Please use 'http://<address>/predict' to POST"""


@app.route("/predict", methods=["POST"])
def predict():
	# initialize the data dictionary that will be returned from the view
	data = {"success": False}
	try:
		# ensure an image was properly uploaded to our endpoint
		if flask.request.method == "POST":
			content = flask.request.get_json(silent=True)
			logger.info(f'Data: {content}')
			vars = ['open', 'high', 'low', 'close', 'EMA_4_ema4o', 'EMA_5_ema5c', 'spread', 'tick_volume', 'CCI_14_0.015', 'RSI_14', 'MOM_14', 'STOCHRSIk_14_14_3_3', 'STOCHRSId_14_14_3_3']
			missing =[]
			ddf = {}
			for el in vars:
				val = content.get(el)
				ddf[el] = val
				if val is None:
					missing.append(el)

			logger.info(f'Missing: {missing}')

			if len(missing) > 0:
				data["missing_features"] = str(missing)
				return flask.jsonify(data)

			df = pd.DataFrame(ddf, index=[0])
			logger.info(f'tmp {df}')
			preds = dmodel.get('model').predict_proba(df)
			data["prediction"] = preds[:, 1][0]
			# indicate that the request was a success
			data["success"] = True
	except AttributeError as e:
		logger.warning(f'{df} Exception: {str(e)}')
		data['predictions'] = str(e)
		data['success'] = False

	# return the data dictionary as a JSON response
	return flask.jsonify(data)


# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
	print(("* Loading the model and Flask starting server...please wait until server has fully started"))
	port = int(os.environ.get('PORT', 8180))
	app.run(host='0.0.0.0', debug=True, port=port)




