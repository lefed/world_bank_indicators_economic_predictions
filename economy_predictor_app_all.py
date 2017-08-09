import flask
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.externals import joblib

#---------- MODEL IN MEMORY ----------------#

# Read the revised WDI Data on economies,
# Build a GradientBoostedTrees Regression predictor on it
WDI_model_df = pd.read_csv("flask_app_df.csv")

#---['SP.DYN.IMRT.IN', 'SH.DYN.MORT','NY.GDP.FRST.RT.ZS','shrinking_next_yr','SP.POP.DPND.OL', 'dev_SP.POP.2024.FE.5Y', 'SP.POP.2024.FE.5Y', 'SP.POP.3034.FE.5Y', 'NY.GNS.ICTR.GN.ZS', 'time_del_MS.MIL.XPND.GD.ZS', 'SP.DYN.TFRT.IN', 'EN.ATM.CO2E.PC', 'SP.POP.5559.FE.5Y', 'NE.IMP.GNFS.ZS', 'NY.GDS.TOTL.ZS', 'TM.VAL.FUEL.ZS.UN', 'FS.AST.PRVT.GD.ZS', 'SE.PRM.ENRR', 'NV.SRV.TETC.ZS', 'dev_AG.PRD.FOOD.XD', 'IT.CEL.SETS.P2', 'SP.DYN.CDRT.IN', 'time_diff_SP.POP.DPND', 'NY.ADJ.DCO2.GN.ZS', 'NV.IND.TOTL.ZS', 'SP.DYN.LE00.MA.IN', 'SP.DYN.TO65.MA.ZS', 'FI.RES.TOTL.MO', 'time_del_AG.PRD.CROP.XD', 'NY.GDP.DEFL.KD.ZG', 'SP.DYN.LE00.FE.IN', 'SP.RUR.TOTL.ZS', 'NE.TRD.GNFS.ZS']-----#

final_useful_features = ['SP.POP.DPND.OL', 'SP.DYN.IMRT.IN', 'SH.DYN.MORT', 'NY.GDP.FRST.RT.ZS', 'dev_SP.POP.2024.FE.5Y', 'SP.POP.2024.FE.5Y', 'SP.POP.3034.FE.5Y', 'NY.GNS.ICTR.GN.ZS',  'time_del_MS.MIL.XPND.GD.ZS', 'SP.DYN.TFRT.IN', 'EN.ATM.CO2E.PC', 'SP.POP.5559.FE.5Y', 'NE.IMP.GNFS.ZS', 'NY.GDS.TOTL.ZS', 'TM.VAL.FUEL.ZS.UN', 'FS.AST.PRVT.GD.ZS', 'SE.PRM.ENRR', 'NV.SRV.TETC.ZS', 'dev_AG.PRD.FOOD.XD', 'IT.CEL.SETS.P2', 'SP.DYN.CDRT.IN', 'time_diff_SP.POP.DPND', 'NY.ADJ.DCO2.GN.ZS', 'NV.IND.TOTL.ZS', 'SP.DYN.LE00.MA.IN', 'SP.DYN.TO65.MA.ZS', 'FI.RES.TOTL.MO', 'time_del_AG.PRD.CROP.XD', 'NY.GDP.DEFL.KD.ZG', 'SP.DYN.LE00.FE.IN', 'SP.RUR.TOTL.ZS', 'NE.TRD.GNFS.ZS']

target_ind = ['shrinking_next_yr']

feature_list = [x for x in final_useful_features if x not in target_ind]

grad_boost_model = joblib.load('wdi_grad_boost_model.pkl')
#grad_boost_model = GradientBoostingClassifier(loss = 'exponential', learning_rate = 0.065, n_estimators = 75, min_samples_split = 3 )

X = WDI_model_df[feature_list]
Y = WDI_model_df[target_ind]

PREDICTOR = grad_boost_model.fit(X,Y)


#---------- URLS AND WEB PAGES -------------#

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """
    Homepage: serve our visualization page, app_landing.html
    """
    with open("app_landing_all.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/score", methods=["POST"])
def score():
    """
    When A POST request with json data is made to this uri,
    Read the example from the json, predict probability and
    send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    x = np.matrix(data['example'])
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    results = {"score": score[0,1]}
    return flask.jsonify(results)
    

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)
app.run(host='0.0.0.0')
app.run(debug=True)
