import joblib, glob, os
import numpy as np

model_file = max(glob.glob('./models/model_rf_*.joblib'), key=os.path.getctime)
pipe = joblib.load(model_file)


X_new = [
    [120000.0, 180000.0, 4.5, 0.002, -0.001, 40000.0, 0.005],  
    [90000.0, 140000.0, 3.1, 0.0005, 0.0008, 35000.0, 0.004], 
]

preds = pipe.predict(X_new)           
probs = pipe.predict_proba(X_new)[:,1] 
for x,p,pr in zip(X_new, preds, probs):
    print(f"features={x} => pred={p} prob_real={pr:.3f}")