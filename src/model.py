from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from statsmodels.tsa.arima.model import ARIMA
FEATURES=['wti','brent','dxy','cnyusd','diesel_lag1','wti_ret','brent_ret','dxy_ret','cny_ret']

def make_features(market, diesel):
    m=market.pivot(index='date',columns='series',values='value').sort_index()
    d=diesel.set_index('date')['value'].rename('diesel')
    x=m.join(d,how='left')
    # Point-in-time joins only. No ffill/bfill/interpolation.
    x['diesel_lag1']=x['diesel'].shift(1)
    for c in ['wti','brent','dxy','cnyusd']:
        x[c+'_ret']=x[c].pct_change()
    x['target']=x['diesel'].shift(-1)
    return x.dropna(subset=FEATURES+['target'])

def forecast(frame, horizons):
    y=frame['diesel'].astype(float)
    if len(y)<120: raise RuntimeError(f'not enough complete real observations: {len(y)} < 120')
    ar=ARIMA(y,order=(1,1,1),trend=None).fit()
    short=np.asarray(ar.forecast(horizons['short'])).tolist()
    train=frame.iloc[:-1]
    rf=RandomForestRegressor(n_estimators=300,max_depth=6,min_samples_leaf=4,random_state=42,n_jobs=-1)
    rf.fit(train[FEATURES],train['target'])
    last=frame.iloc[-1][FEATURES].to_frame().T
    med=float(rf.predict(last)[0])
    ridge=Ridge(alpha=1.0).fit(train[FEATURES],train['target'])
    ridge_pred=float(ridge.predict(last)[0])
    long=float(np.mean([med,ridge_pred]))
    imp=pd.Series(rf.feature_importances_,index=FEATURES).sort_values(ascending=False)
    return {'short':short,'medium':med,'long':long,'feature_importance':imp.to_dict(),'last_diesel':float(y.iloc[-1])}
