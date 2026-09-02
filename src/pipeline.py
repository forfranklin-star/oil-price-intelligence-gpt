from __future__ import annotations
import argparse, json, traceback
from datetime import datetime
import pandas as pd
from .common import load_config, root_path
from .sources import MarketSource, DieselSource, SourceError
from .model import make_features, forecast
from .report import build_report

def run():
    cfg=load_config(); status=[]
    market_parts=[]
    for key in ['wti','brent','dxy','cnyusd']:
        x=MarketSource(cfg).get(key); x['series']=key; market_parts.append(x); status.append({'name':key,'date':str(x['date'].max().date()),'source':x.iloc[-1]['source']})
    market=pd.concat(market_parts,ignore_index=True)
    diesel=DieselSource(cfg).collect()
    latest_diesel=diesel.iloc[-1]
    status.append({'name':'diesel','date':str(latest_diesel['date'].date()),'source':latest_diesel['source']})
    now=pd.Timestamp.now()
    for item,limit in [(market.groupby('series')['date'].max(),cfg['app']['max_age_days']['market'])]:
        for name,date in item.items():
            if (now-date).days>limit: raise SourceError(f'{name} stale: {date.date()}')
    if (now-latest_diesel['date']).days>cfg['app']['max_age_days']['diesel']: raise SourceError(f'diesel stale: {latest_diesel["date"].date()}')
    features=make_features(market,diesel)
    pred=forecast(features,{'short':cfg['model']['short_days'],'medium':cfg['model']['medium_days'],'long':cfg['model']['long_days']})
    latest={s:{'date':str(g.iloc[-1]['date'].date()),'value':float(g.iloc[-1]['value']),'source':str(g.iloc[-1]['source'])} for s,g in market.groupby('series')}
    latest['diesel']={'date':str(latest_diesel['date'].date()),'value':float(latest_diesel['value']),'source':str(latest_diesel['source'])}
    result={'latest':latest,'forecast':pred,'training_rows':len(features),'sources':status}
    build_report(result,cfg['app']['report_dir'])
    return result

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--print-summary',action='store_true'); args=ap.parse_args()
    try:
        result=run(); print(json.dumps(result,ensure_ascii=False,indent=2,default=str)); return 0
    except Exception as e:
        err={'status':'FAILED','time':datetime.now().isoformat(timespec='seconds'),'error':str(e),'traceback':traceback.format_exc()}
        d=root_path(load_config()['app']['report_dir']); d.mkdir(parents=True,exist_ok=True)
        (d/'last_error.json').write_text(json.dumps(err,ensure_ascii=False,indent=2),encoding='utf-8')
        print(json.dumps(err,ensure_ascii=False,indent=2)); return 1
if __name__=='__main__': raise SystemExit(main())
