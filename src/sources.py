from __future__ import annotations
import io, re, html as htmlmod
from datetime import datetime, timezone
from urllib.parse import urljoin
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from .common import load_config

class SourceError(RuntimeError): pass

class HTTP:
    def __init__(self, cfg):
        self.timeout = int(cfg['data']['timeout'])
        self.headers = {'User-Agent': cfg['data']['user_agent'], 'Accept-Language':'zh-CN,zh;q=0.9,en;q=0.8'}
    def get(self, url):
        r = requests.get(url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return r


def _clean(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    date_col = next((c for c in df.columns if str(c).lower() in {'date','日期','datetime','时间'}), df.columns[0])
    val_col = next((c for c in df.columns if c != date_col and str(c).lower() in {'value','price','价格','值'}), df.columns[-1])
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], errors='coerce'), 'value': pd.to_numeric(df[val_col], errors='coerce')})
    out = out.dropna().drop_duplicates('date').sort_values('date')
    out = out[out['date'] <= pd.Timestamp.now(tz='UTC').tz_localize(None)]
    if out.empty: raise SourceError('source returned no numeric observations')
    return out

class MarketSource:
    def __init__(self, cfg=None): self.cfg = cfg or load_config(); self.http=HTTP(self.cfg)
    def fred(self, series):
        url=self.cfg['data']['fred_csv'].format(series=series)
        text=self.http.get(url).text
        df=pd.read_csv(io.StringIO(text))
        return _clean(df)
    def yahoo(self, symbol, days=730):
        url=self.cfg['data']['yahoo_chart'].format(symbol=symbol)
        r=self.http.get(url).json()['chart']['result'][0]
        ts=r.get('timestamp') or []
        q=r['indicators']['quote'][0]['close']
        df=pd.DataFrame({'date':pd.to_datetime(ts,unit='s',utc=True).tz_convert(None),'value':q})
        return _clean(df).tail(days)
    def get(self, key):
        s=self.cfg['series']
        primary=s[key+'_yahoo']
        fallback=s[key+'_fred']
        errors=[]
        for name, fn in [('Yahoo', lambda:self.yahoo(primary,self.cfg['data']['history_days'])),('FRED/EIA',lambda:self.fred(fallback))]:
            try:
                x=fn(); x['source']=name; return x
            except Exception as e: errors.append(f'{name}: {e}')
        raise SourceError(f'{key} unavailable; ' + ' | '.join(errors))

class DieselSource:
    def __init__(self,cfg=None): self.cfg=cfg or load_config(); self.http=HTTP(self.cfg)
    @staticmethod
    def _numbers(text):
        vals=[]
        for m in re.findall(r'(?<![\d.])(\d{4,5}(?:\.\d+)?)(?![\d.])', text.replace(',','')):
            v=float(m)
            if 3000<=v<=20000: vals.append(v)
        return vals
    def _parse_article(self,url, html):
        soup=BeautifulSoup(html,'lxml')
        title=soup.get_text(' ',strip=True)[:300]
        date=None
        m=re.search(r'(20\d{2})[-年./](\d{1,2})[-月./](\d{1,2})', title)
        if m: date=pd.Timestamp(f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}')
        for table in soup.find_all('table'):
            text=table.get_text(' ',strip=True)
            if '柴油' not in text: continue
            vals=self._numbers(text)
            if vals and date is not None:
                return date, float(np.median(vals))
        text=soup.get_text(' ',strip=True)
        if '柴油' in text:
            vals=self._numbers(text)
            if vals and date is not None: return date, float(np.median(vals))
        return None
    def longzhong(self):
        base=self.cfg['diesel']['longzhong_category']
        pages=[base]+[self.cfg['diesel']['longzhong_archive'].format(page=i) for i in range(1,int(self.cfg['diesel']['archive_pages'])+1)]
        seen=set(); rows=[]
        for page in pages:
            try: html=self.http.get(page).text
            except Exception: continue
            soup=BeautifulSoup(html,'lxml')
            links=[]
            for a in soup.find_all('a',href=True):
                txt=a.get_text(' ',strip=True)
                if ('柴油' in txt or '成品油' in txt) and ('2026' in txt or '2025' in txt):
                    links.append(urljoin(page,a['href']))
            for url in links:
                if url in seen or len(seen)>=int(self.cfg['diesel']['max_articles']): continue
                seen.add(url)
                try:
                    got=self._parse_article(url,self.http.get(url).text)
                    if got: rows.append((*got,'Longzhong'))
                except Exception: continue
        if not rows: raise SourceError('Longzhong: no public diesel price observations parsed')
        return pd.DataFrame(rows,columns=['date','value','source']).drop_duplicates('date').sort_values('date')
    def business(self):
        url=self.cfg['diesel']['business_society']; html=self.http.get(url).text
        soup=BeautifulSoup(html,'lxml'); rows=[]
        for tr in soup.find_all('tr'):
            t=[x.get_text(' ',strip=True) for x in tr.find_all(['td','th'])]
            if not t: continue
            text=' '.join(t)
            if '柴油' not in text: continue
            ds=re.findall(r'20\d{2}[-./]\d{1,2}[-./]\d{1,2}',text)
            vals=self._numbers(text)
            if ds and vals: rows.append((pd.Timestamp(ds[0]),float(vals[-1]),'BusinessSociety'))
        if not rows: raise SourceError('Business Society: no public diesel observations parsed')
        return pd.DataFrame(rows,columns=['date','value','source']).drop_duplicates('date').sort_values('date')
    def csv(self):
        url=self.cfg['diesel'].get('configured_csv','').strip()
        if not url: raise SourceError('configured CSV is empty')
        df=pd.read_csv(url)
        return _clean(df).assign(source='configured_csv')
    def collect(self):
        candidates=[]; errors=[]
        for name,fn in [('Longzhong',self.longzhong),('BusinessSociety',self.business),('CSV',self.csv)]:
            try:
                x=fn();
                if len(x)>=int(self.cfg['diesel']['min_rows']): candidates.append(x)
            except Exception as e: errors.append(f'{name}: {e}')
        if not candidates: raise SourceError('diesel unavailable; ' + ' | '.join(errors))
        # Whole-series selection only: no splicing between sources.
        candidates.sort(key=lambda x:(x['date'].max(),len(x)), reverse=True)
        chosen=candidates[0]
        if len(candidates)>1:
            latest=chosen.iloc[-1]['value']; gaps=[abs(latest-x.iloc[-1]['value'])/latest*100 for x in candidates[1:] if latest]
            if gaps and min(gaps)>float(self.cfg['diesel']['crosscheck_gap_pct']):
                raise SourceError(f'diesel cross-check failed: independent sources differ by > {self.cfg["diesel"]["crosscheck_gap_pct"]}%')
        return chosen.reset_index(drop=True)
