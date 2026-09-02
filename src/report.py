from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
from .common import root_path

def build_report(result, out_dir):
    out=root_path(out_dir); out.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    payload={'generated_at':datetime.now().isoformat(timespec='seconds'),'status':'OK',**result}
    latest=result['latest']
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>油价智能日报</title><style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:30px auto;line-height:1.6}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}.ok{{color:green}}</style></head><body><h1>多因素油价智能日报</h1><p class="ok">真实数据状态：OK</p><p>生成时间：{payload['generated_at']}</p><h2>最新真实观测</h2><table><tr><th>指标</th><th>日期</th><th>值</th><th>来源</th></tr>{''.join(f"<tr><td>{k}</td><td>{v['date']}</td><td>{v['value']:.4f}</td><td>{v['source']}</td></tr>" for k,v in latest.items())}</table><h2>预测</h2><p>短期7日最后一期：{result['forecast']['short'][-1]:.2f} 元/吨</p><p>中期30日：{result['forecast']['medium']:.2f} 元/吨</p><p>长期90日：{result['forecast']['long']:.2f} 元/吨</p><h2>学习到的特征权重</h2><table><tr><th>特征</th><th>权重</th></tr>{''.join(f"<tr><td>{k}</td><td>{v:.6f}</td></tr>" for k,v in result['forecast']['feature_importance'].items())}</table></body></html>'''
    (out/'latest_report.html').write_text(html,encoding='utf-8')
    (out/'latest_report.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    (out/f'report_{stamp}.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    return out/'latest_report.html'
