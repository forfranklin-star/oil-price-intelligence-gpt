import json, subprocess, sys
from pathlib import Path
import streamlit as st
ROOT=Path(__file__).resolve().parent
REPORT=ROOT/'reports/latest_report.json'; ERR=ROOT/'reports/last_error.json'
st.set_page_config(page_title='多因素油价智能分析',layout='wide')
st.title('多因素油价智能分析与预测系统')
st.caption('真实数据模式：禁止模拟、随机数据、插值补齐、代理价格替代目标价格。')
def run_pipeline():
    with st.spinner('正在访问真实数据源、校验时效并训练模型…'):
        return subprocess.run([sys.executable,'-m','src.pipeline','--print-summary'],cwd=ROOT,text=True,capture_output=True)

if st.button('立即采集真实数据并生成日报',type='primary'):
    p=run_pipeline()
    if p.returncode==0: st.success('真实数据日报生成成功。'); st.rerun()
    else: st.error('真实数据 Pipeline 失败；系统没有用假数据顶替。'); st.code(p.stdout or p.stderr)

if not REPORT.exists() and not ERR.exists():
    p=run_pipeline()
    if p.returncode==0: st.rerun()
if REPORT.exists():
    data=json.loads(REPORT.read_text(encoding='utf-8'))
    st.success('已有有效日报')
    cols=st.columns(4)
    for i,k in enumerate(['wti','brent','dxy','cnyusd']): cols[i].metric(k.upper(),f"{data['latest'][k]['value']:.4f}",data['latest'][k]['date'])
    st.metric('中国柴油',f"{data['latest']['diesel']['value']:.2f} 元/吨",data['latest']['diesel']['date'])
    st.subheader('预测')
    st.write('7日最后一期：',data['forecast']['short'][-1])
    st.write('30日：',data['forecast']['medium'])
    st.write('90日：',data['forecast']['long'])
    st.subheader('特征权重')
    st.dataframe(data['forecast']['feature_importance'])
    st.download_button('下载日报 JSON',REPORT.read_bytes(),'latest_report.json')
elif ERR.exists():
    err=json.loads(ERR.read_text(encoding='utf-8'))
    st.error('本次真实数据采集失败；系统没有用假数据顶替。')
    st.code(err.get('error','unknown error'))
    st.caption('修复数据源后重新点击上面的按钮即可。')
else:
    st.info('尚未生成日报。点击“立即采集真实数据并生成日报”，无需再打开终端。')
