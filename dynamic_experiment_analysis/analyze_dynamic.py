"""Analyze dynamic ALINA ROI experiments without modifying or rerunning ALINA.
Run from repository root: python analysis/analyze_dynamic.py
Requires: pandas, numpy, matplotlib (pip install pandas matplotlib).
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METHODS = ['hsv', 'hough', 'ransac', 'kmeans', 'gemini']
VIDEOS = ['vidd_1', 'vidd_2', 'vidd_3']

def truthy(v):
    return str(v).strip().lower() in {'true', '1', 'yes'}

def load_coords(path):
    if not path.exists():
        return None
    points = set()
    for line_no, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 2:
            raise ValueError(f'{path}:{line_no} expected 2 columns, got {values}')
        points.add((int(values[0]), int(values[1])))
    return points

def metric_counts(gt, pred):
    tp = len(gt & pred)
    return tp, len(pred) - tp, len(gt) - tp

def save_table(frame, path):
    frame.to_csv(path, index=False, float_format='%.4f')
    print(f'Saved {path}')

def bar_label(ax, rects, fmt='{:.1f}'):
    for rect in rects:
        height = rect.get_height()
        if np.isfinite(height):
            ax.annotate(fmt.format(height), (rect.get_x()+rect.get_width()/2, height),
                        xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

def plot_figures(df, out):
    summary = []
    for method in METHODS:
        part = df[df.method == method]
        if part.empty:
            continue
        executed = part[part.status == 'ok']
        live = executed[~executed.cache_hit]
        summary.append(dict(method=method, evaluated=len(part), execution_errors=int((part.status != 'ok').sum()),
                            frames_with_lines=int(part.detected.sum()), success_rate_pct=100*part.detected.mean(),
                            mean_roi_ms=executed.roi_ms.mean(), mean_alina_ms=executed.alina_ms.mean(),
                            mean_total_ms=executed.total_ms.mean(),
                            live_roi_ms=live.roi_ms.mean(), live_total_ms=live.total_ms.mean(),
                            live_samples=len(live), cache_hits=int(executed.cache_hit.sum())))
    s = pd.DataFrame(summary)
    save_table(s, out/'overall_summary.csv')
    if s.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    rects = ax.bar(s.method, s.success_rate_pct)
    bar_label(ax, rects, '{:.1f}%')
    ax.set_ylim(0, 110); ax.set_ylabel('Frames with ≥1 detected pixel (%)')
    ax.set_title('Figure 1 — Dynamic ROI detection success rate')
    for i, row in s.reset_index().iterrows():
        ax.text(i, -9, f"n={int(row.evaluated)}", ha='center', fontsize=8)
    fig.subplots_adjust(bottom=.18); fig.savefig(out/'figure_1_detection_success.png', dpi=180, bbox_inches='tight'); plt.close(fig)

    # Only measured live calls are valid Gemini inference times. Cache hits are excluded.
    times = []
    for method in METHODS:
        part = df[(df.method == method) & (df.status == 'ok') & (~df.cache_hit)]
        if part.empty:
            continue
        times.append(dict(method=method, n=len(part), roi_seconds=part.roi_ms.mean()/1000,
                          alina_seconds=part.alina_ms.mean()/1000,
                          total_seconds=part.total_ms.mean()/1000))
    t = pd.DataFrame(times)
    save_table(t, out/'processing_time_summary.csv')
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(t)); w = .38
    a = ax.bar(xs-w/2, t.roi_seconds, w, label='ROI selection')
    b = ax.bar(xs+w/2, t.alina_seconds, w, label='ALINA processing')
    ax.set_xticks(xs, t.method); ax.set_ylabel('Mean time per frame (seconds)')
    ax.set_title('Figure 2 — Processing times (uncached executions only)')
    ax.legend(); fig.tight_layout(); fig.savefig(out/'figure_2_processing_time.png', dpi=180); plt.close(fig)

    rows=[]
    for method in METHODS:
        for video in VIDEOS:
            part=df[(df.method==method)&(df.video==video)]
            if len(part):
                rows.append(dict(method=method,video=video,evaluated=len(part),with_lines=int(part.detected.sum()),
                                 execution_errors=int((part.status!='ok').sum()),success_rate_pct=100*part.detected.mean()))
    per=pd.DataFrame(rows);save_table(per,out/'per_video_summary.csv')
    fig, ax=plt.subplots(figsize=(10,5)); xs=np.arange(len(METHODS)); w=.25
    for j, video in enumerate(VIDEOS):
        vals=[per.loc[(per.method==m)&(per.video==video),'success_rate_pct'].iloc[0]
              if ((per.method==m)&(per.video==video)).any() else np.nan for m in METHODS]
        ax.bar(xs+(j-1)*w, vals, w, label=video)
    ax.set_xticks(xs, METHODS);ax.set_ylim(0,105);ax.set_ylabel('Frames with ≥1 detected pixel (%)')
    ax.set_title('Figure 3 — Detection success by video');ax.legend();fig.tight_layout()
    fig.savefig(out/'figure_3_per_video.png',dpi=180);plt.close(fig)
    print('NOTE: Figure 2 excludes Gemini cache hits; see processing_time_summary.csv for sample sizes.')

def find_prediction(root, video, method, stem):
    folder=root/video/method
    candidates=[folder/f'{stem}_coords.txt',folder/f'{stem}.txt',folder/'coords'/f'{stem}.txt']
    return next((p for p in candidates if p.is_file()), None)

def ground_truth_eval(df, root, gtroot, out):
    rows=[]; matched=[]; missing=[]
    for video in VIDEOS:
        number=video.split('_')[-1]
        truth_dir=gtroot/f'canny_textfiles_{number}'
        if not truth_dir.is_dir():
            print(f'WARNING: ground-truth directory missing: {truth_dir}')
            continue
        for frame in sorted(df.loc[df.video==video,'frame'].unique()):
            stem=Path(frame).stem
            gtfile=truth_dir/f'{stem}.txt'
            if not gtfile.exists():
                continue
            gt=load_coords(gtfile)
            matched.append((video,frame))
            for method in METHODS:
                record=df[(df.video==video)&(df.frame==frame)&(df.method==method)]
                if record.empty:
                    continue # Gemini's unsampled frames are not zeros.
                r=record.iloc[0]
                if r.status != 'ok':
                    missing.append(dict(video=video,frame=frame,method=method,reason='execution error'))
                    continue
                predfile=find_prediction(root,video,method,stem)
                if predfile is None:
                    missing.append(dict(video=video,frame=frame,method=method,reason='prediction file missing'))
                    continue
                pred=load_coords(predfile)
                tp,fp,fn=metric_counts(gt,pred)
                precision=tp/(tp+fp) if tp+fp else (1.0 if not gt else 0.0)
                recall=tp/(tp+fn) if tp+fn else 1.0
                f1=2*precision*recall/(precision+recall) if precision+recall else 0.0
                rows.append(dict(video=video,frame=frame,method=method,gt_pixels=len(gt),pred_pixels=len(pred),
                                 tp=tp,fp=fp,fn=fn,precision=precision,recall=recall,f1=f1))
    save_table(pd.DataFrame(rows,columns=['video','frame','method','gt_pixels','pred_pixels','tp','fp','fn','precision','recall','f1']),out/'cbem_per_frame.csv')
    save_table(pd.DataFrame(missing,columns=['video','frame','method','reason']),out/'cbem_missing_predictions.csv')
    print(f'CBEM matches in evaluated frame IDs: {len(set(matched))}; evaluated method/frame pairs: {len(rows)}')
    if not rows:
        print('No CBEM pairs evaluated. Check prediction naming and paths.');return
    d=pd.DataFrame(rows)
    agg=[]
    for method,group in d.groupby('method'):
        tp,fp,fn=group[['tp','fp','fn']].sum()
        p=tp/(tp+fp) if tp+fp else 0.0
        r=tp/(tp+fn) if tp+fn else 0.0
        agg.append(dict(method=method,frames=len(group),micro_precision=p,micro_recall=r,
                        micro_f1=2*p*r/(p+r) if p+r else 0.0,
                        mean_frame_precision=group.precision.mean(),mean_frame_recall=group.recall.mean(),
                        mean_frame_f1=group.f1.mean()))
    save_table(pd.DataFrame(agg),out/'cbem_summary.csv')
    print('CBEM uses EXACT (x,y) coordinate matching; no pixel tolerance. This differs from ALINA repo evaluation, which averages separate X and Y coordinate metrics.')
    print('Verify CBEM and predictions use the same source frame and image resolution before interpreting scores.')

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--results',type=Path,default=Path('outputs/dynamic_experiments/results.csv'))
    p.add_argument('--experiment-root',type=Path,default=Path('outputs/dynamic_experiments'))
    p.add_argument('--gt-root',type=Path,default=Path('data/gt_alina_labels/canny_textfiles'))
    p.add_argument('--output',type=Path,default=Path('outputs/dynamic_analysis'))
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if not a.results.is_file():p.error(f'Missing results CSV: {a.results}')
    df=pd.read_csv(a.results,dtype={'frame':str}).fillna({'error':''})
    required={'video','frame','method','status','roi_ms','alina_ms','total_ms','detected_pixels','had_lines','cache_hit'}
    if not required.issubset(df.columns):p.error(f'Missing CSV columns: {sorted(required-set(df.columns))}')
    df['frame']=df.frame.map(lambda x:Path(x).name)
    for col in ['roi_ms','alina_ms','total_ms','detected_pixels']:
        df[col]=pd.to_numeric(df[col],errors='coerce')
    df['cache_hit']=df.cache_hit.map(truthy)
    df['detected']=(df.status=='ok') & df.had_lines.map(truthy) & (df.detected_pixels>0)
    dup=df.duplicated(['video','frame','method'],keep=False)
    if dup.any():p.error(f'Duplicate experiment records found: {int(dup.sum())}; resolve before analysis')
    plot_figures(df,a.output)
    ground_truth_eval(df,a.experiment_root,a.gt_root,a.output)
    print(f'\nAnalysis complete: {a.output.resolve()}')
if __name__=='__main__':main()
