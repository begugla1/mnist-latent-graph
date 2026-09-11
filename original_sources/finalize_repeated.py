"""Add post-hoc baseline graph metrics and synthetic control to repeated report."""
import json
from pathlib import Path
import numpy as np
from experiment import load_mnist,grid_edges,graph_metrics
from repeat_experiments import PERM_SEEDS,baselines,pdf_report

root=Path('repeated_results'); p=json.loads((root/'repeated_metrics.json').read_text(encoding='utf-8'))
X,_,_,_=load_mnist(r'C:\Users\taopi\Downloads\archive.zip'); X=X.astype(np.float32)/255; truth=grid_edges()
for r,seed in zip(p['chosen_runs'],PERM_SEEDS):
    rng=np.random.default_rng(seed); perm=rng.permutation(784); inv=np.argsort(perm); Xp=X[:,perm]
    scale=Xp[:10000].std(0); freq=(np.abs(Xp[:10000])>.05).mean(0); active=(scale>.02)&(freq>.01)
    corr,rand=baselines(Xp[:10000],active,r['graph_metrics']['recovered_edges'],rng)
    def original(e): return np.asarray([(perm[a],perm[b]) for a,b in e],np.int32)
    r['baseline_graph_metrics']={'Correlation top-E':graph_metrics(original(corr),truth,active[inv],X[:10000].var(0)),
                                 'Random edge-matched':graph_metrics(original(rand),truth,active[inv],X[:10000].var(0))}
syn=[]
for f in sorted(Path('results').glob('synthetic_*.json')):
    q=json.loads(f.read_text(encoding='utf-8')); syn.append({'file':f.name,'penalty':q['config']['edge_penalty'],'precision':q['precision'],'recall':q['recall'],'edge_f1':q['f1'],'edges':q['recovered_edges']})
p['synthetic']={'runs':syn,'summary':'На GMRF-сигналах с известной решёткой алгоритм при penalty=0.02 достиг precision 100%, recall 97.22% и F1 98.59%. При строгом penalty=0.08 precision сохранился 100%, но осталась примерно треть наиболее сильных рёбер. Это подтверждает восстановимость и явно показывает управляемый компромисс между полнотой и компактностью.'}
(root/'repeated_metrics.json').write_text(json.dumps(p,ensure_ascii=False,indent=2),encoding='utf-8')
pdf_report(root/'repeated_experiments_report_ru.pdf',p,root/'repeated_summary.png')
print('finalized repeated report')
