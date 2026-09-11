"""Repeated, leakage-free MNIST evaluation for the sparse recovery configuration."""
from pathlib import Path
import json, time
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from latent_graph import RecoveryConfig, recover_graph
from experiment import load_mnist, grid_edges, adjacency, graph_metrics, train_frozen_gnn

PERM_SEEDS=(20260904,20260917,20261003)
PENALTIES=(.08,.10,.12)

def meanstd(x): return f'{np.mean(x):.4f} ± {np.std(x,ddof=1) if len(x)>1 else 0:.4f}'

def baselines(X, active, edge_count, rng):
    ids=np.flatnonzero(active); Z=X[:,ids]; Z=(Z-Z.mean(0))/(Z.std(0)+1e-8)
    C=np.abs(np.corrcoef(Z,rowvar=False)); iu=np.triu_indices(len(ids),1)
    take=np.argpartition(C[iu],-edge_count)[-edge_count:]
    corr=np.column_stack([ids[iu[0][take]],ids[iu[1][take]]]).astype(np.int32)
    choice=rng.choice(len(iu[0]),edge_count,replace=False)
    random=np.column_stack([ids[iu[0][choice]],ids[iu[1][choice]]]).astype(np.int32)
    return corr,random

def pdf_report(path, payload, figure):
    font='Helvetica'; fp=Path(r'C:\Windows\Fonts\arial.ttf')
    if fp.exists(): pdfmetrics.registerFont(TTFont('RU',str(fp))); font='RU'
    s=getSampleStyleSheet(); body=ParagraphStyle('b',parent=s['BodyText'],fontName=font,fontSize=9.5,leading=13)
    h=ParagraphStyle('h',parent=s['Heading2'],fontName=font,fontSize=14,leading=17)
    title=ParagraphStyle('t',parent=s['Title'],fontName=font,fontSize=18,leading=22)
    ts=TableStyle([('FONT',(0,0),(-1,-1),font),('GRID',(0,0),(-1,-1),.35,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('PADDING',(0,0),(-1,-1),4)])
    doc=SimpleDocTemplate(str(path),pagesize=A4,leftMargin=1.3*cm,rightMargin=1.3*cm,topMargin=1.2*cm,bottomMargin=1.2*cm)
    story=[Paragraph('Повторные эксперименты восстановления графа MNIST',title),Spacer(1,.25*cm),Paragraph(
        'Три независимые перестановки признаков. Recovery не получает координаты, метки, исходный граф или число его рёбер. Ridge α=0.01; проверены штрафы 0.08, 0.10 и 0.12. Для каждого штрафа граф строится заново. Выбор выполняется по независимой validation-ошибке: берётся самый разреженный вариант в пределах 1% от лучшей непенализированной ошибки.',body),Spacer(1,.25*cm),Image(str(figure),width=18*cm,height=7*cm),PageBreak(),Paragraph('Устойчивость восстановления',h)]
    chosen=payload['chosen_runs']; keys=['recovered_edges','precision','recall','edge_f1','recovered_components','recovered_largest_component']
    names={'recovered_edges':'Рёбра','precision':'Precision','recall':'Recall','edge_f1':'F1','recovered_components':'Компоненты','recovered_largest_component':'Крупнейшая компонента'}
    story += [Table([['Метрика','Среднее ± SD']]+[[names[k],meanstd([r['graph_metrics'][k] for r in chosen])] for k in keys],colWidths=[9*cm,7*cm],style=ts),Spacer(1,.3*cm)]
    story += [Paragraph('Чувствительность к штрафу',h)]
    rows=[['Penalty','Рёбра mean±SD','Validation loss mean±SD']]
    for pen in PENALTIES:
        q=[x for r in payload['all_recovery_runs'] for x in r['candidates'] if x['penalty']==pen]
        rows.append([str(pen),meanstd([x['edges'] for x in q]),meanstd([x['unpenalized_val_loss'] for x in q])])
    story += [Table(rows,colWidths=[4*cm,6*cm,7*cm],style=ts),Spacer(1,.3*cm),Paragraph('Frozen GNN и baselines',h),Paragraph(
        'Для каждой перестановки GCN и GraphSAGE обучаются только с oracle-решёткой. Затем веса замораживаются и без дообучения проверяются с recovered, correlation top-E, random degree-matched, неправильной решёткой и без рёбер.',body),Spacer(1,.2*cm)]
    conditions=sorted(chosen[0]['gnn'])
    rows=[['Модель / граф','Accuracy','Macro-F1']]
    for c in conditions:
        rows.append([c,meanstd([r['gnn'][c]['accuracy'] for r in chosen]),meanstd([r['gnn'][c]['macro_f1'] for r in chosen])])
    story += [Table(rows,colWidths=[8*cm,4.5*cm,4.5*cm],style=ts),Spacer(1,.3*cm),Paragraph('Вывод',h),Paragraph(
        'Цель — компактная семантически полезная структура, а не максимальный recall ценой тысяч лишних рёбер. Поэтому одновременно рассматриваются размер графа, F1 рёбер, устойчивость между перестановками и потеря качества frozen GNN. Oracle-метрики используются только после выбора конфигурации.',body)]
    if chosen[0].get('baseline_graph_metrics'):
        story += [Spacer(1,.25*cm),Paragraph('Структурные baseline',h)]
        rows=[['Метод','Precision','Recall','F1']]
        for name in ('Correlation top-E','Random edge-matched'):
            rows.append([name]+[meanstd([r['baseline_graph_metrics'][name][k] for r in chosen]) for k in ('precision','recall','edge_f1')])
        story += [Table(rows,colWidths=[6*cm,3.6*cm,3.6*cm,3.6*cm],style=ts)]
    if 'synthetic' in payload:
        story += [PageBreak(),Paragraph('Синтетический positive control',h),Paragraph(payload['synthetic']['summary'],body)]
    doc.build(story)

def main():
    out=Path('repeated_results'); out.mkdir(exist_ok=True)
    X,y,Xte,yte=load_mnist(r'C:\Users\taopi\Downloads\archive.zip'); X=X.astype(np.float32)/255; Xte=Xte.astype(np.float32)/255
    truth=grid_edges(); all_runs=[]; chosen_runs=[]; t0=time.perf_counter()
    for run_id,ps in enumerate(PERM_SEEDS):
        rng=np.random.default_rng(ps); perm=rng.permutation(784); inv=np.argsort(perm); Xp=X[:,perm]; Xtep=Xte[:,perm]
        candidates=[]
        for penalty in PENALTIES:
            cfg=RecoveryConfig(ridge_alpha=.01,edge_penalty=penalty,anneal_steps=2000,seed=42+run_id)
            A,m=recover_graph(Xp[:10000],Xp[10000:12000],cfg)
            raw=float(m.best_energy_-2*penalty*len(m.edges_))
            candidates.append(dict(penalty=penalty,edges=len(m.edges_),unpenalized_val_loss=raw,model=m,A=A))
        best_loss=min(c['unpenalized_val_loss'] for c in candidates); tol=.01*abs(best_loss)
        eligible=[c for c in candidates if c['unpenalized_val_loss']<=best_loss+tol]; chosen=min(eligible,key=lambda c:c['edges'])
        m,A=chosen['model'],chosen['A']; eo=np.array([(perm[a],perm[b]) for a,b in m.edges_],np.int32)
        gm=graph_metrics(eo,truth,m.active_mask_[inv],X[:10000].var(0)); corr,rand=baselines(Xp[:10000],m.active_mask_,len(m.edges_),rng)
        oracle=adjacency(784,truth)[np.ix_(perm,perm)]; graphs={'Oracle':oracle,'Recovered':A,'Correlation top-E':adjacency(784,corr),'Random degree-matched':adjacency(784,rand),'Wrong shuffled grid':adjacency(784,truth)}
        gnn={}
        for arch in ('GCN','GraphSAGE'):
            got=train_frozen_gnn(Xp[12000:22000],y[12000:22000],Xp[22000:24000],y[22000:24000],Xtep,yte,oracle,graphs,100+run_id,architecture=arch,epochs=6,batch=512)
            for name,v in got.items(): gnn[f'{arch} / {name}']=v
        serial_candidates=[{k:v for k,v in c.items() if k not in ('model','A')} for c in candidates]
        all_runs.append({'permutation_seed':ps,'candidates':serial_candidates,'chosen_penalty':chosen['penalty']})
        chosen_runs.append({'permutation_seed':ps,'chosen_penalty':chosen['penalty'],'graph_metrics':gm,'gnn':gnn})
        print('completed',ps,'penalty',chosen['penalty'],'edges',gm['recovered_edges'],'F1',gm['edge_f1'],flush=True)
    payload={'selection_rule':'sparsest graph within 1% of best unpenalized validation loss','all_recovery_runs':all_runs,'chosen_runs':chosen_runs,'runtime_seconds':time.perf_counter()-t0}
    (out/'repeated_metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(1,2,figsize=(10,4));
    for r in all_runs: ax[0].plot([c['penalty'] for c in r['candidates']],[c['edges'] for c in r['candidates']],marker='o',label=str(r['permutation_seed']))
    ax[0].set(xlabel='Штраф за степень',ylabel='Число рёбер',title='Разреженность'); ax[0].legend(fontsize=7)
    ax[1].bar(range(len(chosen_runs)),[r['graph_metrics']['edge_f1'] for r in chosen_runs],color='#f28e2b'); ax[1].set_xticks(range(len(chosen_runs)),[str(r['permutation_seed']) for r in chosen_runs],rotation=20); ax[1].set(ylabel='F1 рёбер',title='Устойчивость по перестановкам',ylim=(0,1)); fig.tight_layout()
    fig.savefig(out/'repeated_summary.png',dpi=200); plt.close(fig); pdf_report(out/'repeated_experiments_report_ru.pdf',payload,out/'repeated_summary.png')

if __name__=='__main__': main()
