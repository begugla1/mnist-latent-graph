from __future__ import annotations
import argparse, gzip, json, struct, time, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import torch
from torch import nn
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import accuracy_score, f1_score, log_loss
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from latent_graph import RecoveryConfig, recover_graph


def idx(raw: bytes):
    magic, n = struct.unpack(">II", raw[:8])
    if magic == 2051:
        rows, cols = struct.unpack(">II", raw[8:16]); return np.frombuffer(raw, np.uint8, offset=16).reshape(n, rows*cols)
    if magic == 2049: return np.frombuffer(raw, np.uint8, offset=8)
    raise ValueError(f"Unknown IDX magic {magic}")


def load_mnist(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        def one(token):
            matches = [x for x in names if x.split('/')[-1] == token]
            # Some supplied archives contain tiny duplicate/link-like entries.
            chosen = max(matches, key=lambda x: z.getinfo(x).file_size)
            return idx(z.read(chosen))
        return one('train-images.idx3-ubyte'), one('train-labels.idx1-ubyte'), one('t10k-images.idx3-ubyte'), one('t10k-labels.idx1-ubyte')


def grid_edges(side=28):
    e=[]
    for r in range(side):
        for c in range(side):
            i=r*side+c
            if r+1<side:e.append((i,i+side))
            if c+1<side:e.append((i,i+1))
    return np.asarray(e,np.int32)


def adjacency(n, edges):
    A=np.zeros((n,n),np.float32)
    if len(edges): A[edges[:,0],edges[:,1]]=A[edges[:,1],edges[:,0]]=1
    return A


def graph_metrics(pred_edges, truth_edges, active, variances):
    active=set(np.flatnonzero(active).tolist())
    truth={tuple(sorted(map(int,e))) for e in truth_edges if int(e[0]) in active and int(e[1]) in active}
    pred={tuple(sorted(map(int,e))) for e in pred_edges}
    tp=len(truth&pred); fp=len(pred-truth); fn=len(truth-pred)
    precision=tp/max(len(pred),1); recall=tp/max(len(truth),1); f1=2*precision*recall/max(precision+recall,1e-12)
    denom=sum(np.sqrt(variances[a]*variances[b]) for a,b in truth)
    weighted=sum(np.sqrt(variances[a]*variances[b]) for a,b in truth&pred)/max(denom,1e-12)
    n=len(active); vertices_total=len(variances)
    Ap=adjacency(vertices_total,np.asarray(list(pred),dtype=np.int32).reshape(-1,2))
    At=adjacency(vertices_total,np.asarray(list(truth),dtype=np.int32).reshape(-1,2))
    cp,labp=connected_components(csr_matrix(Ap[np.ix_(sorted(active),sorted(active))]),directed=False)
    ct,labt=connected_components(csr_matrix(At[np.ix_(sorted(active),sorted(active))]),directed=False)
    largest_p=int(np.bincount(labp).max()) if len(labp) else 0
    largest_t=int(np.bincount(labt).max()) if len(labt) else 0
    deg=Ap.sum(0)
    return dict(vertices_total=vertices_total,original_grid_edges_full=len(truth_edges),active_vertices=n,
                truth_informative_edges=len(truth),truth_informative_components=int(ct),truth_largest_component=largest_t,
                recovered_edges=len(pred),recovered_average_degree=float(2*len(pred)/max(n,1)),recovered_max_degree=int(deg.max()),
                recovered_density=float(len(pred)/max(n*(n-1)/2,1)),recovered_components=int(cp),recovered_largest_component=largest_p,
                TP=tp,FP=fp,FN=fn,precision=float(precision),recall=float(recall),edge_f1=float(f1),informative_weighted_recall=float(weighted))


def norm_sparse(A, self_loops=True):
    A=A.copy(); np.fill_diagonal(A,1 if self_loops else 0)
    d=np.maximum(A.sum(1),1); A=A/np.sqrt(d[:,None]*d[None,:])
    r,c=np.nonzero(A); ind=torch.tensor(np.stack([r,c]),dtype=torch.long)
    return torch.sparse_coo_tensor(ind,torch.tensor(A[r,c]),A.shape).coalesce()


class TinyGCN(nn.Module):
    def __init__(self,n,hidden=12):
        super().__init__(); self.lin1=nn.Linear(1,hidden); self.lin2=nn.Linear(hidden,hidden); self.out=nn.Linear(hidden*n,10)
    def agg(self,A,h):
        b,n,c=h.shape
        return torch.sparse.mm(A, h.permute(1,0,2).reshape(n,b*c)).reshape(n,b,c).permute(1,0,2)
    def forward(self,x,A):
        h=torch.relu(self.lin1(x[:,:,None])); h=torch.relu(self.lin2(self.agg(A,h))); h=self.agg(A,h)
        # Fixed vertex identities are legitimate (all samples share them); no 2-D
        # coordinates or inverse permutation are supplied.
        return self.out(h.flatten(1))


class FrozenGraphSAGE(nn.Module):
    """Mean-aggregation GraphSAGE (Hamilton et al., 2017)."""
    def __init__(self,n,hidden=12):
        super().__init__(); self.lin1=nn.Linear(2,hidden); self.lin2=nn.Linear(hidden*2,hidden); self.out=nn.Linear(hidden*n,10)
    def mean_neigh(self,A,h):
        b,n,c=h.shape
        return torch.sparse.mm(A,h.permute(1,0,2).reshape(n,b*c)).reshape(n,b,c).permute(1,0,2)
    def forward(self,x,A):
        h0=x[:,:,None]; h=torch.relu(self.lin1(torch.cat([h0,self.mean_neigh(A,h0)],2)))
        h=torch.relu(self.lin2(torch.cat([h,self.mean_neigh(A,h)],2)))
        return self.out(h.flatten(1))


def train_frozen_gnn(Xtr,ytr,Xv,yv,Xte,yte,A_train,A_eval,seed,architecture='GCN',epochs=8,batch=256):
    torch.manual_seed(seed); np.random.seed(seed)
    model=TinyGCN(Xtr.shape[1]) if architecture=='GCN' else FrozenGraphSAGE(Xtr.shape[1])
    opt=torch.optim.AdamW(model.parameters(),lr=2e-3,weight_decay=1e-4)
    lossfn=nn.CrossEntropyLoss(); As=norm_sparse(A_train,self_loops=architecture=='GCN'); best=None; bestloss=np.inf
    for ep in range(epochs):
        model.train(); order=np.random.permutation(len(Xtr))
        for st in range(0,len(order),batch):
            q=order[st:st+batch]; xb=torch.from_numpy(Xtr[q]); yb=torch.from_numpy(ytr[q].astype(np.int64))
            opt.zero_grad(); loss=lossfn(model(xb,As),yb); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            logits=model(torch.from_numpy(Xv),As); vl=lossfn(logits,torch.from_numpy(yv.astype(np.int64))).item()
        if vl<bestloss: bestloss=vl; best={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(best); model.eval(); results={}
    # Weights stay frozen; only the adjacency used by message passing changes.
    with torch.no_grad():
        for name,A in A_eval.items():
            Ae=norm_sparse(A,self_loops=architecture=='GCN'); probs=[]
            for st in range(0,len(Xte),batch): probs.append(torch.softmax(model(torch.from_numpy(Xte[st:st+batch]),Ae),1).numpy())
            p=np.concatenate(probs); pred=p.argmax(1)
            results[name]=dict(accuracy=accuracy_score(yte,pred),macro_f1=f1_score(yte,pred,average='macro'),nll=log_loss(yte,p))
    return results


def draw_graphs(path, image, truth_edges, recovered_edges, active):
    xy=np.array([(i%28,27-i//28) for i in range(784)])
    fig,ax=plt.subplots(1,3,figsize=(15,5)); ax[0].imshow(image.reshape(28,28),cmap='gray'); ax[0].set_title('Пример изображения'); ax[0].axis('off')
    for a,title,edges in [(ax[1],'Исходная решётка (только оценка)',truth_edges),(ax[2],'Восстановленный граф',recovered_edges)]:
        a.add_collection(LineCollection([[xy[i],xy[j]] for i,j in edges],colors='#c62828',linewidths=.35,alpha=.48,rasterized=True))
        if a is ax[2]:
            a.scatter(xy[~active,0],xy[~active,1],s=7,c='#1976d2',zorder=2,label='пассивные')
            a.scatter(xy[active,0],xy[active,1],s=7,c='#f28e2b',zorder=3,label='активные'); a.legend(fontsize=8,loc='lower left')
        else:a.scatter(xy[:,0],xy[:,1],s=6,c='#f28e2b',zorder=2)
        a.set_title(title); a.set_aspect('equal'); a.axis('off')
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)


def make_pdf(path, figpath, gm, runs, cfg, meta):
    font='Helvetica'
    for fp in [r'C:\Windows\Fonts\arial.ttf',r'C:\Windows\Fonts\DejaVuSans.ttf']:
        if Path(fp).exists(): pdfmetrics.registerFont(TTFont('RU',fp)); font='RU'; break
    styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='RUbody',parent=styles['BodyText'],fontName=font,fontSize=10,leading=14))
    styles.add(ParagraphStyle(name='RUtitle',parent=styles['Title'],fontName=font,fontSize=18,leading=22,alignment=TA_CENTER))
    styles.add(ParagraphStyle(name='RUheading',parent=styles['Heading2'],fontName=font,fontSize=14,leading=17))
    body,title,heading=styles['RUbody'],styles['RUtitle'],styles['RUheading']; doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=1.5*cm,leftMargin=1.5*cm,topMargin=1.4*cm,bottomMargin=1.4*cm)
    story=[Paragraph('Восстановление скрытого графа MNIST и проверка GNN',title),Spacer(1,.4*cm),Paragraph(
        'Алгоритм получил только матрицу яркостей с одной неизвестной перестановкой столбцов. Координаты, исходная решётка, метки классов и истинное число рёбер при восстановлении не использовались.',body),Spacer(1,.3*cm)]
    story += [Paragraph('Алгоритм',heading),Paragraph(
        '1. По train выбираются изменяющиеся вершины. 2. Для всех активных пар вычисляется устойчивый корреляционный score: |ρ|·|mean(sign ρᵦ)|/(1+std(|ρᵦ|)). 3. Для каждой вершины сохраняются top-K кандидатов. 4. Lasso с BIC строит начальный граф. 5. Ridge оценивает соседство; энергия равна сумме log(MSE) и штрафа за степень. 6. Отжиг переключает ADD/DELETE рёбра; ухудшение принимается с вероятностью exp(−ΔE/T), история градиента используется только для ранжирования.',body),Spacer(1,.25*cm)]
    story += [Table([['Параметр','Значение'],['active',str(gm['active_vertices'])],['Ridge α',str(cfg.ridge_alpha)],['штраф за степень',str(cfg.edge_penalty)],['цена одного ребра',str(2*cfg.edge_penalty)],['top-K',str(cfg.top_k)],['полных пар',str(meta['screened_pairs'])],['проверено поиском',str(meta['tested_pairs'])],['удалено pruning',str(meta['pruned_edges'])],['шагов отжига',str(cfg.anneal_steps)]],colWidths=[7*cm,7*cm],style=TableStyle([('FONT',(0,0),(-1,-1),font),('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('PADDING',(0,0),(-1,-1),5)])),Spacer(1,.3*cm),Image(str(figpath),width=18*cm,height=6*cm),Paragraph('Координатная раскладка справа применяется только после восстановления для оценки; сам алгоритм координат не видел.',body),PageBreak()]
    story += [Paragraph('Метрики восстановления информативной подрешётки',heading)]
    labels={'vertices_total':'Вершин всего','original_grid_edges_full':'Рёбер исходной решётки','active_vertices':'Активных вершин',
            'truth_informative_edges':'Исходных рёбер между активными','truth_informative_components':'Компонент в активной исходной части',
            'truth_largest_component':'Крупнейшая исходная компонента','recovered_edges':'Рёбер восстановленного графа',
            'recovered_average_degree':'Средняя степень восстановленного','recovered_max_degree':'Максимальная степень восстановленного',
            'recovered_density':'Плотность восстановленного','recovered_components':'Компонент восстановленного',
            'recovered_largest_component':'Крупнейшая восстановленная компонента','TP':'Верных рёбер (TP)','FP':'Лишних рёбер (FP)',
            'FN':'Пропущенных рёбер (FN)','precision':'Precision','recall':'Recall','edge_f1':'F1 рёбер',
            'informative_weighted_recall':'Взвешенный recall'}
    rows=[['Характеристика графа','Значение']]+[[labels.get(k,k),f'{v:.4f}' if isinstance(v,float) else str(v)] for k,v in gm.items()]
    story += [Table(rows,colWidths=[9*cm,5*cm],style=TableStyle([('FONT',(0,0),(-1,-1),font),('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('PADDING',(0,0),(-1,-1),4)])),Spacer(1,.4*cm),Paragraph('Frozen GNN: обучение только на исходной решётке',heading),Paragraph('Проверены две опубликованные архитектуры: GCN (Kipf & Welling, 2017) и mean-GraphSAGE (Hamilton et al., 2017). Для каждого seed модель обучается и выбирается по validation только с истинной решёткой. Затем веса полностью замораживаются. На одном и том же test наборе меняется только матрица смежности; оптимизатор и дообучение больше не запускаются. Координаты в признаки модели не входят.',body),Spacer(1,.2*cm)]
    rows=[['Модель / граф','Accuracy','Macro-F1','NLL']]
    for name,vals in runs.items():
        rows.append([name]+[f"{np.mean([x[k] for x in vals]):.4f} ± {np.std([x[k] for x in vals]):.4f}" for k in ['accuracy','macro_f1','nll']])
    story += [Table(rows,colWidths=[5*cm,4*cm,4*cm,4*cm],style=TableStyle([('FONT',(0,0),(-1,-1),font),('GRID',(0,0),(-1,-1),.4,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.lightgrey),('PADDING',(0,0),(-1,-1),4)])),Spacer(1,.35*cm),Paragraph(
        'Интерпретация: качество классификации показывает полезность найденной структуры, но само по себе не доказывает восстановление рёбер. Полная решётка MNIST неидентифицируема по постоянным углам; поэтому основные precision/recall/F1 рассчитаны по истинным рёбрам, у которых оба конца активны.',body)]
    doc.build(story)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--zip',default=r'C:\Users\taopi\Downloads\archive.zip'); ap.add_argument('--output',default='results'); ap.add_argument('--quick',action='store_true'); args=ap.parse_args()
    out=Path(args.output); out.mkdir(exist_ok=True); X,y,Xtest,ytest=load_mnist(args.zip); X=X.astype(np.float32)/255; Xtest=Xtest.astype(np.float32)/255
    rng=np.random.default_rng(20260904); perm=rng.permutation(784); Xp=X[:,perm]; Xtep=Xtest[:,perm]
    ng=2500 if args.quick else 10000; nv_graph=700 if args.quick else 2000
    cfg=RecoveryConfig(ridge_alpha=.01,edge_penalty=.08,anneal_steps=200 if args.quick else 2000,proposal_batch=6 if args.quick else 12,top_k=16 if args.quick else 24)
    Arec,model=recover_graph(Xp[:ng],Xp[ng:ng+nv_graph],cfg)
    inv=np.argsort(perm); rec_original=np.array([(perm[a],perm[b]) for a,b in model.edges_],np.int32)
    truth=grid_edges(); gm=graph_metrics(rec_original,truth,model.active_mask_[inv],X[:ng].var(0))
    # Every classifier sees shuffled features. Oracle adjacency is permuted consistently.
    oracle=adjacency(784,truth)[np.ix_(perm,perm)]; wrong=adjacency(784,truth)
    cstart=ng+nv_graph; ntrain=3000 if args.quick else 10000; nval=1000 if args.quick else 2000; ntest=2000 if args.quick else 10000
    tr=slice(cstart,cstart+ntrain); va=slice(cstart+ntrain,cstart+ntrain+nval)
    graphs={'Oracle grid':oracle,'Recovered':Arec,'Wrong shuffled grid':wrong}; seeds=[0] if args.quick else [0,1]
    architectures=['GCN'] if args.quick else ['GCN','GraphSAGE']; runs={f'{arch} / {name}':[] for arch in architectures for name in graphs}; t=time.perf_counter()
    for arch in architectures:
        for seed in seeds:
            frozen=train_frozen_gnn(Xp[tr],y[tr],Xp[va],y[va],Xtep[:ntest],ytest[:ntest],oracle,graphs,seed,architecture=arch,epochs=2 if args.quick else 6,batch=256 if args.quick else 512)
            for name,value in frozen.items(): runs[f'{arch} / {name}'].append(value)
    fig=out/'graphs.png'; draw_graphs(fig,Xtest[0],truth,rec_original,model.active_mask_[inv])
    meta=dict(screened_pairs=model.screened_pair_count_,tested_pairs=model.unique_tested_pairs_,pool_size=model.pool_size_,pruned_edges=model.pruned_edges_,effective_edge_penalty=model.edge_penalty_,recovery_seconds=model.fit_seconds_,gnn_seconds=time.perf_counter()-t,best_energy=model.best_energy_,acceptance_rate=model.acceptance_rate_)
    payload={'graph_metrics':gm,'gnn':runs,'recovery_config':model.metadata_,'runtime':meta,'permutation_seed':20260904,'protocol':{'gnn_architectures':['GCN (Kipf & Welling, 2017)','mean-GraphSAGE (Hamilton et al., 2017)'],'gnn_protocol':'train once on oracle grid; freeze weights; evaluate unchanged weights with each adjacency','graph_train':ng,'graph_validation':nv_graph,'classifier_train':ntrain,'classifier_validation':nval,'test':ntest}}
    (out/'metrics.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'recovered_graph.npz',adjacency=Arec,edges_shuffled=model.edges_,active_mask_shuffled=model.active_mask_,permutation=perm)
    make_pdf(out/'mnist_latent_graph_report_ru.pdf',fig,gm,runs,cfg,meta); print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
