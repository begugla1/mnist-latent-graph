from pathlib import Path
import json
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from experiment import load_mnist,grid_edges,graph_metrics
from repeat_experiments import baselines

ROOT=Path('repeated_results')

def ms(xs,d=3): return f'{np.mean(xs):.{d}f} ± {np.std(xs,ddof=1):.{d}f}'

def comparison_figure(path):
    X,_,Xt,_=load_mnist(r'C:\Users\taopi\Downloads\archive.zip'); X=X.astype(np.float32)/255
    z=np.load('results/recovered_graph.npz'); perm=z['permutation']; inv=np.argsort(perm); active=z['active_mask_shuffled']; rec=z['edges_shuffled']
    rng=np.random.default_rng(20260904); check=rng.permutation(784); assert np.array_equal(check,perm)
    corr,rand=baselines(X[:10000,perm],active,len(rec),rng)
    def orig(e): return np.asarray([(perm[a],perm[b]) for a,b in e],np.int32)
    rec,corr,rand=orig(rec),orig(corr),orig(rand); truth=grid_edges(); active_o=active[inv]
    xy=np.array([(i%28,27-i//28) for i in range(784)])
    fig,ax=plt.subplots(2,3,figsize=(15,9)); ax=ax.ravel()
    ax[0].imshow(Xt[0].reshape(28,28),cmap='gray'); ax[0].set_title('Пример MNIST'); ax[0].axis('off')
    panels=[('Oracle: полная решётка',truth),('Наш алгоритм: E определяется автоматически',rec),
            ('Correlation top-E: E получено от нашего метода',corr),('Random edge-matched: тот же E',rand)]
    for a,(title,edges) in zip(ax[1:5],panels):
        a.add_collection(LineCollection([[xy[i],xy[j]] for i,j in edges],colors='#c62828',linewidths=.32,alpha=.42,rasterized=True))
        a.scatter(xy[~active_o,0],xy[~active_o,1],s=5,c='#1976d2',zorder=2)
        a.scatter(xy[active_o,0],xy[active_o,1],s=5,c='#f28e2b',zorder=3)
        a.set_title(title,fontsize=10); a.set_aspect('equal'); a.axis('off')
    # Error overlay for proposed graph.
    a=ax[5]; T={tuple(sorted(x)) for x in truth if active_o[x[0]] and active_o[x[1]]}; P={tuple(sorted(x)) for x in rec}
    for edges,color,label,lw,alpha in [(T-P,'#bdbdbd','FN',.45,.5),(P-T,'#7b1fa2','дополнительные',.45,.55),(P&T,'#c62828','TP',.65,.7)]:
        a.add_collection(LineCollection([[xy[i],xy[j]] for i,j in edges],colors=color,linewidths=lw,alpha=alpha,label=label,rasterized=True))
    a.scatter(xy[active_o,0],xy[active_o,1],s=4,c='#f28e2b'); a.set_title('Наш граф: TP / дополнительные / FN',fontsize=10); a.set_aspect('equal'); a.axis('off')
    a.legend(handles=[plt.Line2D([0],[0],color='#c62828',label='TP'),plt.Line2D([0],[0],color='#7b1fa2',label='дополнительные'),plt.Line2D([0],[0],color='#bdbdbd',label='FN')],fontsize=7,loc='lower left')
    fig.tight_layout(); fig.savefig(path,dpi=220,bbox_inches='tight'); plt.close(fig)

def build():
    p=json.loads((ROOT/'repeated_metrics.json').read_text(encoding='utf-8')); runs=p['chosen_runs']; fig=ROOT/'graph_approaches_comparison.png'; comparison_figure(fig)
    font='Helvetica'; fp=Path(r'C:\Windows\Fonts\arial.ttf')
    if fp.exists(): pdfmetrics.registerFont(TTFont('RU',str(fp))); font='RU'
    ss=getSampleStyleSheet(); body=ParagraphStyle('bodyru',parent=ss['BodyText'],fontName=font,fontSize=10,leading=14,spaceAfter=6)
    small=ParagraphStyle('smallru',parent=body,fontSize=8.2,leading=10.5); h=ParagraphStyle('hru',parent=ss['Heading2'],fontName=font,fontSize=15,leading=18,spaceAfter=8)
    title=ParagraphStyle('tru',parent=ss['Title'],fontName=font,fontSize=20,leading=24)
    style=TableStyle([('FONT',(0,0),(-1,-1),font),('FONTSIZE',(0,0),(-1,-1),8.2),('GRID',(0,0),(-1,-1),.35,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8e8e8')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('PADDING',(0,0),(-1,-1),4)])
    doc=SimpleDocTemplate(str(ROOT/'repeated_experiments_report_ru.pdf'),pagesize=landscape(A4),leftMargin=1.2*cm,rightMargin=1.2*cm,topMargin=1.1*cm,bottomMargin=1.1*cm)
    story=[Paragraph('Повторные эксперименты: активная часть графа и оценка числа рёбер',title),Spacer(1,.25*cm),Paragraph(
      '<b>Главный отдельный результат.</b> Алгоритм без координат, исходной структуры и известного числа рёбер стабильно определяет 487 информативных вершин и бюджет примерно 1037 активных рёбер. После оценки active-множества и E возможны два режима: использовать рёбра полного алгоритма либо выбрать top-E пар по устойчивой корреляции. Второй вариант лучше копирует геометрическую решётку; первый соответствует условным предсказательным зависимостям.',body),Paragraph(
      '<b>Результат полного алгоритма.</b> В найденном графе около 72% геометрически правильных связей, восстановлено около 81% идентифицируемой активной решётки. Frozen GCN теряет в среднем 0.51 п.п., GraphSAGE — 1.16 п.п. Это свидетельство устойчивого компактного графа, но не доказательство превосходства одного способа выбора рёбер над другим.',body),Paragraph(
      '<b>Что дали повторы.</b> Одна удачная перестановка могла быть случайностью или ошибкой индексации. Три независимые перестановки дали 1046, 1035 и 1030 рёбер и близкий F1. Следовательно, результат практически инвариантен к именам столбцов и не зависит от конкретного перемешивания.',body),Spacer(1,.2*cm)]
    rows=[['Run','Permutation','Penalty','Рёбра','Precision','Recall','F1','Компоненты','Frozen Δ GCN','Frozen Δ SAGE']]
    for i,r in enumerate(runs,1):
        g=r['graph_metrics']; dg=r['gnn']['GCN / Recovered']['accuracy']-r['gnn']['GCN / Oracle']['accuracy']; ds=r['gnn']['GraphSAGE / Recovered']['accuracy']-r['gnn']['GraphSAGE / Oracle']['accuracy']
        rows.append([i,r['permutation_seed'],r['chosen_penalty'],g['recovered_edges'],f"{g['precision']:.3f}",f"{g['recall']:.3f}",f"{g['edge_f1']:.3f}",g['recovered_components'],f'{100*dg:+.2f} п.п.',f'{100*ds:+.2f} п.п.'])
    story += [Table(rows,colWidths=[1.2*cm,2.4*cm,1.5*cm,1.5*cm,1.7*cm,1.5*cm,1.3*cm,1.8*cm,2.3*cm,2.3*cm],style=style),PageBreak()]

    story += [Paragraph('Подходы и условия честного сравнения',h)]
    approaches=[
      ('Наш алгоритм','Сам определяет active mask, candidate pool и итоговое E. Stable correlation используется только для screening; Lasso/BIC даёт старт; Ridge validation-energy, штраф, pruning и annealing решают, какие рёбра оставить. Ни координаты, ни E истины не известны.'),
      ('Correlation top-E','Берёт E пар с максимальной устойчивой корреляцией среди active-вершин. Active-множество и E передаются из нашего алгоритма. Поэтому это не конкурент всей процедуре, а второй способ построить рёбра после того, как наш метод решил задачи фильтрации вершин и оценки размера графа.'),
      ('Random edge-matched','Случайно выбирает столько же рёбер среди active-вершин. Показывает, что результат нельзя объяснить одной плотностью. Это слабый контроль; он также получает готовый E.'),
      ('Wrong shuffled grid','Накладывает решётку на уже перемешанные индексы. Число рёбер фиксировано равным 1512, но семантическая привязка неправильна. Нужен для проверки чувствительности frozen GNN к topology.'),
      ('Oracle grid','Истинная решётка из 1512 рёбер. Используется только для обучения frozen GNN и post-hoc оценки, никогда — внутри восстановления или выбора penalty.')]
    data=[['Подход','Что делает и что ему известно']]+[[a,b] for a,b in approaches]
    story += [Table(data,colWidths=[5.2*cm,20*cm],style=style),Spacer(1,.25*cm),Paragraph(
      '<b>Почему correlation top-E имеет более высокий geometric F1?</b> В MNIST соседние пиксели обычно сильно коррелируют, поэтому сортировка |ρ| хорошо воспроизводит локальную решётку. Бюджет E при этом оценивает наш метод. Полный алгоритм оптимизирует условное предсказание: если связь избыточна после учёта других соседей, Ridge/pruning может удалить её; одновременно может остаться дальняя связь частей одного штриха. Практический вывод: оценку active и E можно считать самостоятельным результатом, после которого способ выбора конкретных рёбер определяется целью.',body),PageBreak()]

    story += [Paragraph('Как выглядят графы',h),Image(str(fig),width=25.7*cm,height=15.4*cm),Paragraph(
      'Оранжевые вершины активны, синие пассивны. Координаты применены только после восстановления для визуальной оценки. У correlation-графа заметно доминируют короткие локальные связи; наш граф разрежен, но содержит дополнительные диагональные и дальние статистические зависимости.',small),PageBreak()]

    story += [Paragraph('Численные сравнения',h)]
    names=['Proposed','Correlation top-E','Random edge-matched']; rows=[['Структура','Рёбра','Precision','Recall','F1']]
    rows.append(['Proposed',ms([r['graph_metrics']['recovered_edges'] for r in runs],1),ms([r['graph_metrics']['precision'] for r in runs]),ms([r['graph_metrics']['recall'] for r in runs]),ms([r['graph_metrics']['edge_f1'] for r in runs])])
    for n in names[1:]:
        rows.append([n,ms([r['baseline_graph_metrics'][n]['recovered_edges'] for r in runs],1),ms([r['baseline_graph_metrics'][n]['precision'] for r in runs]),ms([r['baseline_graph_metrics'][n]['recall'] for r in runs]),ms([r['baseline_graph_metrics'][n]['edge_f1'] for r in runs])])
    story += [KeepTogether([Paragraph('Структурное совпадение с активной решёткой',h),Table(rows,colWidths=[5*cm,4*cm,4*cm,4*cm,4*cm],style=style)]),Spacer(1,.35*cm)]
    rows=[['Модель / структура','Accuracy mean ± SD','Δ к своему frozen oracle']]
    for arch in ('GCN','GraphSAGE'):
        oracle=[r['gnn'][f'{arch} / Oracle']['accuracy'] for r in runs]
        for cond in ('Oracle','Recovered','Correlation top-E','Random degree-matched','Wrong shuffled grid'):
            vals=[r['gnn'][f'{arch} / {cond}']['accuracy'] for r in runs]; delta=[v-o for v,o in zip(vals,oracle)]
            rows.append([f'{arch} / {cond}',ms(vals),f'{100*np.mean(delta):+.2f} ± {100*np.std(delta,ddof=1):.2f} п.п.'])
    story += [KeepTogether([Paragraph('Frozen GNN: одни веса, меняется только adjacency',h),Table(rows,colWidths=[8*cm,6*cm,7*cm],style=style)]),PageBreak()]

    story += [Paragraph('Penalty, validation loss и итоговый вердикт',h)]
    rows=[['Penalty','Рёбра mean ± SD','Непенализированный validation loss']]
    for pen in (.08,.10,.12):
        q=[x for rr in p['all_recovery_runs'] for x in rr['candidates'] if x['penalty']==pen]
        rows.append([pen,ms([x['edges'] for x in q],1),ms([x['unpenalized_val_loss'] for x in q],2)])
    story += [Table(rows,colWidths=[4*cm,6*cm,8*cm],style=style),Spacer(1,.3*cm),Paragraph(
      '<b>Что такое validation loss.</b> Ridge обучается на одной части изображений, а затем предсказывает значения вершин на отдельных validation-изображениях. Validation loss — суммарная ошибка этих предсказаний, не использованных при обучении коэффициентов. Чем она меньше, тем лучше граф переносит найденные зависимости на новые изображения. В таблице значения отрицательные из-за логарифма MSE: более отрицательное число лучше, например −1072 лучше, чем −1004.',body),Paragraph(
      '<b>Зачем одновременно нужен penalty.</b> Если минимизировать только validation loss, добавление рёбер часто понемногу улучшает предсказание и граф разрастается. Penalty назначает каждому ребру цену. Поэтому выбирается не абсолютный минимум числа рёбер, а самый компактный граф, validation loss которого остаётся почти таким же хорошим. Штрафы 0.10 и 0.12 уменьшили граф до 879 и 735 рёбер, но заметно ухудшили prediction. Во всех трёх повторах правило «самый разреженный в пределах 1% от лучшего validation loss» выбрало 0.08.',body),Paragraph(
      '<b>Вердикт.</b> Надёжно подтверждён отдельный результат: метод самостоятельно и устойчиво оценивает информативную область и разумное число активных рёбер — около 1037, не зная, что в активной исходной решётке 924 ребра. Далее доступны два осмысленных выхода. Полный Ridge/pruning/annealing граф лучше интерпретировать как граф условных зависимостей. Гибрид active + estimated E + correlation top-E лучше подходит, если цель — именно восстановление локальной геометрии. Это не поражение алгоритма: top-E использует две ключевые оценки, полученные нашим методом. Но в работе нужно честно разделять вклад оценки размера графа и вклад способа ранжирования конкретных рёбер.',body)]
    doc.build(story); print('wrote',ROOT/'repeated_experiments_report_ru.pdf')

if __name__=='__main__': build()
