from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
def run_eval(predictions_save_dest):
    all_lines = open(predictions_save_dest).readlines()
    labs = [int(line.rstrip().split(',')[2]) for line in all_lines]
    preds = [float(line.split(',')[1]) for line in all_lines]

    fmax = 0
    for t in range(1,100):
        thr = t/100
        f1score = f1(preds,labs,thr)
        if f1score > fmax:
            fmax = f1score

    print(f'# of negatives: {len(labs) - sum(labs)}')
    print(f'# of positives: {sum(labs)}')
    print(f'-Fmax score: {fmax}')
    print(f'-F1 score: {f1_score(labs,[1 if p >= 0.5 else 0 for p in preds])}')
    print(f'-auROC: {roc_auc_score(labs,preds)}')
    print(f'-auPRC: {average_precision_score(labs,preds)}')


def f1(preds,labs,thr):
    tp,fp,fn = 0,0,0
    for p,l in zip(preds,labs):
        if p >= thr and l == 1:
            tp += 1
        elif p < thr and l == 1:
            fn += 1
        elif p >= thr and l == 0:
            fp += 1
    print(f'Threshold {thr:1.2f}: TP {tp: >3d} FN {fn: >3d} TN {len(labs)-tp-fp-fn: >5d} FP {fp: >5d}')
    return 2*tp/(2*tp+fp+fn)

# if sys.argv[0] == 'Evaluation.py':
if __name__ == '__main__':
    import sys
    run_eval(sys.argv[1])

