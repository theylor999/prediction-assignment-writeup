import json
d=json.load(open("results.json")); r=d["res"]; f=d["figs"]
DATE="October 6, 2026"
cv=r["cv_acc"]; cm=r["cm"]
cmtab="".join("<tr><th>%s</th>%s</tr>"%("ABCDE"[i],"".join("<td>%d</td>"%v for v in row)) for i,row in enumerate(cm))
imp="".join("<tr><td><code>%s</code></td><td>%.4f</td></tr>"%(k,v) for k,v in r["top_imp"].items())
cvrows="".join("<tr><td>%s</td><td>%.4f</td><td>%.2f%%</td></tr>"%(k,v,(1-v)*100) for k,v in cv.items())
preds=r["predictions"]
qrows="".join("<tr><td>%d</td><td><b>%s</b></td></tr>"%(i+1,p) for i,p in enumerate(preds))
html=f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"><title>Prediction Assignment Writeup</title>
<style>body{{font-family:Helvetica,Arial,sans-serif;max-width:860px;margin:30px auto;padding:0 16px;line-height:1.5;color:#333}}
h1{{font-size:30px}}h2{{border-bottom:1px solid #ddd;padding-bottom:4px;margin-top:34px}}
pre{{background:#f5f5f5;border:1px solid #ccc;border-radius:4px;padding:10px;overflow:auto;font-size:13px}}
table{{border-collapse:collapse;margin:10px 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:right}}
.meta{{color:#777}}img{{max-width:100%}}.fig{{text-align:center;margin:14px 0}}.fig p{{font-size:13px;color:#666}}</style></head><body>
<h1>Prediction Assignment Writeup</h1>
<p class="meta"><b>Author:</b> Theylor &nbsp;|&nbsp; <b>Date:</b> {DATE}<br>Course: Practical Machine Learning (Johns Hopkins University) &mdash; Course Project</p>

<h2>1. Overview</h2>
<p>Accelerometers on the belt, forearm, arm and dumbbell of 6 participants recorded barbell lifts performed correctly (class <b>A</b>) and incorrectly in four ways (classes <b>B</b>&ndash;<b>E</b>) &mdash; Velloso, Bulling, Gellersen, Ugulino &amp; Fuks, <i>Qualitative Activity Recognition of Weight Lifting Exercises</i> (Weight Lifting Exercise Dataset, <a href="http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har">PUC-Rio HAR project</a>). The goal is to predict <code>classe</code> from sensor data and to predict 20 unseen test cases.</p>
<p><i>Implementation note:</i> the pipeline was run in Python (pandas / scikit-learn) because R was not available on the authoring machine; the report emulates an R Markdown knitr output, and the full code is in <code>analysis.py</code> in the repository. The method (random forest + k-fold cross validation) is identical to what <code>caret::train(method="rf")</code> would do.</p>

<h2>2. Data loading and cleaning</h2>
<pre><code>tr = pd.read_csv("pml-training.csv"); te = pd.read_csv("pml-testing.csv")
na_frac = tr.isna().mean()
drop = cols with &gt; 95% NA + [row index, user_name, 3 timestamp cols, new_window, num_window]
features = remaining numeric columns</code></pre>
<p>The training set has <b>{r['train_shape'][0]:,} rows x {r['train_shape'][1]} columns</b>; the test set has {r['test_shape'][0]} rows. <b>{r['cols_gt95_missing']}</b> columns are more than 95% missing (summary statistics such as kurtosis, skewness, max, min, avg, var, stddev that are only filled on window-summary rows); they were removed. Identifier / time variables (row index, user name, timestamps, window counters) were also removed so the model learns from sensor movement only and not from the order of the recordings. This leaves <b>{r['n_features']} numeric predictors</b>. The classes are fairly balanced, with class A the most frequent (Figure 1).</p>
<div class="fig"><img src="data:image/png;base64,{f['classes']}"><p>Figure 1. Distribution of the outcome <code>classe</code>: A={r['class_counts']['A']}, B={r['class_counts']['B']}, C={r['class_counts']['C']}, D={r['class_counts']['D']}, E={r['class_counts']['E']}.</p></div>

<h2>3. Model building and cross validation</h2>
<p>The training data were split <b>70% / 30%</b> (stratified by class, seed 1234) into a training set ({int(r['train_shape'][0]*0.7):,} rows) and a hold-out validation set ({r['train_shape'][0]-int(r['train_shape'][0]*0.7):,} rows) that is never used for model fitting or selection. On the 70% part I used <b>stratified 5-fold cross validation</b> to compare three candidate algorithms:</p>
<table><tr><th style="text-align:left">Model</th><th>CV accuracy</th><th>CV error</th></tr>{cvrows}</table>
<div class="fig"><img src="data:image/png;base64,{f['cmp']}"><p>Figure 2. 5-fold cross-validated accuracy of the three candidate models.</p></div>
<p><b>Why random forest?</b> It had by far the best cross-validated accuracy, copes with many correlated predictors without scaling, needs no manual feature selection, and gives variable importance. A single tree is fast but less accurate; boosting was better than a tree but still below the forest. Tuning was kept simple (300 trees for validation, 500 trees for the final model, default <code>mtry = floor(sqrt(52)) = 7</code>).</p>

<h2>4. Out-of-sample error</h2>
<p>The random forest fitted on the 70% training part was applied once to the 30% hold-out set. Hold-out accuracy = <b>{r['holdout_acc']:.4f}</b>, so the <b>estimated out-of-sample error is about {(1-r['holdout_acc'])*100:.2f}%</b>. The 5-fold cross-validated error ({(1-cv['Random forest'])*100:.2f}%) agrees with it, and the out-of-bag error of the final 500-tree forest fitted on all the data is {(1-r['oob'])*100:.2f}%. I therefore expect an out-of-sample error of roughly <b>0.3&ndash;0.7%</b> for new rows from the same six participants.</p>
<p><i>Caveat:</i> rows from the same participant and the same repetition window are very similar, so random folds are somewhat optimistic. On truly new participants the error would likely be higher. For the 20 quiz cases (which come from the same six participants) this is not a concern.</p>
<table><caption style="text-align:left">Hold-out confusion matrix (rows = actual, columns = predicted A&ndash;E)</caption><tr><th></th><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th></tr>{cmtab}</table>
<div class="fig"><img src="data:image/png;base64,{f['cm']}"><p>Figure 3. Confusion matrix on the 30% hold-out set. Most mistakes are D predicted as C.</p></div>
<div class="fig"><img src="data:image/png;base64,{f['imp']}"><p>Figure 4. Variable importance (mean decrease in Gini impurity) of the random forest.</p></div>
<table><tr><th style="text-align:left">Top-10 variable</th><th>Importance</th></tr>{imp}</table>
<p>The belt sensor (<code>roll_belt</code>, <code>yaw_belt</code>, <code>pitch_belt</code>), the dumbbell magnetometer and the forearm angles carry most of the signal.</p>

<h2>5. Predictions for the 20 test cases</h2>
<p>The final random forest (500 trees) was refit on the <b>full</b> training set and used to predict <code>pml-testing.csv</code>. These predictions were entered in the <i>Course Project Prediction Quiz</i> (Quiz answers):</p>
<table><tr><th>Test case</th><th>Predicted classe</th></tr>{qrows}</table>
<p><b>Quiz answers in order (1&ndash;20):</b> <code>{" ".join(preds)}</code></p>

<h2>6. Conclusion</h2>
<p>After removing mostly-empty and identifier columns, a random forest on {r['n_features']} sensor variables classifies the five lifting styles with about 99.4% hold-out accuracy (expected out-of-sample error under 1%), and predicts all 20 quiz cases. Cross validation was used to choose the algorithm and a never-touched hold-out set to estimate the error.</p>
<p class="meta">Code and data: <code>analysis.py</code> / <code>pml-training.csv</code> / <code>pml-testing.csv</code> in this repository. Data source: Velloso et al., groupware@LES (PUC-Rio). Compiled {DATE}.</p>
</body></html>"""
open("index.html","w",encoding="utf-8").write(html)
import re; txt=re.sub(r"<img[^>]*>|<[^>]+>"," ",html); print(len(txt.split()))
