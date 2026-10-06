import pandas as pd, numpy as np, json, io, base64
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
tr = pd.read_csv("pml-training.csv", low_memory=False); te = pd.read_csv("pml-testing.csv")
res = {"train_shape": tr.shape, "test_shape": te.shape}
na_frac = tr.isna().mean(); res["cols_gt95_missing"] = int((na_frac>0.95).sum())
drop = list(na_frac[na_frac>0.95].index) + ["Unnamed: 0","user_name","raw_timestamp_part_1","raw_timestamp_part_2","cvtd_timestamp","new_window","num_window"]
feat = [c for c in tr.columns if c not in drop and c!="classe"]
# also drop non-numeric
feat = [c for c in feat if pd.api.types.is_numeric_dtype(tr[c])]
res["n_features"]=len(feat)
X=tr[feat]; y=tr["classe"]
res["class_counts"]=y.value_counts().sort_index().to_dict()
Xtr,Xva,ytr,yva = train_test_split(X,y,test_size=0.3,stratify=y,random_state=1234)
models = {"Decision tree (CART)":DecisionTreeClassifier(random_state=1),
          "Gradient boosting":GradientBoostingClassifier(n_estimators=100,random_state=1),
          "Random forest":RandomForestClassifier(n_estimators=300,n_jobs=-1,random_state=1)}
skf=StratifiedKFold(5,shuffle=True,random_state=1234)
cv={}
for n,m in models.items():
    p=cross_val_predict(m,Xtr,ytr,cv=skf,n_jobs=1 if n=="Random forest" else 1)
    cv[n]=accuracy_score(ytr,p); print(n,cv[n],flush=True)
res["cv_acc"]=cv
rf=models["Random forest"].fit(Xtr,ytr)
pv=rf.predict(Xva); res["holdout_acc"]=accuracy_score(yva,pv)
cm=confusion_matrix(yva,pv,labels=list("ABCDE")); res["cm"]=cm.tolist()
imp=pd.Series(rf.feature_importances_,index=feat).sort_values(ascending=False)
res["top_imp"]=imp.head(10).round(4).to_dict()
def b64(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=110,bbox_inches="tight"); plt.close(fig); return base64.b64encode(b.getvalue()).decode()
figs={}
f,ax=plt.subplots(figsize=(5,3.5)); y.value_counts().sort_index().plot.bar(ax=ax,color="#4C72B0"); ax.set_title("Class distribution (classe)"); ax.set_ylabel("count"); plt.xticks(rotation=0); figs["classes"]=b64(f)
f,ax=plt.subplots(figsize=(6,4)); imp.head(15)[::-1].plot.barh(ax=ax,color="#55A868"); ax.set_title("Random forest: top 15 variable importances"); figs["imp"]=b64(f)
f,ax=plt.subplots(figsize=(4.5,4)); ax.imshow(cm,cmap="Blues"); ax.set_xticks(range(5)); ax.set_yticks(range(5)); ax.set_xticklabels("ABCDE"); ax.set_yticklabels("ABCDE"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Hold-out confusion matrix")
for i in range(5):
    for j in range(5): ax.text(j,i,cm[i,j],ha="center",va="center",color="white" if cm[i,j]>cm.max()/2 else "black",fontsize=8)
figs["cm"]=b64(f)
f,ax=plt.subplots(figsize=(6,3.5)); ax.bar(list(cv.keys()),list(cv.values()),color=["#C44E52","#8172B2","#55A868"]); ax.set_ylim(0.5,1); ax.set_ylabel("5-fold CV accuracy"); ax.set_title("Model comparison"); plt.xticks(rotation=10,fontsize=8); figs["cmp"]=b64(f)
# final model on all training
final=RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=1).fit(X,y)
pred=final.predict(te[feat]); res["predictions"]=list(pred)
res["oob"]=None
oobm=RandomForestClassifier(n_estimators=500,n_jobs=-1,random_state=1,oob_score=True).fit(X,y); res["oob"]=oobm.oob_score_
json.dump({"res":res,"figs":figs},open("results.json","w"),default=lambda o:o.item() if hasattr(o,"item") else str(o))
print(json.dumps(res,default=str,indent=1))
