
import onnx, json
from onnx import numpy_helper
MODELS = {
 "det_head": r"C:\Users\26671\lpr-harmony\models_ms\head\y5fu_320x_head_fp32.onnx",
 "rec_rpv3": r"C:\Users\26671\lpr-harmony\models_ms\rpv3\rpv3_mdict_160_r3.onnx",
 "cls_fp32": r"C:\Users\26671\lpr-harmony\models_ms\cls\litemodel_cls_96x_r1_fp32.onnx",
 "lprnet":   r"C:\Users\26671\lpr-harmony\models_ms\lprnet\lprnet.onnx",
}
res={}
for k,p in MODELS.items():
    try: m=onnx.load(p)
    except Exception as e:
        res[k]={"err":str(e)[:80]}; continue
    w={i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    tot=good=small=0; bad=[]
    for n in m.graph.node:
        if n.op_type!="Conv": continue
        if len(n.input)<2 or n.input[1] not in w: continue
        shp=w[n.input[1]].shape
        if len(shp)!=4: continue
        co,ci=shp[0],shp[1]
        tot+=1
        if ci%16==0 and co%16==0: good+=1
        elif ci<16 and co<16: small+=1
    ops={}
    for n in m.graph.node: ops[n.op_type]=ops.get(n.op_type,0)+1
    res[k]={"conv":tot,"both16":good,"bothSmall":small,"ops":ops}
print(json.dumps(res,indent=1))
