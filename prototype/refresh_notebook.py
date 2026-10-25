"""Execute the six demo code cells and keep their real outputs in the notebook."""
import contextlib
import io
import json
import os
from pathlib import Path


def refresh():
    root=Path(__file__).resolve().parent
    path=root/"demo.ipynb"
    if not path.exists(): return
    nb=json.loads(path.read_text()); namespace={}; count=0
    previous=Path.cwd()
    try:
        os.chdir(root)
        for cell in nb["cells"]:
            if cell["cell_type"]!="code": continue
            source=''.join(cell["source"]).replace(
                'client = data.sort_values("risk_any", ascending=False).iloc[:1].copy()',
                'client = data[data.offer.gt(0)].sort_values("risk_any", ascending=False).iloc[:1].copy()')
            cell["source"]=source.splitlines(True)
            count+=1; buf=io.StringIO()
            with contextlib.redirect_stdout(buf):
                exec(compile(source,f'demo-cell-{count}','exec'),namespace)
            cell["execution_count"]=count
            cell["outputs"]=[{"output_type":"stream","name":"stdout","text":buf.getvalue().splitlines(True)}]
        path.write_text(json.dumps(nb,ensure_ascii=False,indent=1))
    finally:
        os.chdir(previous)
    return count


if __name__=="__main__": print("Executed demo cells:",refresh())
