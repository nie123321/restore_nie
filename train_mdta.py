"""Train A+MDTA with the A55 recipe and evaluate its best checkpoint."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import torch
ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs'/'a_mdta_bottleneck_whole_b8_e55_seed100_20260928'
REFERENCE = ROOT/'runs'/'a_whole224x448_b8_e55_seed100_20260927'/'config.json'

def main():
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError(f'Run directory is not empty: {RUN}')
    c=json.loads(REFERENCE.read_text(encoding='utf-8'))
    cmd=[sys.executable,'-u','-X','utf8',str(ROOT/'run_demo.py'),'train','--run-dir',str(RUN),'--data-root',c['data_root']]
    keys={'epochs':'epochs','batch-size':'batch_size','seed':'seed','workers':'workers','lr':'lr','lr-schedule':'lr_schedule','min-lr':'min_lr','weight-decay':'weight_decay','grad-clip':'grad_clip','best-metric':'best_metric','val-every':'val_every','save-every':'save_every'}
    for flag,key in keys.items(): cmd.extend(['--'+flag,str(c[key])])
    for key in ('width','spectral_mode','output_mode'): cmd.extend(['--'+key.replace('_','-'),str(c['model'][key])])
    cmd.extend(['--bottleneck-attention','mdta','--device','cuda','--amp','--log-every','50'])
    print(json.dumps({'phase':'training','reference':str(REFERENCE),'command':cmd},ensure_ascii=False),flush=True)
    try:
        subprocess.run(cmd,check=True)
        best=RUN/'best_val.pt'
        checkpoint=torch.load(best,map_location='cpu',weights_only=False)
        step=checkpoint['step']; del checkpoint
        out=RUN/f'test_best{step}_20260928'
        result={'state':'testing','best_step':step,'output':str(out)}
        result_path=RUN/'experiment_result.json'
        result_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        subprocess.run([sys.executable,'-u','-X','utf8',str(ROOT/'test_best.py'),'--checkpoint',str(best),'--data-root',c['data_root'],'--output-dir',str(out),'--method-label','A-MDTA-bottleneck-55ep-test','--device','cuda'],check=True)
        result.update(state='complete',updated_at=datetime.now().isoformat())
        result_path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print('A_MDTA_TRAIN_AND_TEST_COMPLETE',flush=True)
    except Exception as e:
        if RUN.exists():
            (RUN/'experiment_result.json').write_text(json.dumps({'state':'failed','error':str(e),'updated_at':datetime.now().isoformat()},indent=2)+'\n',encoding='utf-8')
        raise
if __name__=='__main__': main()
