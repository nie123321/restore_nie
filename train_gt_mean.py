"""Train A+bottom MDTA with GT-Mean L1 (sigma=0.1), then test best raw val L1."""
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parent
NAME = 'a_mdta_gtmean_l1_sigma01_whole_b8_e55_seed100_20260928'
RUN = ROOT / 'runs' / NAME
REFERENCE = ROOT / 'runs' / 'a_mdta_bottleneck_whole_b8_e55_seed100_20260928' / 'config.json'


def main():
    if RUN.exists() and any(RUN.iterdir()):
        raise RuntimeError(f'Run directory is not empty: {RUN}')
    config = json.loads(REFERENCE.read_text(encoding='utf-8'))
    assert config['model']['bottleneck_attention'] == 'mdta'
    assert config['crop_size'] is None and not config['flip_hv']
    command = [sys.executable, '-u', '-X', 'utf8', str(ROOT / 'run_demo.py'),
               'train', '--run-dir', str(RUN), '--data-root', config['data_root']]
    keys = {'epochs': 'epochs', 'batch-size': 'batch_size', 'seed': 'seed',
            'workers': 'workers', 'lr': 'lr', 'lr-schedule': 'lr_schedule',
            'min-lr': 'min_lr', 'weight-decay': 'weight_decay',
            'grad-clip': 'grad_clip', 'best-metric': 'best_metric',
            'val-every': 'val_every', 'save-every': 'save_every'}
    for flag, key in keys.items():
        command.extend(['--' + flag, str(config[key])])
    for key in ('width', 'spectral_mode', 'output_mode'):
        command.extend(['--' + key.replace('_', '-'), str(config['model'][key])])
    command.extend(['--bottleneck-attention', 'mdta', '--half-encoder-frequency', 'none',
                    '--half-decoder', 'gated', '--fusion-mode', 'gated',
                    '--loss-mode', 'gt-mean-l1', '--gt-mean-sigma', '0.1', '--device', 'cuda',
                    '--amp' if config['amp'] else '--no-amp', '--log-every', '50'])
    print(json.dumps({'phase': 'training', 'reference': str(REFERENCE),
                      'initialization': 'from scratch; no checkpoint warm start',
                      'command': command}, ensure_ascii=False), flush=True)
    result_path = RUN / 'experiment_result.json'
    result = {'reference': str(REFERENCE), 'loss_mode': 'gt-mean-l1', 'sigma': 0.1}
    try:
        subprocess.run(command, check=True)
        best = RUN / 'best_val.pt'
        checkpoint = torch.load(best, map_location='cpu', weights_only=False)
        step = checkpoint['step']
        del checkpoint
        output = RUN / f'test_best{step}_20260928'
        result.update(state='testing', best_step=step, output=str(output))
        result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        subprocess.run([sys.executable, '-u', '-X', 'utf8', str(ROOT / 'test_best.py'),
                        '--checkpoint', str(best), '--data-root', config['data_root'],
                        '--output-dir', str(output), '--method-label',
                        'A-MDTA-GTMeanL1-sigma0.1-55ep-test', '--device', 'cuda'], check=True)
        result.update(state='complete', updated_at=datetime.now().isoformat())
        result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print('A_MDTA_GT_MEAN_TRAIN_AND_TEST_COMPLETE', flush=True)
    except Exception as error:
        if RUN.exists():
            result.update(state='failed', error=str(error), updated_at=datetime.now().isoformat())
            result_path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
