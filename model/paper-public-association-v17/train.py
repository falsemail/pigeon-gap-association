"""Five pre-registered matched pilots. No reserve-test source is read."""
from pathlib import Path
import sys, os, json, time, argparse, hashlib, subprocess, datetime, importlib.util
import numpy as np
import torch
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'paper-public-association-v14'))
from visible_data import prepare, write
from kinematic_mixture import KinematicMixture, VARIANTS, inputs, auxiliary
from learners import objective
spec = importlib.util.spec_from_file_location('v17_metrics', ROOT / 'paper-public-association-v9/run.py')
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)
torch.set_num_threads(1)


def register():
    p = HERE / 'registration.json'
    if p.exists():
        return json.loads(p.read_text())
    d = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             hypothesis='Explicit joint position/velocity density and correlated uncertainty may improve long-gap association. Velocity already entered the existing encoders; no new sensor channel is added.',
             variants=VARIANTS, seed=11, epochs=25, batch=24,
             training='Same V14 frozen 7200 training, 612 validation, 687 opened development conditions; no resampling or new test access.',
             optimizer=dict(name='AdamW', lr=.0007, weight_decay=.0003, gradient_clip=3),
             loss='Same mean bidirectional conditional CE + 0.1 true-pair conditional density NLL. Same whole-batch masks. No supervised hidden trajectories.',
             selection='Same source-balanced record-macro validation F1, 11 null thresholds [-6,-4,-3,-2,-1,0,1,2,3,4,6], best of 25 epochs. Development never selects weights/threshold.',
             variables='Position relative to observed group median /20m; velocity relative to observed group median /5m/s; same observed heading frames and clipping as V14.',
             covariance='Four components. Diagonal variance .0025+softplus(z)^2. Full covariance L L^T+1e-6 I; diag(L)=sqrt(diagonal variance), offdiag(L)=0.5*tanh(z). Missing z/vz marginalized by covariance principal submatrix.',
             anchors='Alternating observed-frame-preserving and observed-heading-rotated relative states; velocity rotates in the joint variant. These are hypotheses, not a guarantee of hidden physical motion.',
             non_neural_control='global_joint_full: one global 4-component distribution and scalar null, no encoder/context-conditioned neural prediction. Same objective/optimizer/steps, learned parameters only. Do not call this an author implementation.',
             comparisons='2x2 ablation of scored variable (position vs position+velocity) and covariance (diagonal vs full); matched global non-neural control; all five results retained.',
             prior='Bishop1994 MDN principle; joint kinematic likelihood and Gaussian marginalization are established tools, not new theoretical claims. Our contribution remains a hypothesis requiring evidence.',
             limitations='Estimated velocities correlate with position noise; joint covariance does not prove correct calibration. Few independent dense development records. Single-seed pilots require replication before confirmation.',
             new_test='Huang2026 and Coordinated D1/D2 remain unscored.',
             files={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')},
             data_registration_sha256=hashlib.sha256((ROOT / 'paper-public-association-v14/registration.json').read_bytes()).hexdigest())
    write(p, d)
    return d


def train(name):
    reg = register()
    assert reg['variants'] == VARIANTS
    seed = reg['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, va = prepare('train'), prepare('validation')
    assert (len(tr), len(va)) == (7200, 612)
    model = KinematicMixture(**VARIANTS[name]).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=.0007, weight_decay=.0003)
    folder = HERE / 'models'
    folder.mkdir(exist_ok=True)
    best, history, start = -1., [], time.time()
    for epoch in range(25):
        model.train()
        losses = []
        order = rng.permutation(len(tr))
        for off in range(0, len(tr), 24):
            s = inputs([tr[j] for j in order[off:off + 24]], 'cuda', True)
            opt.zero_grad(set_to_none=True)
            o = model(s)
            loss = objective(o, s) + .1 * auxiliary(o, s)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite loss: {name} epoch {epoch} offset {off}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 3, error_if_nonfinite=True)
            opt.step()
            losses.append(float(loss.detach()))
        val = metrics.calibrate(va, metrics.scores(model, va, dict(corrected=True)))
        row = dict(epoch=epoch + 1, loss=float(np.mean(losses)), validation=val, wall_seconds=time.time() - start)
        history.append(row)
        if val['source_balanced']['f1'] > best:
            best = val['source_balanced']['f1']
            torch.save(dict(state=model.state_dict(), config=VARIANTS[name], variant=name, seed=seed, epoch=epoch + 1, validation=val), folder / f'{name}-{seed}.pt')
        write(folder / f'{name}-{seed}-history.json', history)
        print(name, epoch + 1, val['source_balanced'], flush=True)
    # Opened development only after checkpoint/threshold are frozen.
    ck = torch.load(folder / f'{name}-{seed}.pt', weights_only=False)
    model.load_state_dict(ck['state'])
    de = prepare('development')
    assert len(de) == 687
    out, rows = metrics.evaluate(de, metrics.scores(model, de, dict(corrected=True)), ck['validation']['shift'])
    write(HERE / f'{name}-{seed}-development.json', dict(config=VARIANTS[name], validation=ck['validation'], checkpoint_epoch=ck['epoch'], development=out, cases=[dict(source=q['source'], record=q.get('cluster', q['record']), gap=q['sample']['meta']['gap'], metrics=r) for q, r in zip(de, rows)], parameters=sum(p.numel() for p in model.parameters()), training_wall_seconds=time.time() - start, scope='Registered matched redevelopment pilot; concurrent wall times are not isolated inference costs. No new reserve opened.'))


def queue():
    register()
    verification = json.loads((HERE / 'verification.json').read_text())
    assert verification['all_passed']
    if (HERE / 'progress.json').exists():
        raise RuntimeError('Existing progress: inspect before resuming; no automatic rerun')
    logs = HERE / 'logs'
    logs.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    done = []
    # One additional worker, sequential variants. Existing author jobs continue.
    for name in VARIANTS:
        with (logs / f'{name}-11.log').open('w') as f:
            r = subprocess.run([sys.executable, '-u', __file__, '--variant', name], env=env, stdout=f, stderr=subprocess.STDOUT)
        done.append(dict(variant=name, returncode=r.returncode))
        write(HERE / 'progress.json', done)
        print(done[-1], flush=True)
        if r.returncode:
            raise RuntimeError('Pilot failed; preserve evidence and inspect')
    write(HERE / 'completion.json', dict(status='five matched pilots complete', reserve_opened=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant', choices=list(VARIANTS))
    p.add_argument('--register', action='store_true')
    a = p.parse_args()
    if a.register:
        register()
    elif a.variant:
        train(a.variant)
    else:
        queue()
