import csv, json, platform, datetime
from pathlib import Path
import numpy as np

CODE = Path(__file__).resolve().parents[2]
DATA = CODE / 'results' / 'data'
LOGS = CODE / 'results' / 'logs'
DX, R_FAIL, DELTA_MIN = 0.05, 0.5, 0.15

def rd(n):
    with open(DATA / n, encoding='utf-8') as fh:
        return list(csv.DictReader(fh))

def tr(rows, line, m, n):
    for r in rows:
        if r['line'] == line and r['method'] == m and int(r['gap_traces']) == n:
            return float(r['R'])
    return None

def w50(p):
    j = json.loads(Path(p).read_text(encoding='utf-8'))
    L = j['lines']
    if isinstance(L, dict):
        return {n: {k: float(v) for k, v in e['W50_m'].items() if v is not None} for n, e in L.items()}
    return {e['line']: {k: float(v) for k, v in e['W50_m'].items() if v is not None} for e in L}

def main():
    kr, ar, cr = rd('field_R_guarded.csv'), rd('field_R_aquileia_full.csv'), rd('field_R_offtarget.csv')
    kw = w50(LOGS / 'field_analysis_karst_guarded.json')
    aw = w50(LOGS / 'field_analysis_aquileia_full.json')

    on = {}
    for tbl, ws, src in ((kr, kw, 'field_R_guarded.csv'), (ar, aw, 'field_R_aquileia_full.csv')):
        for name, w in ws.items():
            for m in ('zero', 'linear'):
                if m not in w:
                    continue
                n = int(round(w[m] / DX))
                r = tr(tbl, name, m, n)
                if r is not None:
                    on[(name, m)] = {'W50': w[m], 'n': n, 'R': r, 'source': src}

    off = {}
    for r in cr:
        if abs(float(r['offset_m'])) < 1e-9:
            continue
        off.setdefault((r['line'], r['method']), {})[float(r['offset_m'])] = float(r['R'])

    checks = {}
    for m in ('zero', 'linear'):
        checks[m] = []
        for line in sorted({k[0] for k in off if k[1] == m}):
            if (line, m) not in on:
                print('  note:', line, m, 'censored in the table, no on-target value')
                continue
            vals = [off[(line, m)][o] for o in (1.0, -1.0) if o in off[(line, m)]]
            base = on[(line, m)]['R']
            checks[m].append({'line': line, 'R_on_table': base, 'R_at_1m': vals,
                              'min_R_at_1m': min(vals) if vals else None,
                              'delta_min': min(v - base for v in vals) if vals else None,
                              'passes_min_R': bool(vals) and min(vals) >= R_FAIL,
                              'passes_delta': bool(vals) and min(vals) - base >= DELTA_MIN})

    pool = {'zero': {}, 'linear': {}}
    for m in ('zero', 'linear'):
        for o in (1.0, 2.0):
            vals = [v[o] for k, v in off.items() if k[1] == m and o in v]
            pool[m]['%.1f' % o] = {'mean': float(np.mean(vals)), 'n': len(vals),
                                   'min': float(min(vals)), 'max': float(max(vals))}

    pred = {
        'P5': {'rule': 'at |offset| = 1.0 m zero fill keeps R >= 0.5 on every profile and exceeds the on-target value by at least 0.15',
               'met': all(e['passes_min_R'] and e['passes_delta'] for e in checks['zero'])},
        'P6': {'rule': 'profile-mean R rises from 1.0 m to 2.0 m',
               'mean_1m': pool['zero']['1.0']['mean'], 'mean_2m': pool['zero']['2.0']['mean'],
               'met': pool['zero']['2.0']['mean'] > pool['zero']['1.0']['mean']},
        'P7': {'rule': 'linear fill keeps R >= 0.5 at 1.0 m on every profile',
               'met': all(e['passes_min_R'] for e in checks['linear'])},
    }

    payload = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'script': 'scripts/field_control_summarise.py',
               'on_target_value_source': 'frozen tables, not recomputed',
               'constants': {'DX': DX, 'R_FAIL': R_FAIL, 'DELTA_MIN': DELTA_MIN},
               'on_target': {a + '|' + b: v for (a, b), v in sorted(on.items())},
               'checks': checks, 'pooled': pool, 'predictions': pred,
               'numpy': np.__version__, 'python': platform.python_version()}
    out = LOGS / 'field_control_summarise.json'
    out.write_text(json.dumps(payload, indent=1), encoding='utf-8')

    print('profile      method  R_on(table)  R_at_1m')
    for m in ('zero', 'linear'):
        for e in checks[m]:
            print('  %-11s %-7s %.4f       %s' % (e['line'], m, e['R_on_table'],
                  ', '.join('%.4f' % v for v in e['R_at_1m'])))
    print('pooled zero: 1 m %.4f (n=%d)  2 m %.4f (n=%d)' % (
        pool['zero']['1.0']['mean'], pool['zero']['1.0']['n'],
        pool['zero']['2.0']['mean'], pool['zero']['2.0']['n']))
    for k, v in pred.items():
        print(k, 'met:', v['met'])
    print('written:', out)

if __name__ == '__main__':
    main()